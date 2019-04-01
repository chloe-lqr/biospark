from robertslab.ncbi import db

class TaxonomyNode:
    def __init__(self, taxid, name, rank):
        self.taxid=taxid
        self.name=name
        self.rank=rank
        self.parent=None
        self.children=[]
        self.leafCount=1
        
    def __str__(self, level=0):
        s=""
        for i in range(0,level):
            s+="  "
        ret=s+"%s (%d)\n"%(self.name,self.leafCount)
        for child in self.children:
            ret+=child.__str__(level+1)
        return ret  #or results[3] == 131567

    def getLeaves(self):
        ret=[]
        if len(self.children) == 0:
            ret.append(self)
        else:
            for child in self.children:
                ret.extend(child.getLeaves())
        return ret
        
    def getAncestory(self):
        if parent == None:
            return [self]
        else:
            ret=parent.getAncestory()
            ret.append(self)
            return ret

def getTaxidFromGi(gi):
    cursor = db.cursor()
    cursor.execute("SELECT taxid FROM tax_gi_to_taxid WHERE gi=%s",gi)
    if cursor.rowcount == 1:
        results=cursor.fetchone()
        return results[0]
    else:
        return None
    
def getName(taxid):
    cursor = db.cursor()
    cursor.execute("SELECT name FROM tax_nodes NATURAL JOIN tax_names WHERE tax_nodes.taxid=%s AND tax_names.class='scientific name'",taxid)
    if cursor.rowcount == 1:
        results=cursor.fetchone()
        return results[0]
    else:
        return None

def getLineage(taxid, showHidden=True):
    cursor = db.cursor()
    rootNode=None
    while True:
        cursor.execute("SELECT taxid,rank,name,parent_taxid,hidden_node FROM tax_nodes NATURAL JOIN tax_names WHERE tax_nodes.taxid=%s AND tax_names.class='scientific name';",taxid)
        if cursor.rowcount == 1:
            results=cursor.fetchone()
            
            # Add the node to the lineage.
            if showHidden or results[4] == 0:
                newRootNode=TaxonomyNode(results[0],results[2],results[1])
                if rootNode != None:
                    newRootNode.children.append(rootNode)
                    rootNode.parent=newRootNode
                rootNode=newRootNode
            
            # If the parent is the same as the node, we are done.
            if results[0] == results[3]:
                break
                
            # Set the new node to be the parent and loop again.
            taxid=results[3]
            
        elif cursor.rowcount == 0:
            raise Exception('Unknown taxid',taxid)
        else:
            raise Exception('Invalid taxid',taxid)
    
    return rootNode
    
def mergeTrees(taxonomyTrees):
    
    root=taxonomyTrees[0]
    for treeIndex in range(1,len(taxonomyTrees)):
        tree=taxonomyTrees[treeIndex]
        
        # Make sure the trees have the same root.
        if tree.taxid != root.taxid:
            raise Exception('Taxonomy trees to merge must have the same root')
            
        root.leafCount += tree.leafCount
    
        # Go through each child in the tree.
        for child in tree.children:
            
            # Go through each child in the root.
            added=False
            for i in range(0,len(root.children)):
                rootChild=root.children[i]
                
                # If the child is an exact match, merge it with the root's child.
                if child.taxid == rootChild.taxid:
                    mergeTrees([rootChild,child])
                    added=True
                    break
                    
                # If the child is lexicographically less than the root child, this must be the place to insert it. 
                elif child.name < rootChild.name:
                    root.children.insert(i,child)
                    child.parent=root;
                    added=True
                    break

            # If we didn't add it, put it add the end.
            if not added:
                root.children.append(child)
            
    return root
