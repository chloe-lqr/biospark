from robertslab.rawdata import db

class Dataset:
    def __init__(self, type):
        self.id=-1
        self.type=type
        self.properties={}
        self.files=[]
        
    def __init__(self, id, type):
        self.id=id
        self.type=type
        self.properties={}
        self.files=[]
        
    def __str__(self):
        ret="%d"%(id)
        return ret

class DatasetFile:
    def __init__(self, filename, mimetype):
        self.filename=filename
        self.mimetype=mimetype


def loadDataset(id):
    id=int(id)
    cursor = db.cursor()
    cursor.execute("SELECT ds_id,ds_type FROM datasets WHERE ds_id=%d AND deleted=FALSE",id)
    if cursor.rowcount == 1:
        results=cursor.fetchone()
        dataset=Dataset(results[0], results[1])
        
        # Load the properties.
        cursor.execute("SELECT name,value FROM dataset_properties WHERE ds_id=%d",id)
        while True:
            results=cursor.fetchone()
            if results == None:
                break
            dataset.properties[results[0]]=results[1]
        
        # Load the files.
        cursor.execute("SELECT filename,mimetype FROM dataset_files WHERE ds_id=%d AND deleted=FALSE SORT BY filename",id)
        while True:
            results=cursor.fetchone()
            if results == None:
                break
            dataset.files.append(DatasetFile(results[0],results[1]))
        
        return dataset
    else:
        return None
    
def saveDataset(dataset):
    
    cursor = db.cursor()
    
    # See if we are updating a dataset or adding a new one.
    if dataset.id == -1:
        
        # Get the next id to use.
        cursor.execute("SELECT MAX(ds_id) FROM datasets")
        if cursor.rowcount == 1:
            results=cursor.fetchone()
            dataset.id = int(results[0])+1
        else:
            raise Exception('Could not add new dataset.')
            
        # Insert the new dataset.
        cursor.execute("INSERT INTO datasets(ds_id,ds_type,add_date,add_user) VALUES(%d,%d,NOW,%s)",[dataset.id,dataset.type,user])
    else:
        # Update the existing dataset.
        cursor.execute("UPDATE datasets SET ds_type=%d,update_date=NOW,update_user=%s WHERE ds_id=%d",[dataset.type,user,dataset.id])
        
    # Update the properties.
    if name in dataset.properties:
        value=dataset.properties[name]
        
    
    # Update the files.
    for f in dataset.files:
        
        # See if we have a file with the same name.
        cursor.execute("SELECT mimetype,size,hashcode,deleted FROM dataset_files WHERE ds_id=%d AND filename=%s",[dataset.id,f.filename])
        if cursor.rowcount == 1:
            # Update the file, if necessary.
            results=cursor.fetchone()
            #if results[0] != f.mimetype or results[1] != f.size or results[2] != f.hashcode or results[3]==True:
                #cursor.execute("UPDATE datasets SET ds_type=%d,update_date=NOW,update_user=%s WHERE ds_id=%d",[dataset.type,user,dataset.id])
            
        #else:
            # Add the file.
        
    #cursor = db.cursor()
    #cursor.execute("SELECT name FROM tax_nodes NATURAL JOIN tax_names WHERE tax_nodes.taxid=%s AND tax_names.class='scientific name'",taxid)
    #if cursor.rowcount == 1:
    #    results=cursor.fetchone()
    #    return results[0]
    #else:
    #    return None
    
    # Commit the updates.
    db.commit()

