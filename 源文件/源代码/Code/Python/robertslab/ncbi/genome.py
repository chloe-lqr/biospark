from robertslab.ncbi import db

class Genome:
    def __init__(self, accession, genomeid, taxid, filename):
        self.accession=accession
        self.genomeid=genomeid
        self.taxid=taxid
        self.filename=filename
        
    def __str__(self):
        return self.accession

def getGenomes():
    ret=[]
    cursor = db.cursor()
    cursor.execute("SELECT accession,genomeid,taxid,filename FROM genomes_arch_bact WHERE filename != ''")
    while True:
        results=cursor.fetchone()
        if results == None:
            return ret
        ret.append(Genome(results[0],results[1],results[2],results[3]))
    

