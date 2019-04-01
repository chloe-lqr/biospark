#!/usr/bin/python

NCBI_DIR="/home/databases/ncbi"
MYSQL_USER="ncbi"
MYSQL_PASSWORD="ncbi"

import glob
import re
import subprocess
import MySQLdb as mysql

from robertslab.ncbi import *

genomeExceptionMap={
'225603': '1266845',
'157907': '1091500',
'203334': '861365',
'225604': '1406863',
'184827': '1209989',
'222805': '1400053',
'206518': '1313290',
'225028': '1399116',
'225030': '1273548',
'162179': '1091501',
'218470': '104623',
'182935': '1216962',
'225602': '1183438',
'84383': '216599',
'13130': '820',
'218006': '1320309',
'225029': '1341692'
}

db = mysql.connect(host="xanthus.bph.jhu.edu", user=MYSQL_USER, passwd=MYSQL_PASSWORD, db="ncbi")
cursor1 = db.cursor()
cursor2 = db.cursor()
cursor3 = db.cursor()

i=0
j=0
k=0
l=0
for file in glob.glob(NCBI_DIR+"/genomes/Archaea_Bacteria/*/*.faa"):
    
    # Extract the genome id and the accession values.
    match = re.search('uid([0-9]+)\/', file)
    if match == None:
        print "Could not find uid for: "+file
        continue
    genomeid = match.group(1)
    match = re.search('\/(N[CST]_[0-9]+).faa', file)
    if match == None:
        print "Could not find accession for: "+file
        continue
    accession = match.group(1)
    
    # See if we can find a match in the database.
    matched=False
    cursor1.execute("SELECT accession FROM genomes_arch_bact WHERE genomeid=%s AND accession LIKE %s",[genomeid,"%s%%"%(accession)])
    if cursor1.rowcount == 1:
        results=cursor1.fetchone()
        accession = results[0]
        cursor2.execute("UPDATE genomes_arch_bact SET filename=%s WHERE accession=%s",[file,accession])
        matched=True
        i+=1
        
    # If we didn't find a match, try to get the taxonomy from the first sequence.
    if not matched:
        proc = subprocess.Popen(['head','-n1',file],stdout=subprocess.PIPE)
        match = re.search('gi\|([0-9]+)\|', proc.stdout.readline())
        if match != None:
            gi = match.group(1)
            taxid = taxonomy.getTaxidFromGi(gi)
            if taxid != None:
                name = taxonomy.getName(taxid)
                cursor3.execute("INSERT INTO genomes_arch_bact(accession,replicon,genomeid,taxid,len,name,filename) VALUES (%s,'unknown',%s,%s,0,%s,%s)",[accession,genomeid,taxid,name,file])
                matched=True
                j+=1
                
    # If we didn't find a match, try to exception list.
    if not matched:
        if genomeid in genomeExceptionMap:
            taxid = genomeExceptionMap[genomeid]
            if taxid != None:
                name = taxonomy.getName(taxid)
                cursor3.execute("INSERT INTO genomes_arch_bact(accession,replicon,genomeid,taxid,len,name,filename) VALUES (%s,'unknown',%s,%s,0,%s,%s)",[accession,genomeid,taxid,name,file])
                matched=True
                k+=1

    # If we didn't find a match, print a message
    if not matched:
        print "No match found for %s: %s,%s"%(file,genomeid,accession)
        l+=1
        
db.commit()
print "Matched from summary: %d, matched from sequence: %d, matched from exception list: %d, failed: %d"%(i,j,k,l)







