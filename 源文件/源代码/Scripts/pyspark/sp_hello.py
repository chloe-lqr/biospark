## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --conf spark.driver.maxResultSize=0 --num-executors 3 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_hello.py
##

from operator import add
from pyspark import SparkConf, SparkContext
import sys
import time

def tokenize(text):
    return text.split()

def main(sc):

    data=range(0,1000)
    rdd=sc.parallelize(data)
    rdd2=rdd.map(incrementRecord)
    rdd2.cache()
    values=rdd2.collect()
    print values

    values2=rdd2.reduce(sumRecords)
    print values2

    time.sleep(90)

    # Load the file.
    #records = sc.newAPIHadoopFile(sys.argv[1], "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Print the record list.
    #output = records.map(listmap).keys().collect()
    #for r in output:
    #    print "%s %s %d"%r


#def listmap(record):
#    (name,datatype)=record[0]
#    data=record[1]
#    return ((name,datatype,len(data)),"")

def incrementRecord(record):
    print "Hello world %d"%record
    return record+1

def sumRecords(record1, record2):
    print "Reducing %d and %d"%(record1,record2)
    return record1+record2

if __name__ == "__main__":
    
    print "Starting script"
    
    #if len(sys.argv) < 1:
    ##    print "Usage: sfile"
    #    quit()

    
    # Configure Spark
    conf = SparkConf().setAppName("Hello World")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)
