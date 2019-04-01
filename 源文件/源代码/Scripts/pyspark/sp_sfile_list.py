## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --conf spark.driver.maxResultSize=0 --num-executors 4 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_sfile_list.py /user/erober32/tmp/pdf3.sfile
##

from operator import add
from pyspark import SparkConf, SparkContext
import sys

def tokenize(text):
    return text.split()

def main(sc):

    # Load the file.
    records = sc.newAPIHadoopFile(sys.argv[1], "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Print the record list.
    output = records.map(listmap).keys().collect()
    for r in output:
        print "%s %s %d"%r


def listmap(record):
    (name,datatype)=record[0]
    data=record[1]
    return ((name,datatype,len(data)),"")

if __name__ == "__main__":
    
    print "Starting script"
    
    if len(sys.argv) < 1:
        print "Usage: sfile"
        quit()

    
    # Configure Spark
    conf = SparkConf().setAppName("List SFile")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)
