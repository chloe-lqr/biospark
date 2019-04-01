## Spark Application - execute with spark-submit, e.g.
##

from pyspark import SparkConf, SparkContext
import sys

from robertslab.spark.helper import newAPIHadoopFileConfig

def tokenize(text):
    return text.split()

def main(sc):

    # Load the file.
    records = sc.newAPIHadoopFile(sys.argv[1], "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter", conf=newAPIHadoopFileConfig())

    # Print the record count.
    output = records.map(listmap).keys().count()
    print "Read %d sfile records."%(output)


def listmap(record):
    (name,datatype)=record[0]
    data=record[1]
    return (name,"")

if __name__ == "__main__":
    
    print "Starting script"
    
    if len(sys.argv) < 1:
        print "Usage: sfile"
        quit()

    
    # Configure Spark
    conf = SparkConf().setAppName("SFile Read Throughput")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc)
