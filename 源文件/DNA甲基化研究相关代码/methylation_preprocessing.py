## Spark Application - execute with spark-submit

try:
    from pyspark import SparkConf, SparkContext
except:
    print
    "No spark libraries found, ensure you are running locally."

import os,sys,stat
import lm.io.Methylation_pb2 as Methylation_pb2
import numpy as np

# Global variables.
output_dir=""

def main_spark(sc, outputFileAddress, inputFileName):

    global output_dir

    # Broadcast the global variables.
    output_dir = sc.broadcast(outputFileAddress)

    # load the original file from HDFS
    # records = sc.textFile("/user/biospark/bsgs/test_data_bismark_bt2.deduplicated.txt")
    records = sc.textFile(inputFileName)

    # filter the alignment results of every read
    result_filter_part2 = records\
        .filter(lambda line: not line.startswith('@'))

    # extract 1,3,4,6,14,15,16 columns
    # convert original information to protobuf format
    # serialize and output every reads into a file
    result_extract_information = result_filter_part2 \
        .map(extract_alignment_information)\
        .map(convert_to_protobuf)\
        .reduceByKeyLocally(lambda a, b: a + b)

    print "Methylation processing done successfully!"



def extract_alignment_information(record):
    result_in_list=record.split('\t')
    new_record=[]
    new_record.append(result_in_list[0])
    new_record.append(result_in_list[2])
    new_record.append(result_in_list[3])
    new_record.append(result_in_list[5])
    new_record.append(result_in_list[13])
    new_record.append(result_in_list[14])
    new_record.append(result_in_list[15])
    return new_record

#make the record output file and write the serialized alignment information in protobuf format into the file
def output_to_file(filename,protobuf_record):

    global output_dir

    #output_dir="/home/biospark/Tutorials/bsgs_tut_files/methylation/data_output"
    #output_dir_copy=output_dir
	
    data=protobuf_record
    path=output_dir.value+'/'+filename
    #os.mknod(filename)
    #modify the permissions
    '''
    os.chmod(output_dir, stat.S_IRWXO)
    os.chmod(output_dir, stat.S_IRWXU)
    os.chmod(output_dir, stat.S_IRWXG)
    '''
    #open a file by WRITE mode, if this file doesn't exist, make a new file
    fp = open(path,"wb")
    fp.write(data.SerializeToString())
    fp.close()
    print "write "+filename+" successfully!"


def convert_to_protobuf(record):
    #transfer unicode to string or int
    filename=record[0].encode("utf-8")
    protobuf_record=Methylation_pb2.Methylation()
    protobuf_record.rname=record[1].encode("utf-8")
    protobuf_record.pos=int(record[2].encode("utf-8"))
    protobuf_record.cigar=record[3].encode("utf-8")
    protobuf_record.xm=record[4].encode("utf-8")
    protobuf_record.xr=record[5].encode("utf-8")
    protobuf_record.xg=record[6] .encode("utf-8")
    #make a file, write data
    output_to_file(filename,protobuf_record)
    return (("successful",1))




if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Error: the number of arguments should be two, the outputAddress and the inputFile "
        quit()

    outputFileAddress = sys.argv[1]  ## the address of the local output file to create
    inputFileName = sys.argv[2]  ## the input SFile in HDFS

    # Execute Main functionality
    conf = SparkConf().setAppName("Methylation_Preprocessing")
    sc = SparkContext(conf=conf)
    main_spark(sc, outputFileAddress, inputFileName)

