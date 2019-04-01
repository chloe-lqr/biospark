try:
    from pyspark import SparkConf, SparkContext
except:
    print "Warning: No spark libraries found. Ignore this message if you are running locally."
    from robertslab.spark.helper import SparkLocalGlobal
    from robertslab.sfile import SFile

from argparse import ArgumentParser, ArgumentTypeError
from ast import literal_eval as make_tuple
import numpy as np
import os,sys
import re
import zlib

import robertslab.cellio as cellio
import lm.io.FirstPassageTimes_pb2 as FirstPassageTimes_pb2

# Convert a string argument to a Python tuple.
def argtuple(s):
    t = make_tuple(s)
    if type(t) != type((0,)):
        raise ArgumentTypeError("Argument was not a valid tuple: %s"%s)
    return t


def main():

    # Parse the command line arguments.
    parser = ArgumentParser(description='A Biospark script to extract first passage times (FPTs) from LMES simulation data.')
    parser.add_argument('output_path',                              help='Local path for the location to save the output as an HDF5 file.')
    parser.add_argument('input_sfile',                              help='Hadoop path for the location of the input file.')
    parser.add_argument('-oi', '--output-indices', type=argtuple,   help='If this argument is specified, the output file will be saved using the cellio naming convention with the specified indices. The indices must be a formatted as a Python tuple, e.g., \"(1,2,3)\". The output_file argument should be a directory in this case.')
    args = parser.parse_args()

    # Make sure the output file doesn't already exists.
    if cellio.exists(args.output_path, indices=args.output_indices, format="hdf5"):
        print "\nError: The specified output filename already exists. Remove the existing file before running the analysis again.\n"
        quit()

    # Configure the spark context.
    conf = SparkConf().setAppName("LM First Passage Time")
    sc = SparkContext(conf=conf)

    # Execute the spark script.
    main_spark(sc, args.input_sfile, args.output_path, args.output_indices)


def main_spark(sc, inputFilename, outputFilename, outputIndices):
    
    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(inputFilename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Extract the records and cache the results.
    records = allRecords.filter(filterRecordType).map(extractRecord)
    records.cache()

    # Calculate the maximum count and number of trajectories for each species, these will be the matrix dimensions.
    dims = {}
    for dim in records.aggregateByKey((0,0),calculateDims1,calculateDims2).collect():
        dims[dim[0]] = dim[1]

    # Allocate the matrices.
    fpts = {}
    nextRows = {}
    for species in dims.keys():
        fpts["/FPT/%d"%species] = np.full(dims[species],np.nan,dtype=float)
        nextRows[species] = 0

    # Collect the data and fill in the matrices.
    for (species,(trajectory,minCount,maxCount,times)) in records.collect():
        row = nextRows[species]
        fpts["/FPT/%d"%species][row,minCount:maxCount+1] = times
        nextRows[species] += 1

    # Save the first passage times.
    cellio.cellsave(outputFilename, fpts, indices=outputIndices, format="hdf5")
    print "Saved FPTs from %d species to %s"%(len(fpts), outputFilename)


def filterRecordType(record):
    
    # Extract the data.
    (name,dataType)=record[0]
    
    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.FirstPassageTimes":
        return True            
    return False


def extractRecord(record):

    # Parse the record.
    (name,dataType)=record[0]
    data=record[1]

    # Deserialize the data.
    fpt=FirstPassageTimes_pb2.FirstPassageTimes()
    fpt.ParseFromString(str(data))

    # Convert the data to a numpy array.
    times=np.zeros((fpt.number_entries,),dtype=float)
    for i,val in enumerate(fpt.first_passage_time):
        times[i] = val

    return (fpt.species,(fpt.trajectory_id,fpt.species_count[0],fpt.species_count[fpt.number_entries-1],times))

def calculateDims1(D, V):

    cols = max(D[1], V[2]+1)
    return (D[0]+1,cols)

def calculateDims2(D1, D2):

    cols = max(D1[1], D2[1])
    return (D1[0]+D2[0],cols)


# Call the main function when the script is executed.
if __name__ == "__main__":
    main()
