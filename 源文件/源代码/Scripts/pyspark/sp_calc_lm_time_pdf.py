## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf3.sfile --skip_less_than=1.0 0 1 2 3 4 5 6 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf4*.sfile --skip_less_than=1e1 0 --interactive
## spark-submit --master yarn-client --num-executors 1 --driver-memory 4g --executor-memory 4g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf6.sfile --skip_less_than=1e1 0 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf7.sfile --skip_less_than=1e1 0 --interactive
## rm -rf pdf.dat && spark-submit --master yarn-client --num-executors 5 --driver-memory 4g --executor-memory 4g --executor-cores 4 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /tmp/test.sfile 0 --interactive
##
try:
    from pyspark import SparkConf, SparkContext
except:
    print "Warning: No spark libraries found. Ignore this message if you are running locally."
    from robertslab.spark.helper import SparkLocalGlobal
    from robertslab.sfile import SFile

from argparse import ArgumentParser, ArgumentTypeError
from ast import literal_eval as make_tuple
import numpy as np
import math
import os,sys
import re
import zlib

import robertslab.cellio as cellio
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import lm.io.SpeciesCounts_pb2 as SpeciesCounts_pb2
import lm.io.SpeciesTimeSeries_pb2 as SpeciesTimeSeries_pb2


# Global variables.
globalSpeciesToBin=None
globalTimeBinWidth=None


# Convert a string argument to a Python tuple.
def argtuple(s):
    t = make_tuple(s)
    if type(t) != type((0,)):
        raise ArgumentTypeError("Argument was not a valid tuple: %s"%s)
    return t

def main():

    # Parse the command line arguments.
    parser = ArgumentParser(description='A Biospark script to calculate a time-dependent PDF from LMES simulation data.')
    parser.add_argument('output_path',                              help='Local path for the location to save the output as an HDF5 file.')
    parser.add_argument('input_sfile',                              help='Hadoop path for the location of the input file.')
    parser.add_argument('time_bin_width', type=float,               help='The width of the time bins.')
    parser.add_argument('species', nargs='+', type=int,             help='The species for which to calculate the PDF.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--output-indices', type=argtuple,           help='If this argument is specified, the output file will be saved using the cellio naming convention with the specified indices. The indices must be a formatted as a Python tuple, e.g., \"(1,2,3)\". The output_file argument should be a directory in this case.')
    group.add_argument('--using-indices', type=argtuple, dest='output_indices', help='Synonym for --output-indices.')
    args = parser.parse_args()

    # Make sure the output file doesn't already exists.
    if cellio.exists(args.output_path, indices=args.output_indices, format="hdf5"):
        print "\nError: The specified output filename already exists. Remove the existing file before running the analysis again.\n"
        quit()

    # Configure Spark
    conf = SparkConf().setAppName("LM Time-Dependent PDF")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main_spark(sc, args.input_sfile, args.output_path, args.output_indices, args.species, args.time_bin_width)


def main_spark(sc, inputFilename, outputFilename, outputIndices, speciesToBin, timeBinWidth):

    global globalSpeciesToBin, globalTimeBinWidth

    # Broadcast the global variables.
    globalSpeciesToBin=sc.broadcast(speciesToBin)
    globalTimeBinWidth=sc.broadcast(timeBinWidth)

    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(inputFilename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Bin the species counts records and sum across all of the bins.
    results = allRecords.filter(filterRecordType).flatMap(binSpeciesCountsByTime).reduceByKeyLocally(addBins)

    # Find the min and max time bins and min and max species counts.
    minTimeBin=None
    maxTimeBin=None
    minSpeciesCount={}
    maxSpeciesCount={}
    for species,timeBin in results.keys():

        # Check the time bin.
        if minTimeBin is None or timeBin < minTimeBin:
            minTimeBin = timeBin
        if maxTimeBin is None or timeBin > maxTimeBin:
            maxTimeBin = timeBin

        # Check the species counts.
        minCount=results[(species,timeBin)][0]
        maxCount=results[(species,timeBin)][1]
        if species not in minSpeciesCount or minCount < minSpeciesCount[species]:
            minSpeciesCount[species] = minCount
        if species not in maxSpeciesCount or maxCount > maxSpeciesCount[species]:
            maxSpeciesCount[species] = maxCount

    # Create the pdf arrays.
    pdfs = {}
    for species in speciesToBin:
        pdfs["/%d/P"%(species)] = np.zeros((maxTimeBin-minTimeBin+1,maxSpeciesCount[species]-minSpeciesCount[species]+1))
        pdfs["/%d/T"%(species)] = (np.arange(minTimeBin,maxTimeBin+1).astype(float)*timeBinWidth)+(timeBinWidth/2)
        pdfs["/%d/N"%(species)] = np.arange(minSpeciesCount[species],maxSpeciesCount[species]+1).astype(float)

    # Convert the counts into pdfs.
    numberRecords = 0
    for species,timeBin in results.keys():
        minCount=results[(species,timeBin)][0]
        maxCount=results[(species,timeBin)][1]
        bins=results[(species,timeBin)][2]
        pdf = pdfs["/%d/P"%(species)]
        numberRecords = sum(bins)
        pdf[timeBin-minTimeBin,minCount-minSpeciesCount[species]:maxCount-minSpeciesCount[species]+1] = bins.astype(float)/float(numberRecords)

    # Output the pdfs.
    cellio.cellsave(outputFilename, pdfs, indices=outputIndices, format="hdf5")
    print "Saved time-dependent pdfs with %d time bins from %d records per bin to %s"%(maxTimeBin-minTimeBin+1,numberRecords,outputFilename)


def filterRecordType(record):

    # Extract the data.
    (name,dataType)=record[0]

    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.SpeciesCounts" or dataType == "protobuf:lm.io.SpeciesTimeSeries":
        return True
    return False


def binSpeciesCountsByTime(record):

    global globalSpeciesToBin, globalTimeBinWidth

    # Get the global variables.
    speciesToBin = globalSpeciesToBin.value
    timeBinWidth = globalTimeBinWidth.value

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # If this is a SpeciesCounts object, extract the data.
    if dataType == "protobuf:lm.io.SpeciesCounts":

        # Deserialize the data.
        obj=SpeciesCounts_pb2.SpeciesCounts()
        obj.ParseFromString(str(data))

        # If there are no entries, return an empty set.
        if obj.number_entries == 0:
            return []

        # Convert the data to a numpy array.
        species_counts=np.zeros((obj.number_entries*obj.number_species,),dtype=int)
        for i,val in enumerate(obj.species_count):
            species_counts[i] = val
        species_counts.shape=(obj.number_entries,obj.number_species)
        times=np.zeros((obj.number_entries,),dtype=float)
        for i,val in enumerate(obj.time):
            times[i] = val

    # If this is a SpeciesCounts object, extract the data.
    elif dataType == "protobuf:lm.io.SpeciesTimeSeries":

        # Deserialize the data.
        obj=SpeciesTimeSeries_pb2.SpeciesTimeSeries()
        obj.ParseFromString(str(data))

        # Make sure the data is consistent.
        if len(obj.counts.shape) != 2 or len(obj.times.shape) != 1:
            raise ValueError("Invalid array shape.")
        if obj.counts.shape[0] != obj.times.shape[0]:
            raise ValueError("Inconsistent array sizes.")
        if obj.counts.data_type != NDArray_pb2.NDArray.int32 or obj.times.data_type != NDArray_pb2.NDArray.float64:
            raise TypeError("Invalid array data types.")

        # If there are no entries, return an empty set.
        if obj.counts.shape[0] == 0:
            return []

        # Convert the data to a numpy array.
        if obj.counts.compressed_deflate:
            species_counts=np.reshape(np.fromstring(zlib.decompress(obj.counts.data), dtype=np.int32), obj.counts.shape)
        else:
            species_counts=np.reshape(np.fromstring(obj.counts.data, dtype=np.int32), obj.counts.shape)
        if obj.times.compressed_deflate:
            times=np.reshape(np.fromstring(zlib.decompress(obj.times.data), dtype=np.float64), obj.times.shape)
        else:
            times=np.reshape(np.fromstring(obj.times.data, dtype=np.float64), obj.times.shape)

    else:
        raise TypeError("Invalid record, unknown data type.")

    # Find the min and max values.

    # Figure out the rows that are in each time bin.
    timeBins=[]
    firstRow={}
    lastRow={}
    for i in range(0,times.shape[0]):
        timeBin = int(math.floor(times[i]/timeBinWidth))
        if timeBin not in timeBins:
            timeBins.append(timeBin)
            firstRow[timeBin] = i
        lastRow[timeBin] = i

    # Go through each time bin and species pair and then bin the data.
    allRecords=[]
    for timeBin in timeBins:
        counts=species_counts[firstRow[timeBin]:lastRow[timeBin]+1,:]
        minCounts=counts.min(0)
        maxCounts=counts.max(0)
        for species in speciesToBin:
            bins,edges=np.histogram(counts[:,species],np.arange(minCounts[species],maxCounts[species]+2,dtype=int))
            allRecords.append(((species,timeBin),(minCounts[species],maxCounts[species],bins)))

    # Return the records.
    return allRecords

def addBins(data1, data2):

    # Make sure both records are formatted correctly.
    if len(data1) != 3: raise Exception("data1 record did not have the correct format: len=%d"%(len(data1)))
    if len(data2) != 3: raise Exception("data2 record did not have the correct format: len=%d"%(len(data2)))

    # Extract the data.
    (minCount1,maxCount1,bins1)=data1
    (minCount2,maxCount2,bins2)=data2

    # Find the range of the combined bins.
    minCount=min(minCount1,minCount2)
    maxCount=max(maxCount1,maxCount2)

    # If bins1 is already the correct size, use it.
    if minCount1 == minCount and maxCount1 == maxCount:
        bins=bins1
        bins[minCount2-minCount:minCount2-minCount+len(bins2)]+=bins2

    # If bins2 is already the correct size, use it.
    elif minCount2 == minCount and maxCount2 == maxCount:
        bins=bins2
        bins[minCount1-minCount:minCount1-minCount+len(bins1)]+=bins1

    # Otherwise create new bins.
    else:
        bins=np.zeros((maxCount-minCount+1,),dtype=np.int)
        bins[minCount1-minCount:minCount1-minCount+len(bins1)]+=bins1
        bins[minCount2-minCount:minCount2-minCount+len(bins2)]+=bins2

    # Return the combined bins.
    return (minCount,maxCount,bins)

# Call the main function when the script is executed.
if __name__ == "__main__":
    main()