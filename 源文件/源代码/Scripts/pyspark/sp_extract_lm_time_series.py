## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf3.sfile --skip_less_than=1.0 0 1 2 3 4 5 6 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf4*.sfile --skip_less_than=1e1 0 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf6.sfile --skip_less_than=1e1 0 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf7.sfile --skip_less_than=1e1 0 --interactive
##
try:
    from pyspark import SparkConf, SparkContext
except:
    print "No spark libraries found, ensure you are running locally."
    from robertslab.spark.helper import SparkLocalGlobal
    from robertslab.sfile import SFile

from argparse import ArgumentParser
from ast import literal_eval as make_tuple
import cStringIO
import math
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import re
from scipy.stats import norm
import time
from timeit import default_timer as timer
import zlib

import robertslab.cellio as cellio
from robertslab.helper.sparkProfileHelper import SparkProfileDecorator
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import lm.io.SpeciesCounts_pb2 as SpeciesCounts_pb2
import lm.io.SpeciesTimeSeries_pb2 as SpeciesTimeSeries_pb2

# Global variables.
globalReplicates=None

def main_spark(sc, outputFilename, outputIndices, filename, replicates, interactive):
    
    global globalReplicates
    
    # Broadcast the global variables.
    globalReplicates=sc.broadcast(replicates)
    
    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(filename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Bin the species counts records and sum across all of the bins.
    results = allRecords.filter(filterReplicatesAndRecordType).map(extractTimeSeries).reduceByKeyLocally(combineTimeSeries)

    # Combine the records.
    combinedRecords = combineArrays(results)

    # Save the records.
    saveRecords(outputFilename, outputIndices, combinedRecords)


def main_local(outputFilename, outputIndices, filename, replicates, interactive):

    global globalReplicates

    # Broadcast the global variables.
    globalReplicates=SparkLocalGlobal(replicates)

    # Open the file.
    fp = SFile.fromFilename(filename, "r")

    # Loop through the records.
    reducedRecords = {}
    while True:
        rawRecord = fp.readNextRecord()
        if rawRecord is None: break

        # Filter the record.
        sparkRecord = [(rawRecord.name,rawRecord.dataType)]
        if filterReplicatesAndRecordType(sparkRecord):
            sparkRecord.append(fp.readData(rawRecord.dataSize))
            mapRecord = extractTimeSeries(sparkRecord)
            replicate = mapRecord[0]
            if replicate not in reducedRecords: reducedRecords[replicate] = ([],[])
            reducedRecords[replicate] = combineTimeSeries(reducedRecords[replicate], mapRecord[1])
        else:
            fp.skipData(rawRecord.dataSize)

    combinedRecords = combineArrays(reducedRecords)
    saveRecords(outputFilename, outputIndices, combinedRecords)


def filterReplicatesAndRecordType(record):

    global globalReplicates

    # Get the global variables.
    replicates = globalReplicates.value

    # Extract the data.
    (name,dataType)=record[0]

    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.SpeciesCounts":
        m=re.match("(.*)/Simulations/(\d+)/SpeciesCounts",name)
        if m is not None and int(m.group(2)) in replicates:
            return True
    elif dataType == "protobuf:lm.io.SpeciesTimeSeries":
        m=re.match("(.*)/Simulations/(\d+)/SpeciesTimeSeries",name)
        if m is not None and int(m.group(2)) in replicates:
            return True
    return False
    

def extractTimeSeries(record):


    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # If this is a SpeciesCounts object, extract the data.
    if dataType == "protobuf:lm.io.SpeciesCounts":
        m=re.match("(.*)/Simulations/(\d+)/SpeciesCounts",name)
        if m is None:
            raise ValueError("Invalid record, no replicate number.")
        replicateNumber=int(m.group(2))
        
        # Deserialize the data.
        obj=SpeciesCounts_pb2.SpeciesCounts()
        obj.ParseFromString(str(data))
            
        # If there are no entries, return an empty set.
        if obj.number_entries == 0:
            return (replicateNumber,([],[]))
        
        # Convert the data to a numpy array.
        species_counts=np.zeros((obj.number_entries*obj.number_species,),dtype=int)
        for i,val in enumerate(obj.species_count):
            species_counts[i] = val
        species_counts.shape=(obj.number_entries,obj.number_species)
        times=np.zeros((obj.number_entries,),dtype=float)
        for i,val in enumerate(obj.time):
            times[i] = val

        return (replicateNumber,([times],[species_counts]))

    # If this is a SpeciesTimeSeries object, extract the data.
    elif dataType == "protobuf:lm.io.SpeciesTimeSeries":
        m=re.match("(.*)/Simulations/(\d+)/SpeciesTimeSeries",name)
        if m == None:
            raise ValueError("Invalid record, no replicate number.")
        replicateNumber=int(m.group(2))
    
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
            return (replicateNumber,([],[]))
    
        # Convert the data to a numpy array.
        if obj.counts.compressed_deflate:
            species_counts=np.reshape(np.fromstring(zlib.decompress(obj.counts.data), dtype=np.int32), obj.counts.shape)
        else:
            species_counts=np.reshape(np.fromstring(obj.counts.data, dtype=np.int32), obj.counts.shape)
        if obj.times.compressed_deflate:
            times=np.reshape(np.fromstring(zlib.decompress(obj.times.data), dtype=np.float64), obj.times.shape)
        else:
            times=np.reshape(np.fromstring(obj.times.data, dtype=np.float64), obj.times.shape)

        return (replicateNumber,([times],[species_counts]))

    raise TypeError("Invalid record, unknown data type.")


def combineTimeSeries(data1, data2):

    # Make sure we have the correct records.
    if data1 is None: return data2
    if data2 is None: return data1
    if len(data1) != 2 or not isinstance(data1[0],list) or not isinstance(data1[1],list):
        raise TypeError("Invalid type for record 1 (%s) during combine time series"%(type(data1[0])))
    if len(data2) != 2 or not isinstance(data2[0],list) or not isinstance(data2[1],list):
        raise TypeError("Invalid type for record 2 (%s) during combine time series"%(type(data2[0])))

    # Get the lists.
    accumulator_times = data1[0]
    accumulator_counts = data1[1]
    new_times = data2[0]
    new_counts = data2[1]

    # Find the position where the new records should be added in the list.
    for i in range(0, len(new_times)):
        added = False
        for j in range(0, len(accumulator_times)):
            if new_times[i][-1] < accumulator_times[j][0]:
                accumulator_times.insert(j, new_times[i])
                accumulator_counts.insert(j, new_counts[i])
                added = True
                break
        if not added:
            accumulator_times.append(new_times[i])
            accumulator_counts.append(new_counts[i])
    return (accumulator_times,accumulator_counts)


def combineArrays(reducedRecords):

    combinedRecords={}
    for replicate in reducedRecords.keys():
        (times,counts)=reducedRecords[replicate]
        numberRows = 0
        for time in times:
            numberRows += time.shape[0]
        allTimes = np.zeros((numberRows))
        allCounts = np.zeros((numberRows,counts[0].shape[1]))
        start=0
        end=0
        for i in range(0, len(times)):
            end += times[i].shape[0]
            allTimes[start:end] = times[i]
            allCounts[start:end,:] = counts[i]
            start += times[i].shape[0]
        combinedRecords["/%d/Times"%(replicate)] = allTimes
        combinedRecords["/%d/Counts"%(replicate)] = allCounts

    return combinedRecords


def saveRecords(outputFilename, outputIndices, records):

    cellio.cellsave(outputFilename, records, indices=outputIndices, format="hdf5")
    print "Saved %d data sets to %s"%(len(records),outputFilename)



if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        print "Usage: output_filename hdfs_sfile min_replicate max_replicate [--using-indices=output_indices] [--local] [--interactive]"
        quit()

    outputFilename = sys.argv[1]
    outputIndices = None
    filename = sys.argv[2]
    replicates = range(int(sys.argv[3]),int(sys.argv[4])+1)
    interactive = False
    local = False
    skipNextArg=False
    for i in range(5,len(sys.argv)):

        # Skip this arg, if necessary.
        if skipNextArg:
            skipNextArg=False
            continue

        if sys.argv[i] == "--using-indices":
            outputIndices = make_tuple(sys.argv[i+1])
            skipNextArg=True
            continue

        m=re.match(r'--using-indices=(.+)',sys.argv[i])
        if m != None:
            outputIndices = make_tuple(m.group(1))
            continue

        if sys.argv[i] == "--interactive":
            interactive=True
            continue

        if sys.argv[i] == "--local":
            local=True
            continue

    # Make sure the output indices were valid.
    if outputIndices is not None and type(outputIndices) != type((0,)):
        print "Error: output indices must be specified as a python formatted tuple, got: %s"%(type(outputIndices))
        quit()



        # Execute Main functionality
    if not local:
        conf = SparkConf().setAppName("LM Time Series")
        sc = SparkContext(conf=conf)
        main_spark(sc, outputFilename, outputIndices, filename, replicates, interactive)
    else:
        main_local(outputFilename, outputIndices, filename, replicates, interactive)
