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
import lm.io.LatticeTimeSeries_pb2 as LatticeTimeSeries_pb2

# Global variables.
globalReplicates=None

def main_spark(sc, outputFilename, outputIndices, filename, replicates, interactive):
    
    global globalSpeciesList, globalReplicates
    
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
    if dataType == "protobuf:lm.io.LatticeTimeSeries":
        m=re.match("/Simulations/(\d+)/LatticeTimeSeries",name)
        if m is not None and int(m.group(1)) in replicates:
            return True
    return False
    

def extractTimeSeries(record):

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]
    
    # If this is a SpeciesCounts object, extract the data.
    if dataType == "protobuf:lm.io.LatticeTimeSeries":
        m=re.match("/Simulations/(\d+)/LatticeTimeSeries",name)
        if m is None:
            raise ValueError("Invalid record, no replicate number.")
        replicateNumber=int(m.group(1))

        # Deserialize the data.
        obj=LatticeTimeSeries_pb2.LatticeTimeSeries()
        obj.ParseFromString(str(data))

        # If there are no entries, return an empty set.
        if obj.number_entries == 0:
            return (replicateNumber,([],[]))

        # See if this is a v1 lattice time series.
        if len(obj.v1_times) == obj.number_entries:

            # Make sure the data is consistent.
            if len(obj.lattices) != obj.number_entries:
                raise "Invalid array shape."

            # Convert the times to a numpy array.
            times=np.zeros((obj.number_entries,),dtype=float)
            for i,val in enumerate(obj.v1_times):
                times[i] = val

            # Create a tuple with the lattice shape.
            latticeShape=(obj.lattices[0].x_size,obj.lattices[0].y_size,obj.lattices[0].z_size,obj.lattices[0].particles_per_site)

            # Go through each lattice.
            latticeList=[]
            for lattice in obj.lattices:

                # Convert the data to a numpy array.
                if lattice.v1_particles_compressed_deflate:
                    latticeData=np.reshape(np.fromstring(zlib.decompress(lattice.v1_particles), dtype=np.uint8), latticeShape)
                else:
                    latticeData=np.reshape(np.fromstring(lattice.v1_particles, dtype=np.uint8), latticeShape)

                latticeList.append(latticeData)

        return (replicateNumber,([times],[latticeList]))

    raise TypeError("Invalid record, unknown data type.")


def combineTimeSeries(data1, data2):

    # Make sure we have some new data.
    if data1 is None: return data2
    if data2 is None: return data1
    if len(data1) != 2 or not isinstance(data1[0],list) or not isinstance(data1[1],list):
        raise TypeError("Invalid type for record 1 (%s) during combine time series"%(type(data1[0])))
    if len(data2) != 2 or not isinstance(data2[0],list) or not isinstance(data2[1],list):
        raise TypeError("Invalid type for record 2 (%s) during combine time series"%(type(data2[0])))

    # Get the lists.
    accumulator_times = data1[0]
    accumulator_lattices = data1[1]
    new_times = data2[0]
    new_lattices = data2[1]

    # Find the position where the new records should be added in the list.
    for i in range(0, len(new_times)):
        added = False
        for j in range(0, len(accumulator_times)):
            if new_times[i][-1] < accumulator_times[j][0]:
                accumulator_times.insert(j, new_times[i])
                accumulator_lattices.insert(j, new_lattices[i])
                added = True
                break
        if not added:
            accumulator_times.append(new_times[i])
            accumulator_lattices.append(new_lattices[i])
    return (accumulator_times,accumulator_lattices)

def combineArrays(reducedRecords):

    combinedRecords={}
    for replicate in reducedRecords.keys():
        (times,latticeLists)=reducedRecords[replicate]
        numberRows = 0
        for time in times:
            numberRows += time.shape[0]
        allTimes = np.zeros((numberRows,))
        start=0
        end=0
        for i in range(0, len(times)):
            end += times[i].shape[0]
            allTimes[start:end] = times[i]
            for j,lattice in enumerate(latticeLists[i]):
                combinedRecords["/%d/Lattice/%d"%(replicate,start+j)] = lattice
            start += times[i].shape[0]
        combinedRecords["/%d/Times"%(replicate)] = allTimes

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
        conf = SparkConf().setAppName("LM RDME Time Series")
        sc = SparkContext(conf=conf)
        main_spark(sc, outputFilename, outputIndices, filename, replicates, interactive)
    else:
        main_local(outputFilename, outputIndices, filename, replicates, interactive)
