## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf3.sfile --skip_less_than=1.0 0 1 2 3 4 5 6 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf4*.sfile --skip_less_than=1e1 0 --interactive
## spark-submit --master yarn-client --num-executors 1 --driver-memory 4g --executor-memory 4g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf6.sfile --skip_less_than=1e1 0 --interactive
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf7.sfile --skip_less_than=1e1 0 --interactive
## rm -rf pdf.dat && spark-submit --master yarn-client --num-executors 5 --driver-memory 4g --executor-memory 4g --executor-cores 4 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_stationary_pdf.py pdf.dat "(0,)" /tmp/test.sfile 0 --interactive
##
from pyspark import SparkConf, SparkContext

from ast import literal_eval as make_tuple
import numpy as np
import os,sys
import re
import zlib

import robertslab.cellio as cellio
from robertslab.helper.sparkProfileHelper import SparkProfileDecorator
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import lm.io.SpeciesCounts_pb2 as SpeciesCounts_pb2
import lm.io.SpeciesTimeSeries_pb2 as SpeciesTimeSeries_pb2

# Global variables.
globalSpeciesToBin=None
globalSkipTime=None

def main(sc, outputFilename, outputIndices, filename, speciesToBin, skipTime, interactive, profiled):
    
    global globalSpeciesToBin, globalSkipTime
    
    # Broadcast the global variables.
    globalSpeciesToBin=sc.broadcast(speciesToBin)
    globalSkipTime=sc.broadcast(skipTime)
    
    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(filename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Bin the species counts records and sum across all of the bins.
    if profiled:
        profiler = SparkProfileDecorator(sc=sc)
        results = allRecords.filter(profiler(filterSpeciesCounts)).flatMap(profiler(binSpeciesCounts)).reduceByKey(profiler(addBins)).collect()
        profiler.print_stats()
    else:
        results = allRecords.filter(filterRecordType).flatMap(binSpeciesCounts).reduceByKeyLocally(addBins)

    # Convert the counts into pdfs.
    pdfs = {}
    numberRecords = 0
    for species in results.keys():
        minCount=results[species][0]
        maxCount=results[species][1]
        bins=results[species][2]
        pdf = np.zeros((len(bins),2),dtype=float)
        pdf[:,0]=np.arange(minCount,maxCount+1).astype(float)
        numberRecords = sum(bins)
        pdf[:,1]=bins.astype(float)/float(numberRecords)
        pdfs["/%d/P"%(species)] = pdf

    # Output the pdfs.
    cellio.cellsave(outputFilename, pdfs, indices=outputIndices, format="hdf5")
    print "Saved pdfs from %d records to %s"%(numberRecords,outputFilename)

    # If interactive, show the pdf.
    #if interactive:
    #    for i in range(0,len(speciesToBin)):
    #         pdf=pdfs["/%d"%(species)]
    #         if pdf.shape[0] >= 2:
    #             plt.figure()
    #             plt.subplot(1,1,1)
    #             plt.bar(np.arange(minCount-0.4,maxCount+1-0.4),bins)
    #             io.show()
    #         else:
    #             print "Warning: cannot plot PDF with only a single bin."

    
def filterRecordType(record):
    
    # Extract the data.
    (name,dataType)=record[0]
    
    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.SpeciesCounts" or dataType == "protobuf:lm.io.SpeciesTimeSeries":
        return True            
    return False
    

def binSpeciesCounts(record):
    
    global globalSpeciesToBin, globalSkipTime
    
    # Get the global variables.
    speciesToBin = globalSpeciesToBin.value
    skipTime = globalSkipTime.value
    
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
    minCounts=species_counts.min(0)
    maxCounts=species_counts.max(0)

    # Figure out if we need to skip any rows.
    if skipTime <= times[0]:
        firstRow=0
    elif skipTime > times[-1]:
        return []
    else:
        firstRow = np.searchsorted(times,skipTime)

    # Go through each species.
    allRecords=[]
    for species in speciesToBin:

        # Bin the data.
        bins,edges=np.histogram(species_counts[firstRow:,species],np.arange(minCounts[species],maxCounts[species]+2,dtype=int))
        allRecords.append((species,(minCounts[species],maxCounts[species],bins)))
            
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
    
if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print "Usage: output_filename hdfs_sfile species+ [--skip-less-than=<time>] [--using-indices=output_indices] [--interactive]"
        quit()

    outputFilename = sys.argv[1]
    outputIndices = None
    filename = sys.argv[2]
    interactive = False
    skipTime=0.0
    species=[]
    profiled=False
    skipNextArg=False
    for i in range(3,len(sys.argv)):

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

        if sys.argv[i] == "--skip-less-than":
            skipTime = float(sys.argv[i+1])
            skipNextArg=True
            continue

        m=re.match(r'--skip-less-than=([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)',sys.argv[i])
        if m != None:
            skipTime=float(m.group(1))
            continue

        m = re.match(r'--profiled', sys.argv[i])
        if m:
            profiled = True
            continue

        # Otherwise, this must be a species.
        species.append(int(sys.argv[i]))

    # Make sure we got at least one species.
    if len(species) == 0:
        print "Error: at least one species must be specified."
        quit()

    # Make sure the output indices were valid.
    if outputIndices is not None and type(outputIndices) != type((0,)):
        print "Error: output indices must be specified as a python formatted tuple, got: %s"%(type(outputIndices))
        quit()

    # Configure Spark
    conf = SparkConf().setAppName("LM Stationary PDF")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc, outputFilename, outputIndices, filename, species, skipTime, interactive, profiled)
