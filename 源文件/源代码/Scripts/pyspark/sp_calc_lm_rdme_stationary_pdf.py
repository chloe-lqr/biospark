## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 12 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_rdme_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf3.sfile --skip_less_than=1.0 0 1 2 3 4 5 6 --interactive
##

from ast import literal_eval as make_tuple
import os
from pyspark import SparkConf, SparkContext
import sys
from timeit import default_timer as timer

import numpy as np
import math
import re
import sys
import time
import cStringIO
import zlib

from scipy.stats import norm

import matplotlib.pyplot as plt
from skimage import io, filter

import robertslab.cellio as cellio
import robertslab.rzlib as rzlib
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import lm.io.LatticeTimeSeries_pb2 as LatticeTimeSeries_pb2

# Global variables.
globalSpeciesToBin=None
globalSkipTime=None
globalSparse=None

def main(sc, outputDir, outputIndices, filename, speciesToBin, skipTime, sparse, interactive):
    
    global globalSpeciesToBin, globalSkipTime, globalSparse
    
    # Broadcast the global variables.
    globalSpeciesToBin=sc.broadcast(speciesToBin)
    globalSkipTime=sc.broadcast(skipTime)
    globalSparse=sc.broadcast(sparse)
    
    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(filename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Bin the species counts records and sum across all of the bins.
    results = allRecords.filter(filterLatticeTimeSeries).map(binLatticeOccupancy).reduceByKey(addLatticeBins).values().collect()
    totalTimePoints=results[0][0]
    bins=results[0][1]
    bins[:,:,:,:,0] += totalTimePoints
    print "Recovered bins from %d total time points"%(totalTimePoints)
    print bins.shape

    # Save the pdfs into a .mat file in the output directory named according to the output indices.
    pdfs=np.zeros((len(speciesToBin),),dtype=object)
    for i in range(0,len(speciesToBin)):
        pdf=bins[i,:,:,:,:].astype(float)/float(np.sum(bins[i,0,0,0,:]))
        pdfs[i] = pdf
    cellio.cellsave(outputDir,pdfs,outputIndices);
    print("Binned species data into %s"%(outputDir))

    # If interactive, show the pdf.
    if interactive:
        subvolumeCounts=bins.sum(axis=1).sum(axis=1).sum(axis=1)
        for i in range(0,len(speciesToBin)):
            print "Subvolume distribution for species %d"%(speciesToBin[i])
            plt.figure()
            plt.subplot(1,1,1)
            plt.bar(np.arange(0,subvolumeCounts.shape[1]),np.log10(subvolumeCounts[i,:]))
            io.show()

    # Save a copy of the records, for examples purposes only here.
    #records.saveAsNewAPIHadoopFile(sys.argv[1]+".copy", "robertslab.hadoop.io.SFileOutputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.PythonToSFileHeaderConverter", valueConverter="robertslab.spark.sfile.PythonToSFileRecordConverter", conf=conf)
    
    
def filterLatticeTimeSeries(record):
    
    # Extract the data.
    (name,dataType)=record[0]
    
    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.LatticeTimeSeries":
        m=re.match("/Simulations/(\d+)/LatticeTimeSeries",name)
        if m == None:
            return False
        replicateNumber=int(m.group(1))
        if replicateNumber > 100:
            return False
        return True
    return False
    

def binLatticeOccupancy(record):
    
    global globalSpeciesToBin, globalSkipTime, globalSparse

    start=timer()

    # Get the global variables.
    speciesToBin = globalSpeciesToBin.value
    skipTime = globalSkipTime.value
    sparse = globalSparse.value
    
    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]
    
    # If this is a SpeciesCounts object, extract the data.
    if dataType == "protobuf:lm.io.LatticeTimeSeries":
        m=re.match("/Simulations/(\d+)/LatticeTimeSeries",name)
        if m == None:
            return None
        replicateNumber=int(m.group(1))
    
        # Deserialize the data.
        obj=LatticeTimeSeries_pb2.LatticeTimeSeries()
        obj.ParseFromString(str(data))

        # If there are no entries, return an empty set.
        if obj.number_entries == 0:
            allBins=[]
            for species in speciesToBin:
                allBins.append(())
            return (0,(0,))

        # See if this is a v1 lattice time series.
        if len(obj.v1_times) == obj.number_entries:

            # Make sure the data is consistent.
            if len(obj.lattices) != obj.number_entries:
                raise "Invalid array shape."

            # Convert the times to a numpy array.
            t0=timer()
            times=np.zeros((obj.number_entries,),dtype=float)
            i=0
            for val in obj.v1_times:
                times[i] = val
                i+=1

            # Create a tuple with the lattice shape.
            latticeShape=(obj.lattices[0].x_size,obj.lattices[0].y_size,obj.lattices[0].z_size,obj.lattices[0].particles_per_site)

            # Create the bins for this species.
            t1=timer()
            if not sparse:
                bins = np.zeros((len(speciesToBin),latticeShape[0],latticeShape[1],latticeShape[2],latticeShape[3]+1), dtype=np.int32)
            else:
                nonzeroCounts={}
                binsShape = (len(speciesToBin),latticeShape[0],latticeShape[1],latticeShape[2],latticeShape[3]+1)

            # Go through each lattice.
            t2=timer()
            totalTimePoints = 0
            for i in range(0,len(obj.lattices)):

                # Get the lattice.
                lattice = obj.lattices[i]

                t3=0.0
                t8=0.0
                t9=0.0
                t10=0.0

                # Version of the algorithm that will work better for highly populated lattices.
                if not sparse:

                    # Convert the data to a numpy array.
                    t=timer()
                    if lattice.v1_particles_compressed_deflate:
                        latticeData=np.reshape(np.fromstring(zlib.decompress(lattice.v1_particles), dtype=np.uint8), latticeShape)
                    else:
                        latticeData=np.reshape(np.fromstring(lattice.v1_particles, dtype=np.uint8), latticeShape)
                    t3=timer()-t

                    # Go through each species and bin the counts by lattice site.
                    t4=timer()
                    for j in range(0,len(speciesToBin)):

                        # Create a new lattice with the counts for this species.
                        t=timer()
                        latticeCounts=np.sum((latticeData==(speciesToBin[j]+1)), axis=3, dtype=np.int8)
                        t8+=timer()-t

                        # Find all the subvolumes with non-zero counts.
                        t=timer()
                        indices = np.argwhere(latticeCounts)
                        t9+=timer()-t

                        # Increment the bins.
                        t=timer()
                        for index in indices:
                            count = latticeCounts[index[0],index[1],index[2]]
                            bins[j,index[0],index[1],index[2],count] += 1
                            bins[j,index[0],index[1],index[2],0] -= 1

                        t10+=timer()-t
                    t5=timer()

                # Version of the algorithm that will work better for sparsely populated lattices.
                else:


                    t=timer()
                    if lattice.v1_particles_compressed_deflate:

                        latticeDataZ=np.fromstring(lattice.v1_particles, dtype=np.uint8)
                        flatIndices=np.empty((np.product(latticeShape)/1000,), dtype=np.int32)
                        flatParticles=np.empty((np.product(latticeShape)/1000,), dtype=np.uint8)
                        numberFound=rzlib.decompress_nonzero(latticeDataZ,np.product(latticeShape),flatIndices,flatParticles)
                        if numberFound < 0:
                            raise "Exception during decompress_nonzero: %d."%(numberFound)
                        elif numberFound > flatIndices.shape[0]:
                            raise "Too many nonzero indices found, lattice was not sparse."
                    else:
                        #flatIndices=np.flatnonzero(latticeData)
                        raise "Uncompressed data not currently supported."
                    t3=timer()-t

                    t4=timer()

                    t=timer()
                    (xs,ys,zs,ps)=np.unravel_index(flatIndices[0:numberFound], latticeShape)
                    for i in range(0,len(xs)):
                        species=flatParticles[i]-1
                        key=(species,xs[i],ys[i],zs[i])
                        nonzeroCounts[key] = nonzeroCounts.get(key,0)+1
                    t8+=timer()-t


                    t5=timer()


                # Update the number of lattices that we have processed.
                totalTimePoints += 1

            # Update the zero bin with the remainder, since we didn't track those as we went.
            t6=timer()
            t7=timer()

            #print "Map binLatticeOccupancy took %0.3f seconds for %d lattices."%(timer()-start,len(obj.lattices))
            stop = timer()
            #print "Map binLatticeOccupancy (%0.4f - %0.4f) took %0.3f seconds for %d lattices: %0.3f %0.3f %0.3f %0.3f (%0.3f %0.3f %0.3f) %0.3f"%(start,stop,stop-start,len(obj.lattices),t1-t0,t2-t1,t3,t5-t4,t8,t9,t10,t7-t6)

            # Return the bins.
            if not sparse:
                return (0,(totalTimePoints,bins))
            else:
                return (0,(totalTimePoints,nonzeroCounts,binsShape))

            # See if this is a v2 lattice time series.
        elif obj.HasField("times"):
            raise "V2 lattice time series not yet supported."

            # Make sure the data is consistent.
            #if len(obj.counts.shape) != 2 or len(obj.times.shape) != 1:
            #    raise "Invalid array shape."
            #if obj.counts.shape[0] != obj.times.shape[0]:
            #    raise "Inconsistent array sizes."
            #if obj.counts.data_type != NDArray_pb2.NDArray.int32 or obj.times.data_type != NDArray_pb2.NDArray.float64:
            #    raise "Invalid array data types."

        else:
            raise "Unknown lattice time series message version."


def addLatticeBins(bins1, bins2):

    start=timer()

    # See if both records have data in them.
    if bins1[0] > 0 and bins2[0] > 0:

        totalTimePoints = bins1[0]+bins2[0]

        if isinstance(bins1[1],np.ndarray) and isinstance(bins2[1],np.ndarray):
            bins = bins1[1]
            bins += bins2[1]
            stop = timer()
            #print "Reduce addLatticeBins/1 (%0.4f - %0.3f) took %0.3f seconds."%(start,stop,stop-start)
            return (totalTimePoints,bins)

        elif isinstance(bins1[1],np.ndarray) and isinstance(bins2[1],dict):
            bins=bins1[1]
            nonzeroCounts=bins2[1]
            for key in nonzeroCounts.keys():
                bins[key[0],key[1],key[2],key[3],nonzeroCounts[key]] += 1
                bins[key[0],key[1],key[2],key[3],0] -= 1
            stop = timer()
            #print "Reduce addLatticeBins/2 (%0.4f - %0.4f) took %0.3f seconds."%(start,stop,stop-start)
            return (totalTimePoints,bins)

        elif isinstance(bins1[1],dict) and isinstance(bins2[1],np.ndarray):
            bins=bins2[1]
            nonzeroCounts=bins1[1]
            for key in nonzeroCounts.keys():
                bins[key[0],key[1],key[2],key[3],nonzeroCounts[key]] += 1
                bins[key[0],key[1],key[2],key[3],0] -= 1
            stop = timer()
            #print "Reduce addLatticeBins/3 (%0.4f - %0.4f) took %0.3f seconds."%(start,stop,stop-start)
            return (totalTimePoints,bins)

        elif isinstance(bins1[1],dict) and isinstance(bins2[1],dict):
            binsShape = bins1[2]
            bins = np.zeros(binsShape, dtype=np.int32)
            nonzeroCounts=bins1[1]
            for key in nonzeroCounts.keys():
                bins[key[0],key[1],key[2],key[3],nonzeroCounts[key]] += 1
                bins[key[0],key[1],key[2],key[3],0] -= 1
            nonzeroCounts=bins2[1]
            for key in nonzeroCounts.keys():
                bins[key[0],key[1],key[2],key[3],nonzeroCounts[key]] += 1
                bins[key[0],key[1],key[2],key[3],0] -= 1
            stop = timer()
            #print "Reduce addLatticeBins/4 (%0.4f - %0.4f) took %0.3f seconds."%(start,stop,stop-start)
            return (totalTimePoints,bins)

    # Otherwise, if just record 1 had data, return it.
    elif bins1[0] > 0:
        return bins1

    # Otherwise, if just record 2 had data, return it.
    elif bins2[0] > 0:
        return bins2

    # Otherwise, neither record had data.
    else:
        return bins1

if __name__ == "__main__":
    
    if len(sys.argv) < 5:
        print "Usage: output_dir output_indices hdfs_sfile [--skip_less_than=<time>] [--not-sparse] species+ [--interactive]"
        quit()
    
    outputDir = sys.argv[1]
    outputIndices = make_tuple(sys.argv[2])
    if type(outputIndices) != type((0,)):
        print "Error: output indices must be specified in as a python formatted tuple, got: %s"%(type(outputIndices))
        quit()
    filename = sys.argv[3]
    interactive = False
    sparse = True
    skipTime=0.0
    species=[]
    for i in range(4,len(sys.argv)):
        if sys.argv[i] == "--interactive":
            interactive=True
            continue
        if sys.argv[i] == "--not-sparse":
            sparse=False
            continue

        m=re.match(r'--skip_less_than=([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)',sys.argv[i])
        if m != None:
            skipTime=float(m.group(1))
            continue

        m = re.match(r'-o=(\S+)', sys.argv[i])
        if m:
            outputDir = m.group(1) + '.dat'
            continue

        m = re.match(r'--suffix=(\S+)', sys.argv[i])
        if m:
            outputName,outputExt = os.path.splitext(outputDir)
            outputDir = outputName + m.group(1) + outputExt
            continue

        species.append(int(sys.argv[i]))
        
    # Make sure we got at least one species.
    if len(species) == 0:
        print "Error: at least one species must be specified."
        quit()
        
    # Configure Spark
    conf = SparkConf().setAppName("LM RDME PDF Calculation")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc, outputDir, outputIndices, filename, species, skipTime, sparse, interactive)
