## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 12 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_calc_lm_rdme_stationary_pdf.py pdf.dat "(0,)" /user/erober32/tmp/pdf3.sfile --skip_less_than=1.0 0 1 2 3 4 5 6 --interactive
##
from __future__ import division

try:
    from pyspark import SparkConf, SparkContext
except:
    print "No spark libraries found, ensure you are running locally."
    from robertslab.spark.helper import SparkLocalGlobal
    from robertslab.sfile import SFile

from ast import literal_eval as make_tuple
import sys

import numpy as np
import math
import re
import sys
import zlib

import robertslab.cellio as cellio
import robertslab.rzlib as rzlib
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import lm.io.LatticeTimeSeries_pb2 as LatticeTimeSeries_pb2

# Global variables.
globalSpeciesToBin=None
globalTimeBinWidth=None
globalSparse=None

def main_spark(sc, outputFilename, outputIndices, filename, species, timeBinWidth, sparse, interactive):
    
    global globalSpeciesToBin, globalTimeBinWidth, globalSparse
    
    # Broadcast the global variables.
    globalSpeciesToBin=sc.broadcast(species)
    globalTimeBinWidth=sc.broadcast(timeBinWidth)
    globalSparse=sc.broadcast(sparse)
    
    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(filename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Bin the species counts records and sum across all of the bins.
    results = allRecords.filter(filterRecordType).flatMap(binLatticeOccupancyByTime).reduceByKeyLocally(addBins)

    saveRecords(outputFilename, outputIndices, species, timeBinWidth, results)

    # If interactive, show the pdf.
    #if interactive:
    #    subvolumeCounts=bins.sum(axis=1).sum(axis=1).sum(axis=1)
    #    for i in range(0,len(speciesToBin)):
    #        print "Subvolume distribution for species %d"%(speciesToBin[i])
    #        plt.figure()
    #        plt.subplot(1,1,1)
    #        plt.bar(np.arange(0,subvolumeCounts.shape[1]),np.log10(subvolumeCounts[i,:]))
    #        io.show()


def main_local(outputFilename, outputIndices, filename, species, timeBinWidth, sparse, interactive):

    global globalSpeciesToBin, globalTimeBinWidth, globalSparse

    # Broadcast the global variables.
    globalSpeciesToBin=SparkLocalGlobal(species)
    globalTimeBinWidth=SparkLocalGlobal(timeBinWidth)
    globalSparse=SparkLocalGlobal(sparse)

    # Open the file.
    fp = SFile.fromFilename(filename, "r")

    # Loop through the records.
    reducedRecords = {}
    while True:
        rawRecord = fp.readNextRecord()
        if rawRecord is None: break

        # Filter the record.
        sparkRecord = [(rawRecord.name,rawRecord.dataType)]
        if filterRecordType(sparkRecord):
            print "Processing %s/%s"%(rawRecord.name,rawRecord.dataType)
            sparkRecord.append(fp.readData(rawRecord.dataSize))
            mapRecordList = binLatticeOccupancyByTime(sparkRecord)
            for mapRecord in mapRecordList:
                timeBin = mapRecord[0]
                if timeBin not in reducedRecords:
                    #if len(reducedRecords.keys()) >= 10:
                    #    saveRecords(outputFilename, outputIndices, reducedRecords)
                    #    reducedRecords.clear()
                    reducedRecords[timeBin] = None
                reducedRecords[timeBin] = addBins(reducedRecords[timeBin], mapRecord[1])
        else:
            fp.skipData(rawRecord.dataSize)


    saveRecords(outputFilename, outputIndices, species, timeBinWidth, reducedRecords)


def filterRecordType(record):

    # Extract the data.
    (name,dataType)=record[0]

    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.LatticeTimeSeries":
        return True
    return False



def binLatticeOccupancyByTime(record):

    global globalSpeciesToBin, globalTimeBinWidth, globalSparse

    # Get the global variables.
    speciesToBin = globalSpeciesToBin.value
    timeBinWidth = globalTimeBinWidth.value
    sparse = globalSparse.value

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
            return []

        # See if this is a v1 lattice time series.
        if len(obj.v1_times) == obj.number_entries:

            # Make sure the data is consistent.
            if len(obj.lattices) != obj.number_entries:
                raise ValueError("Invalid array shape.")

            # Create a tuple with the lattice shape.
            latticeShape=(obj.lattices[0].x_size,obj.lattices[0].y_size,obj.lattices[0].z_size,obj.lattices[0].particles_per_site)

            # Figure out the bins size, adding one to the max species count for a count of zero.
            binsShape=(latticeShape[0],latticeShape[1],latticeShape[2],obj.lattices[0].particles_per_site+1)

            # Figure out the time bins for this record.
            timeBins=[]
            for time in obj.v1_times:
                timeBins.append(int(math.floor(time/timeBinWidth)))
            print"%0.3f - %0.3f (%d)"%(obj.v1_times[0],obj.v1_times[-1],len(obj.v1_times))

            # Version of the algorithm that will work better for highly populated lattices.
            if not sparse:

                # Go through each lattice and bin the data.
                allRecords=[]
                bins = None
                currentTimeBin = None
                for i,lattice in enumerate(obj.lattices):

                    # See if this is a new time bin.
                    if timeBins[i] != currentTimeBin:

                        # Save any existing bins.
                        if bins is not None:
                            for species in speciesToBin:
                                allRecords.append(((species,currentTimeBin),bins[species]))

                        # Make the new bins.
                        bins = {}
                        for species in speciesToBin:
                            bins[species] = np.zeros(binsShape, dtype=np.int32)
                        currentTimeBin = timeBins[i]

                    # Convert the data to a numpy array.
                    if lattice.v1_particles_compressed_deflate:
                        latticeData=np.reshape(np.fromstring(zlib.decompress(lattice.v1_particles), dtype=np.uint8), latticeShape)
                    else:
                        latticeData=np.reshape(np.fromstring(lattice.v1_particles, dtype=np.uint8), latticeShape)

                    # Go through each species and bin the counts by lattice site.
                    for species in speciesToBin:

                        # Create a new lattice with the counts for this species.
                        latticeCounts=np.sum((latticeData==(species+1)), axis=3, dtype=np.int8)

                        # Find all the subvolumes with non-zero counts.
                        indices = np.argwhere(latticeCounts)

                        # Increment the bins.
                        bins[species][:,:,:,0] += 1
                        for index in indices:
                            count = latticeCounts[index[0],index[1],index[2]]
                            bins[species][index[0],index[1],index[2],count] += 1
                            bins[species][index[0],index[1],index[2],0] -= 1

                # Add any remaining bins.
                for species in speciesToBin:
                    allRecords.append(((species,currentTimeBin),bins[species]))

                # Return the records.
                return allRecords

            # Version of the algorithm that will work better for sparsely populated lattices.
            else:

                # Go through each lattice and bin the data.
                allRecords=[]
                for i,lattice in enumerate(obj.lattices):

                    if lattice.v1_particles_compressed_deflate:

                        latticeDataZ=np.fromstring(lattice.v1_particles, dtype=np.uint8)
                        flatIndices=np.empty((np.product(latticeShape)//1000,), dtype=np.int32)
                        flatParticles=np.empty((np.product(latticeShape)//1000,), dtype=np.uint8)
                        numberFound=rzlib.decompress_nonzero(latticeDataZ,np.product(latticeShape),flatIndices,flatParticles)
                        if numberFound < 0:
                            raise RuntimeError("Exception during decompress_nonzero: %d."%(numberFound))
                        elif numberFound > flatIndices.shape[0]:
                            raise RuntimeError("Too many nonzero indices found (%d), lattice was not sparse."%(numberFound))
                    else:
                        #flatIndices=np.flatnonzero(latticeData)
                        raise ValueError("Uncompressed data not currently supported.")

                    # Create a list of particle locations for each species.
                    (xs,ys,zs,ps)=np.unravel_index(flatIndices[0:numberFound], latticeShape)
                    nonzeroCounts={}
                    for j in range(0,len(xs)):
                        species=flatParticles[j]-1
                        if species not in nonzeroCounts: nonzeroCounts[species] = {}
                        location=(xs[j],ys[j],zs[j])
                        nonzeroCounts[species][location] = nonzeroCounts[species].get(location,0)+1

                    # Add the records to the list.
                    for species in speciesToBin:
                        allRecords.append(((species,timeBins[i]),(nonzeroCounts.get(species,{}),binsShape)))

                # Return the records.
                return allRecords

        # See if this is a v2 lattice time series.
        elif obj.HasField("times"):
            raise TypeError("V2 lattice time series not yet supported.")

            # Make sure the data is consistent.
            #if len(obj.counts.shape) != 2 or len(obj.times.shape) != 1:
            #    raise "Invalid array shape."
            #if obj.counts.shape[0] != obj.times.shape[0]:
            #    raise "Inconsistent array sizes."
            #if obj.counts.data_type != NDArray_pb2.NDArray.int32 or obj.times.data_type != NDArray_pb2.NDArray.float64:
            #    raise "Invalid array data types."

        else:
            raise TypeError("Unknown lattice time series message version.")

    raise TypeError("Invalid record, unknown data type.")


def addBins(data1, data2):

    # Make sure we have the correct records.
    if data1 is None: return data2
    if data2 is None: return data1
    if not (isinstance(data1,np.ndarray) or (len(data1) == 2 and isinstance(data2[0],dict))):
        raise TypeError("Invalid type for record 1: len=%d (%s) during add bins"%(len(data1),type(data1)))
    if not (isinstance(data2,np.ndarray) or (len(data2) == 2 and isinstance(data2[0],dict))):
        raise TypeError("Invalid type for record 2: len=%d (%s) during add bins"%(len(data2),type(data2)))

    # Process the bins.
    if isinstance(data1,np.ndarray) and isinstance(data2,np.ndarray):
        data1 += data2
        return data1

    elif isinstance(data1,np.ndarray) and len(data2) == 2:
        bins = data1
        bins[:,:,:,0] += 1
        (nonzeroCounts,binsShape)=data2
        for key in nonzeroCounts.keys():
            (x,y,z) = key
            count = nonzeroCounts[key]
            bins[x,y,z,count] += 1
            bins[x,y,z,0] -= 1
        return bins

    elif len(data1) == 2 and isinstance(data2,np.ndarray):
        bins = data2
        bins[:,:,:,0] += 1
        (nonzeroCounts,binsShape)=data1
        for key in nonzeroCounts.keys():
            (x,y,z) = key
            count = nonzeroCounts[key]
            bins[x,y,z,count] += 1
            bins[x,y,z,0] -= 1
        return bins

    elif len(data1) == 2 and len(data2) == 2:
        (nonzeroCounts,binsShape)=data1
        bins = np.zeros(binsShape)
        bins[:,:,:,0] += 2
        for key in nonzeroCounts.keys():
            (x,y,z) = key
            count = nonzeroCounts[key]
            bins[x,y,z,count] += 1
            bins[x,y,z,0] -= 1
        (nonzeroCounts,binsShape)=data2
        for key in nonzeroCounts.keys():
            (x,y,z) = key
            count = nonzeroCounts[key]
            bins[x,y,z,count] += 1
            bins[x,y,z,0] -= 1
        return bins

    raise RuntimeError("Invalid condition during add bins")

def saveRecords(outputFilename, outputIndices, speciesToBin, timeBinWidth, results):

    # Find the min and max time bins and min and max species counts.
    minTimeBin=None
    maxTimeBin=None
    for species,timeBin in results.keys():

        # Check the time bin.
        if minTimeBin is None or timeBin < minTimeBin:
            minTimeBin = timeBin
        if maxTimeBin is None or timeBin > maxTimeBin:
            maxTimeBin = timeBin

    pdfs={}

    # Create the arrays for the times.
    for species in speciesToBin:
        pdfs["/%d/T"%(species)] = (np.arange(minTimeBin,maxTimeBin+1).astype(float)*timeBinWidth)+(timeBinWidth/2)

    # Convert the bins into pdfs.
    for species,timeBin in results.keys():
        bins = results[(species,timeBin)]
        pdfs["/%d/P/%d"%(species,timeBin-minTimeBin)] = bins[:,:,:,:].astype(float)/float(np.sum(bins[0,0,0,:]))

    # Output the pdfs.
    cellio.cellsave(outputFilename, pdfs, indices=outputIndices, format="hdf5")
    print "Saved time-dependent pdfs with %d time bins to %s"%(maxTimeBin-minTimeBin+1,outputFilename)


if __name__ == "__main__":

    if len(sys.argv) < 6:
        print "Usage: output_filename hdfs_sfile time_bin_width species+ [--using-indices=output_indices] [--not-sparse] [--local] [--interactive]"
        quit()

    outputFilename = sys.argv[1]
    outputIndices = None
    filename = sys.argv[2]
    timeBinWidth = float(sys.argv[3])
    sparse = True
    interactive = False
    local = False
    species=[]
    skipNextArg=False
    for i in range(4,len(sys.argv)):

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

        if sys.argv[i] == "--local":
            local=True
            continue

        if sys.argv[i] == "--not-sparse":
            sparse=False
            continue

        if sys.argv[i] == "--interactive":
            interactive=True
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

    # Execute Main functionality
    if not local:
        conf = SparkConf().setAppName("LM RDME Time-Dependent PDF")
        sc = SparkContext(conf=conf)
        main_spark(sc, outputFilename, outputIndices, filename, species, timeBinWidth, sparse, interactive)
    else:
        main_local(outputFilename, outputIndices, filename, species, timeBinWidth, sparse, interactive)

