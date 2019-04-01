## Spark Application - execute with spark-submit, e.g.
## hdfs dfs -rm -r -f /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out && spark-submit --master yarn-client --num-executors 4 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_align_frames.py /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/test2.sfile index.txt reference.png /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out/test.sfile /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out/test.txt no
##

import csv
from operator import add
import os
from pyspark import SparkConf, SparkContext, StorageLevel
import sys
from timeit import default_timer as timer

import copy
import math
import pickle
import re
import sys
import time
import cStringIO
import zlib

import matplotlib.pyplot as plt
from skimage import io
from robertslab.sfile import *
import robertslab.imaging.bead as beadlib
import robertslab.imaging.imaging as imaging
import robertslab.imaging.yeast.segment as segment
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2

# Global variables.
globalRefImageData=None
globalFrameIndex=None

beadValidateOptions={}
beadValidateOptions["interiorIntensity"] = (0.6,1.0,0.50)
beadValidateOptions["radialSymmetry"] = (0.15,)

def main(sc, imageFilename, indexFilename, referenceImageFilename, outImageFilename, outAlignmentFilename, interactive):
    
    global globalRefImageData, globalFrameIndex

    # Load the frame index file.
    frameIndex={}
    firstLine=[]
    with open(indexFilename) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            if firstLine == [] and line[0] == "Frame":
                firstLine = line
            elif len(line) == len(firstLine):
                frameMap={}
                for i in range(0,len(line)):
                    frameMap[firstLine[i]] = line[i]
                frameIndex[int(line[0])] = frameMap
    globalFrameIndex=sc.broadcast(frameIndex)
    
    # Process the reference image.
    print "Processing reference image: %s"%(referenceImageFilename)
    refImageData=loadReferenceImage(referenceImageFilename)
    globalRefImageData=sc.broadcast(refImageData)
    print "Finished processing reference image."

    # Load the records from the sfile.
    rawImages = sc.newAPIHadoopFile(imageFilename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Align the images.
    alignedImages = rawImages.filter(filterPhaseContrastFrames).map(alignAndNormalize)
    alignedImages.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

    # Save a copy the the alignment data as a text file.
    alignedImages.map(formatAlignmentRecords).saveAsTextFile(outAlignmentFilename)

    print "Processed %d images, saved alignment statistics into file: %s."%(alignedImages.count(),outAlignmentFilename)

    # Save a copy of the aligned images as an sfile.
    alignedImages.map(formatImages).saveAsNewAPIHadoopFile(outImageFilename, "robertslab.hadoop.io.SFileOutputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.PythonToSFileHeaderConverter", valueConverter="robertslab.spark.sfile.PythonToSFileRecordConverter")

    print "Processed %d images, saved images into hdfs at: %s."%(alignedImages.count(),outImageFilename)

    # Print out some statistics of the alignment.
    if interactive:
        print "Intensities:"
        for x in alignedImages.map(lambda r: (r[0],r[1][0][2])).sortByKey().collect():
            print "%d = %0.2f"%(x[0],x[1])
        print "Alignments:"
        for x in alignedImages.map(lambda r: (r[0],(r[1][0][0],r[1][0][1]))).sortByKey().collect():
            print "%d = %0.2f,%0.2f"%(x[0],x[1][0],x[1][1])
        for x in alignedImages.map(lambda r: (r[0],r[1][1])).sortByKey().take(2):
            print "Aligned image %d"%(x[0])
            imAligned=np.reshape(np.fromstring(zlib.decompress(x[1]), dtype=np.uint8), refImageData[0].shape)
            plt.figure()
            plt.subplot(1,1,1)
            io.imshow(imAligned)
            io.show()



    
def loadReferenceImage(referenceImageFilename, showIntermediates=False):
    
    # Load the image.
    refImage=io.imread(referenceImageFilename)

    # Calculate the reference background stats.
    (refBgMean,refBgStd)=imaging.calculateBackgroundStats(refImage, showIntermediates=showIntermediates)

    # Calculate the gradient of the image.
    refGradCutoff=0.17
    (G,Gthresh,Gtheta)=segment.calculateGradient(refImage, cutoff=refGradCutoff, showIntermediates=showIntermediates)
    #(G,Gthresh,Gtheta,refGradCutoff)=segment.calculateGradient(refImage, showIntermediates=showIntermediates)

    # Find the bead.
    refBead=beadlib.findBead(refImage, G, Gthresh, Gtheta, showIntermediates=showIntermediates)
    print refBead

    return (refImage,refBgMean,refBgStd,refGradCutoff,refBead)
    

def filterPhaseContrastFrames(record):
    
    global globalFrameIndex

    start=timer()

    # Extract the data.
    (name,dataType)=record[0]
    
    # Make sure the record and and type are correct.
    m=re.search("frame-(\d\d\d\d\d\d).png",name)
    if m == None or dataType != "mime:image/png":
        return False
    frameNumber=int(m.group(1))
        
    # Make sure the image is a phase contrast in the index file.
    frameIndex = globalFrameIndex.value
    if frameIndex[frameNumber]["Phase_Contrast"] != "1":
        print "Skipping frame %d as it was not phase contrast"%(frameNumber)
        return False


    #print "Map filterPhaseContrastFrames took %0.4f seconds"%(timer()-start)

    # Otherwise, we will process this frame.
    return True
    

def alignAndNormalize(record):
    
    global globalRefImageData
    
    start = timer()
    
    # Load the global variables.
    (refImage,refBgMean,refBgStd,refGradCutoff,refBead)=globalRefImageData.value
    
    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # Extract the frame number.
    m=re.search("frame-(\d\d\d\d\d\d).png",name)
    if m == None or dataType != "mime:image/png":
        return None
    frameNumber=int(m.group(1))

    # Load the image.
    im=io.imread(cStringIO.StringIO(data))
    t1 = timer()

    # Extract a region around the position of the reference bead from the current frame.
    imBeadX=int(refBead.xc)-300
    imBeadY=int(refBead.yc)-300
    imBead=im[imBeadY:imBeadY+600,imBeadX:imBeadX+600]
    
    # Find the center of the bead.
    (G,Gthresh,Gtheta)=segment.calculateGradient(imBead, cutoff=refGradCutoff)
    t2 = timer()
    bead=beadlib.findBead(imBead, G, Gthresh, Gtheta)
    t3 = timer()

    # Figure out which bead is closest to the reference bead.
    bead.xc += imBeadX
    bead.yc += imBeadY

    # Extract a region around the bead in the reference image and the current frame.
    imBeadRefX=int(refBead.xc)-100
    imBeadRefY=int(refBead.yc)-100
    imBeadRef=refImage[imBeadRefY:imBeadRefY+200,imBeadRefX:imBeadRefX+200]
    imBeadX=int(bead.xc)-100
    imBeadY=int(bead.yc)-100
    imBead=im[imBeadY:imBeadY+200,imBeadX:imBeadX+200]
    
    # Align the region around the bead with the reference image.
    (dx,dy)=segment.alignImages(imBead, imBeadRef, maxOffset=1)
    t4 = timer()

    # Figure out the total image alignment.
    frameDX=imBeadX-imBeadRefX+dx
    frameDY=imBeadY-imBeadRefY+dy

    # Re-extract the aligned region around the bead for use in background calculation.
    imBeadRefX-=50
    imBeadRefY-=50
    imBead=im[imBeadRefY+frameDY:imBeadRefY+frameDY+300,imBeadRefX+frameDX:imBeadRefX+frameDX+300]
    
    # Calculate the frame background from the region near the bead.
    (frameMean,frameStd)=imaging.calculateBackgroundStats(imBead)
    t5 = timer()

    # Store out the alignment and normalization statistics.
    stats=np.zeros((4,),dtype=np.float64)
    stats[0]=frameDX
    stats[1]=frameDY
    stats[2]=frameMean
    stats[3]=frameStd
    
    # Normalize and align the image.
    if frameDX <= 0:
        x1s=0
        x2s=im.shape[1]+frameDX
        x1d=-frameDX
        x2d=im.shape[1]
    else:
        x1s=frameDX
        x2s=im.shape[1]
        x1d=0
        x2d=im.shape[1]-frameDX
    if frameDY <= 0:
        y1s=0
        y2s=im.shape[0]+frameDY
        y1d=-frameDY
        y2d=im.shape[0]
    else:
        y1s=frameDY
        y2s=im.shape[0]
        y1d=0
        y2d=im.shape[0]-frameDY
    imAligned=np.zeros(im.shape,dtype=np.uint8)+int(frameMean)
    imAligned[y1d:y2d,x1d:x2d] = im[y1s:y2s,x1s:x2s]
    imAligned -= int(frameMean-refBgMean)
    
    # Compress the aligned image.
    imAlignedZ=zlib.compress(imAligned.tostring(),1)
    t6 = timer()

    stop=timer()
    #print "Map alignAndNormalize took %0.4f seconds (%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f)"%(stop-start,t1-start,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5)

    # Return the aligned image.
    return (frameNumber,(stats,im.shape[0],im.shape[1],imAlignedZ))
    
    

def formatImages(record):

    start = timer()
    
    # Extract the record.
    frameNumber=record[0]
    normStats=record[1][0]
    height=record[1][1]
    width=record[1][2]
    dataZ=record[1][3]

    # Create an sfile header for the record.
    sfileHeader = ("frame-aligned-%d"%(frameNumber),"protobuf:robertslab.pbuf.NDArray")

    # Create a protobuf NDArray object.
    obj = NDArray_pb2.NDArray()
    obj.array_order = NDArray_pb2.NDArray.ROW_MAJOR
    obj.byte_order = NDArray_pb2.NDArray.LITTLE_ENDIAN
    obj.data_type = NDArray_pb2.NDArray.uint8
    obj.shape.append(height)
    obj.shape.append(width)
    obj.data = dataZ
    obj.compressed_deflate = True

    # Serialize the data into an sfile record.
    sfileRecord = bytearray(obj.SerializeToString())

    #print "Map formatImages took %0.4f seconds"%(timer()-start)

    # Return the aligned image.
    return (sfileHeader,sfileRecord)


def formatAlignmentRecords(record):

    start=timer()

    # Extract the record.
    frameNumber=record[0]
    stats=record[1][0]

    #print "Map formatAlignmentRecords took %0.4f seconds"%(timer()-start)

    # Return a string representation of the record.
    return "%d,%0.4f,%0.4f,%0.4f,%0.4f"%(frameNumber,stats[0],stats[1],stats[2],stats[3])


if __name__ == "__main__":
    
    if len(sys.argv) < 6:
        print "Usage: image_sfile index_file reference_image out_image_sfile out_alignment_txt [interactive]"
        quit()
    
    imageFilename = sys.argv[1]
    indexFilename = sys.argv[2]
    referenceImageFilename = sys.argv[3]
    outImageFilename = sys.argv[4]
    outAlignmentFilename = sys.argv[5]
    interactive=False
    if len(sys.argv) >= 7 and sys.argv[6].lower() in ['true', '1', 'yes']:
        interactive=True
        
    # Configure Spark
    conf = SparkConf().setAppName("Image Alignment")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc, imageFilename, indexFilename, referenceImageFilename, outImageFilename, outAlignmentFilename, interactive)
