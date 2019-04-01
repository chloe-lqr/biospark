## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 16 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_yeast_segment.py /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out/test.sfile index.txt reference.png cells.pkl --interactive
##

import csv
from operator import add
import os
from pyspark import SparkConf, SparkContext
import sys
from timeit import default_timer as timer

import copy
import math
import pickle
import re
import sys
import time
import uuid
import cStringIO
import zlib

import matplotlib.pyplot as plt
from skimage import io
from robertslab.sfile import *
import robertslab.imaging.bead as beads
import robertslab.imaging.imaging as imaging
import robertslab.imaging.yeast.contour as contour
import robertslab.imaging.yeast.kvarn_contour as kvarn
import robertslab.imaging.yeast.segment as segment
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import robertslab.pbuf.imaging.yeast.CellContour_pb2 as CellContour_pb2

# Global variables.
globalRefImageData=None
globalFrameIndex=None
globalContourPoints=None
globalTolerance=None
globalValidateOptions=None
globalFrameRange=None

validateOptions={}
validateOptions["area"] = (250.0,5600.0)
validateOptions["interiorIntensity"] = (0.27,0.5,0.33)
validateOptions["borderIntensity"] = (0.2,0.5,0.33)
validateOptions["interiorBorderIntensityRatio"] = (0.94,1.5)
validateOptions["radius"] = (5.0,45.0,0.19)
validateOptions["gradientEnergy"] = (-0.2,-0.02,1.0)

beadValidateOptions={}
beadValidateOptions["interiorIntensity"] = (0.6,1.0,0.50)
beadValidateOptions["radialSymmetry"] = (0.15,)


def main(sc, imageFilename, indexFilename, referenceImageFilename, outputFilename, contourPoints, tolerance, imageMask, frameRange, validateOptions, segmentationTasks, interactive):
    
    global globalRefImageData, globalFrameIndex, globalContourPoints, globalTolerance, globalValidateOptions, globalFrameRange

    # Broadcast any global options.
    globalContourPoints=sc.broadcast(contourPoints)
    globalTolerance=sc.broadcast(tolerance)
    globalValidateOptions=sc.broadcast(validateOptions)
    globalFrameRange=sc.broadcast(frameRange)


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
    refImageData=loadReferenceImage(referenceImageFilename)
    globalRefImageData=sc.broadcast(refImageData)
    refImage=refImageData[0]
    refBeads=refImageData[4]

    # Validate the image mask.
    if imageMask is not None and not(len(imageMask) == 4 and imageMask[0] < imageMask[2] and imageMask[0] >= 0 and imageMask[2] < refImage.shape[1] and imageMask[1] < imageMask[3] and imageMask[1] >= 0 and imageMask[3] < refImage.shape[0]):
        raise Exception("Invalid image mask:",imageMask)


    # Load the records from the sfile.
    allAlignedImages = sc.newAPIHadoopFile(imageFilename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")
    alignedImages = allAlignedImages.filter(filterFrames).map(extractFrame)
    alignedImages.cache()

    # Calculate the gradient.
    gradients = alignedImages.map(computeGradient)
    gradients.cache()
    
    # Join the gradients with the images.
    imagesAndGradients = alignedImages.join(gradients)

    # Get a list of all the frames available.
    frameList=imagesAndGradients.keys().collect()
    frameList.sort()

    # Loop to find all cells on the first frame.
    frameNumber=frameList[0]
    print "DEBUG: Processing frame %d"%(frameNumber)
    cells={frameNumber:[]}
    while True:

        # Create an exclusion mask.
        exclusionMask = np.zeros(refImage.shape, dtype=bool)

        # Apply the image mask, if we have one.
        if imageMask is not None:
            x1,y1,x2,y2=imageMask
            exclusionMask[0:y1,:] = True
            exclusionMask[y1:y2,0:x1] = True
            exclusionMask[y1:y2,x2:] = True
            exclusionMask[y2:,:] = True

        # Add the beads and any identified cells to the exclusion mask.
        for refBead in refBeads:
            refBead.fillImageMask(exclusionMask, 5.0)
        for cell in cells[frameNumber]:
            cell.fillImageMask(exclusionMask, 5.0)

        # Look for new cells outside the mask.
        newCellsRDD=imagesAndGradients.filter(lambda x: x[0] == frameNumber).flatMap(lambda x: extractPotentialNewCells(x,exclusionMask)).repartition(segmentationTasks).flatMap(lambda x: findNewCellContour(x,None))
        newCells=newCellsRDD.values().collect()

        # If there were no new cells, we are done.
        if len(newCells) == 0:
            break

        # Add the new cells to the list.
        cells[frameNumber].extend(newCells)

    # Optimize all of the cells for interactions.
    print "DEBUG: Globally refining cell interactions."
    cells[frameNumber]=contour.refineCellInteractions(cells[frameNumber], tolerance=tolerance)

    print "DEBUG: Processed frame %d, %d cells"%(frameNumber,len(cells[frameNumber]))

    # Print out some statistics of the initial cells.
    if interactive:
        print "Initial cells %d"%(frameNumber)
        IZ=alignedImages.filter(lambda x: x[0] == frameNumber).values().collect()[0][1]
        I=np.reshape(np.fromstring(zlib.decompress(IZ), dtype=np.uint8), refImageData[0].shape)
        plt.figure()
        io.imshow(I)
        for cell in cells[frameNumber]:
            plt.scatter(cell.xc+cell.Ix0,cell.yc+cell.Iy0,color='r',marker='x')
            plt.plot(cell.xs+cell.Ix0,cell.ys+cell.Iy0,color='r')
        plt.axis([0,I.shape[1],I.shape[0],0])
        #plt.axis([600,1100,1000,700])
        io.show()
        
    # Loop over the frames one at a time and propagate the cells.
    previousFrameNumber=frameNumber
    for frameNumber in frameList[1:]:

        # Initialize the cell list.
        cells[frameNumber]=[]
        print "DEBUG: Processing frame %d"%(frameNumber)

        # Refine the existing cells for this frame.
        print "DEBUG: Processing %d existing cells"%(len(cells[previousFrameNumber]))
        existingCellsRDD=alignedImages.filter(lambda x: x[0] == frameNumber).flatMap(lambda x: extractExistingCells(x,cells[previousFrameNumber])).repartition(segmentationTasks).flatMap(lambda x: findExistingCellContour(x,cells[previousFrameNumber]))
        existingCells=existingCellsRDD.values().collect()
        cells[frameNumber].extend(existingCells)

        # Calculate an exclusion mask for this frame.
        exclusionMask[:,:] = False
        if imageMask is not None:
            x1,y1,x2,y2=imageMask
            exclusionMask[0:y1,:] = True
            exclusionMask[y1:y2,0:x1] = True
            exclusionMask[y1:y2,x2:] = True
            exclusionMask[y2:,:] = True
        for refBead in refBeads:
            refBead.fillImageMask(exclusionMask, 5.0)
        for cell in existingCells:
            cell.fillImageMask(exclusionMask, 5.0)

        # See what the Hough transform for the remainder of the frame looks like.
        if frameNumber >= 1300:
            print "Cropped Hough transform: %d"%(frameNumber)
            GZs=gradients.filter(lambda x: x[0] == frameNumber).values().collect()
            GZ=GZs[0][0]
            GthreshZ=GZs[0][1]
            GthetaZ=GZs[0][2]
            G=np.reshape(np.fromstring(zlib.decompress(GZ), dtype=np.float32), refImage.shape)
            Gthresh=np.reshape(np.fromstring(zlib.decompress(GthreshZ), dtype=np.uint8), refImage.shape)
            Gtheta=np.reshape(np.fromstring(zlib.decompress(GthetaZ), dtype=np.float32), refImage.shape)
            (H,Hthresh)=segment.calculateHoughTransform(G, Gthresh, Gtheta, 50, direction=0, exclusionMask=exclusionMask, showIntermediates=True)
            segment.estimateCenters(H, Hthresh, exclusionMask=exclusionMask, showIntermediates=True)

        # Find any new cells for this frame.
        newCellsRDD=imagesAndGradients.filter(lambda x: x[0] == frameNumber).flatMap(lambda x: extractPotentialNewCells(x,exclusionMask)).repartition(segmentationTasks).flatMap(lambda x: findNewCellContour(x,existingCells))
        newCells=newCellsRDD.values().collect()

        # Add the new cells to the list, if their centers don't overlap with an existing cell.
        for newCell in newCells:
            print "DEBUG: Found new cell %s at %d,%d"%(newCell.id,newCell.Ix0+newCell.xc,newCell.Iy0+newCell.yc)
            distinctCell = True
            for existingCell in existingCells:
                print "DEBUG: Checking against %s at %d,%d"%(existingCell.id,existingCell.Ix0+existingCell.xc,existingCell.Iy0+existingCell.yc)
                if existingCell.containsPoint(newCell.Ix0+newCell.xc, newCell.Iy0+newCell.yc):
                    distinctCell = False
                    print "DEBUG: Overlapped with %s"%existingCell.id
                    break
            if distinctCell:
                cells[frameNumber].append(newCell)
                print "DEBUG: Added"

        # Optimize all of the cells for interactions.
        print "DEBUG: Globally refining cell interactions."
        cells[frameNumber]=contour.refineCellInteractions(cells[frameNumber], tolerance=tolerance)

        print "DEBUG: Processed frame %d, %d cells"%(frameNumber,len(cells[frameNumber]))

        # Save this frame number as the previous frame.
        previousFrameNumber=frameNumber

        if interactive:
            print "Existing cells %d"%(frameNumber)
            IZ=alignedImages.filter(lambda x: x[0] == frameNumber).values().collect()[0][1]
            I=np.reshape(np.fromstring(zlib.decompress(IZ), dtype=np.uint8), refImageData[0].shape)
            plt.figure()
            io.imshow(I)
            #for cell in cells[frameList[0]]:
            #    plt.scatter(cell.xc+cell.Ix0,cell.yc+cell.Iy0,color='b',marker='x')
            #    plt.plot(cell.xs+cell.Ix0,cell.ys+cell.Iy0,color='b')
            for cell in existingCells:
                plt.scatter(cell.xc+cell.Ix0,cell.yc+cell.Iy0,color='r',marker='x')
                plt.plot(cell.xs+cell.Ix0,cell.ys+cell.Iy0,color='r')
            for cell in newCells:
                plt.scatter(cell.xc+cell.Ix0,cell.yc+cell.Iy0,color='r',marker='x')
                plt.plot(cell.xs+cell.Ix0,cell.ys+cell.Iy0,color='r')
            plt.axis([0,I.shape[1],I.shape[0],0])
            #plt.axis([600,1100,1000,700])
            io.show()


    # See if we should save everything as a python pickle file.
    if outputFilename.endswith(".pkl"):
        output = open(outputFilename, 'wb')
        pickle.dump(cells, output)
        output.close()

    # Otherwise, write the contours out as a hadoop file.
    else:
        allCells=[]
        for key, values in cells.iteritems():
            for value in values:
                allCells.append((key,value))
        allCellsRDD = sc.parallelize(allCells)
        allCellsRDD.map(formatCellContours).saveAsNewAPIHadoopFile(outputFilename, "robertslab.hadoop.io.SFileOutputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.PythonToSFileHeaderConverter", valueConverter="robertslab.spark.sfile.PythonToSFileRecordConverter")


    
def loadReferenceImage(referenceImageFilename, showIntermediates=False):
    
    # Load the image.
    refImage=io.imread(referenceImageFilename)

    # Calculate the reference background stats.
    (refBgMean,refBgStd)=imaging.calculateBackgroundStats(refImage, showIntermediates=showIntermediates)

    # Calculate the gradient of the image.
    refGradCutoff = 0.17
    (G,Gthresh,Gtheta)=segment.calculateGradient(refImage, cutoff=refGradCutoff, showIntermediates=showIntermediates)
    #(G,Gthresh,Gtheta,refGradCutoff)=segment.calculateGradient(refImage, showIntermediates=showIntermediates)

    # Find the bead.
    refBeads=beads.findBeads(refImage, G, Gthresh, Gtheta, beadValidateOptions, showIntermediates=showIntermediates)

    return (refImage,refBgMean,refBgStd,refGradCutoff,refBeads)
    

def filterFrames(record):
    
    global globalFrameIndex, globalFrameRange
    
    # Extract the data.
    (name,dataType)=record[0]

    # Load the global variables.
    frameIndex = globalFrameIndex.value
    frameRange = globalFrameRange.value


    # Make sure the record and and type are correct.
    m=re.search("frame-aligned-(\d+)",name)
    if m == None or dataType != "protobuf:robertslab.pbuf.NDArray":
        return False
    frameNumber=int(m.group(1))

    # Make sure the frame is in range.
    if frameRange is not None and len(frameRange) == 2 and (frameNumber < frameRange[0] or frameNumber > frameRange[1]):
        print "Skipping frame %d as it was not in the processing range %d-%d"%(frameNumber,frameRange[0],frameRange[1])
        return False

    # Make sure the image is a phase contrast in the index file.
    if frameIndex[frameNumber]["Phase_Contrast"] != "1":
        print "Skipping frame %d as it was not phase contrast"%(frameNumber)
        return False
        
    # Otherwise, we will process this frame.
    return True
    

def extractFrame(record):
    
    start = timer()

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # Extract the frame number.
    m=re.search("frame-aligned-(\d+)",name)
    if m == None or dataType != "protobuf:robertslab.pbuf.NDArray":
        return None
    frameNumber=int(m.group(1))

    print "Extracting frame %d (%d)"%(frameNumber,len(data))

    # Extract the array.
    obj=NDArray_pb2.NDArray()
    obj.ParseFromString(str(data))

    # Make sure the array has the right type and shape.
    if obj.data_type != NDArray_pb2.NDArray.uint8:
        raise "Invalid data type for pbuf.NDArray"
    if len(obj.shape) != 2:
        raise "Invalid shape for pbuf.NDArray"

    # Create the numpy array for the image.
    if obj.compressed_deflate:
        imAlignedZ=obj.data
    else:
        imAlignedZ=zlib.compress(obj.data,1)


    print "Map extractFrame took %0.3f seconds."%(timer()-start)
    
    # Return the aligned image.
    return (frameNumber,(tuple(obj.shape),imAlignedZ))
    
    

def computeGradient(record):

    global globalRefImageData

    start = timer()
    
    # Extract the record.
    frameNumber=record[0]
    imShape=record[1][0]
    imAlignedZ=record[1][1]

    # Load the global variables.
    (refImage,refBgMean,refBgStd,refGradCutoff,refBead)=globalRefImageData.value

    # Decompress the aligned image.
    imAligned=np.reshape(np.fromstring(zlib.decompress(imAlignedZ), dtype=np.uint8), imShape)
            
    # Calculate the gradient.
    (grad,gradThresh,gradDirection)=segment.calculateGradient(imAligned, cutoff=refGradCutoff)
    
    # Threshold the gradient and the gradient direction.
    grad[gradThresh==0]=0.0
    gradDirection[gradThresh==0]=0.0

    # Compress the gradient arrays.
    gradZ=zlib.compress(grad.astype(np.float32).tostring(),1)
    gradThreshZ=zlib.compress(gradThresh.tostring(),1)
    gradDirectionZ=zlib.compress(gradDirection.astype(np.float32).tostring(),1)
        
    print "Map computeGradient took %0.3f seconds."%(timer()-start)

    # Return the aligned image.
    return (frameNumber,(gradZ,gradThreshZ,gradDirectionZ))

def extractPotentialNewCells(record, exclusionMask=None):

    global globalRefImageData

    start = timer()

    frameNumber=record[0]

    # Load the global variables.
    (refImage,refBgMean,refBgStd,refGradCutoff,refBead)=globalRefImageData.value

    # Extract the record.
    imShape=record[1][0][0]
    IZ=record[1][0][1]
    GZ=record[1][1][0]
    GthreshZ=record[1][1][1]
    GthetaZ=record[1][1][2]
    I=np.reshape(np.fromstring(zlib.decompress(IZ), dtype=np.uint8), imShape)
    G=np.reshape(np.fromstring(zlib.decompress(GZ), dtype=np.float32), imShape)
    Gthresh=np.reshape(np.fromstring(zlib.decompress(GthreshZ), dtype=np.uint8), imShape)
    Gtheta=np.reshape(np.fromstring(zlib.decompress(GthetaZ), dtype=np.float32), imShape)

    # Calculate the Hough transform.
    (H,Hthresh)=segment.calculateHoughTransform(G, Gthresh, Gtheta, 50, direction=0, exclusionMask=exclusionMask)

    # Estimate the cluster centers from the Hough transform.
    centers=segment.estimateCenters(H, Hthresh, exclusionMask=exclusionMask)
    print "DEBUG: found %d potential cells in frame %d"%(len(centers),frameNumber)

    # Go through all of the centers.
    potentialCells=[]
    for (xc,yc,r) in centers:

        # Extract the cell image into a separate buffer.
        imCell = imaging.extractImageRegion(I, int(xc)-300, int(yc)-300, int(xc)+300, int(yc)+300, int(refBgMean))

        # Add this cell to the list.
        potentialCells.append((frameNumber,(300,300,imCell,int(xc)-300,int(yc)-300)))

    #print "Map extractPotentialNewCells %d took %0.3f seconds."%(frameNumber,timer()-start)

    # Return the images sections to be processed.
    return potentialCells

def findNewCellContour(record, previousCells):

    global globalContourPoints
    global globalTolerance
    global globalValidateOptions
    contourPoints = globalContourPoints.value
    tolerance = globalTolerance.value
    validateOptions = globalValidateOptions.value

    start = timer()

    frameNumber=record[0]
    print "DEBUG: Finding new cell contour for frame %d"%(frameNumber)
    xc=record[1][0]
    yc=record[1][1]
    I=record[1][2]
    Ix0=record[1][3]
    Iy0=record[1][4]

    # Find an initial guess for the contour using the Kvarnstrom contour extraction.
    cell=kvarn.findCellContour(I, Ix0, Iy0, xc, yc, 100.0, M=contourPoints)

    # Validate the cell.
    if not contour.validateCellContour(cell, showIntermediates=True, **validateOptions): return []

    # Refine the initial guess for the contour using the complete cell model.
    cell=contour.refineCellContour(cell, neighboringCells=previousCells, tolerance=tolerance)

    # Validate the cell.
    if not contour.validateCellContour(cell, showIntermediates=True, **validateOptions): return []

    # Generate a new id for the cell.
    cell.id = uuid.uuid4().hex

    #print "Map findNewCells %d took %0.3f seconds."%(frameNumber,timer()-start)

    # Return the cells.
    return [(frameNumber,cell)]


def extractExistingCells(record, previousCells):

    global globalRefImageData

    start = timer()

    frameNumber=record[0]

    # Load the global variables.
    (refImage,refBgMean,refBgStd,refGradCutoff,refBead)=globalRefImageData.value

    # Extract the record.
    IZ=record[1][1]
    I=np.reshape(np.fromstring(zlib.decompress(IZ), dtype=np.uint8), refImage.shape)

    # Go through all of the centers.
    existingCellImages=[]
    for cell in previousCells:

        # Extract the cell image into a separate buffer.
        imCell = imaging.extractImageRegion(I, cell.Ix0, cell.Iy0, cell.Ix0+600, cell.Iy0+600, int(refBgMean))

        # Add this cell to the list.
        existingCellImages.append((frameNumber,(cell,imCell)))

    #print "Map extractExistingCells %d took %0.3f seconds."%(frameNumber,timer()-start)

    # Return the images sections to be processed.
    return existingCellImages

def findExistingCellContour(record, previousCells):

    global globalTolerance
    global globalValidateOptions
    tolerance = globalTolerance.value
    validateOptions = globalValidateOptions.value

    start = timer()

    frameNumber=record[0]
    print "DEBUG: Finding existing cell contour for frame %d"%(frameNumber)
    previousCell=record[1][0]
    I=record[1][1]

    # Copy the cell from the previous frame and set its image to be from the current frame.
    cell=copy.deepcopy(previousCell)
    cell.I = I

    # Refine the initial guess for the contour using the complete cell model.
    cell=contour.refineCellContour(cell, neighboringCells=previousCells, tolerance=tolerance)

    # Validate the cell.
    if not contour.validateCellContour(cell, showIntermediates=True, **validateOptions): return []

    #print "Map findExistingCellContour %d took %0.3f seconds."%(frameNumber,timer()-start)

    # Return the cells.
    return [(frameNumber,cell)]


def formatCellContours(record):

    start = timer()

    # Extract the record.
    frameNumber=record[0]
    cell=record[1]

    # Create an sfile header for the record.
    sfileHeader = ("/%d/%s"%(frameNumber,cell.id),"protobuf:robertslab.pbuf.imaging.yeast.CellContour")

    # Create a protobuf CellContour object.
    obj = CellContour_pb2.CellContour()
    obj.frame = frameNumber
    obj.id = cell.id
    obj.xc = cell.Ix0 + cell.xc
    obj.yc = cell.Iy0 + cell.yc
    obj.thetas.extend(cell.thetas.tolist())
    obj.rs.extend(cell.rs.tolist())

    # Serialize the data into an sfile record.
    sfileRecord = bytearray(obj.SerializeToString())

    print "Map formatCellContours took %0.3f seconds."%(timer()-start)

    # Return the aligned image.
    return (sfileHeader,sfileRecord)

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print "Usage: aligned_image_sfile index_file reference_image output_contour_file [options]"
        print "options: --contour-points=number_points"
        print "         --image-mask=x1,y1,x2,y2"
        print "         --frame-range=first-last"
        print "         --segmentation_tasks=number_tasks"
        print "         --interactive"
        quit()
    
    imageFilename = sys.argv[1]
    indexFilename = sys.argv[2]
    referenceImageFilename = sys.argv[3]
    outputFilename = sys.argv[4]
    interactive=False
    contourPointsArg=50
    toleranceArg=1e-2
    imageMaskArg=None
    frameRangeArg=None
    segmentationTasksArg=1
    if "BS_JOB_TOTAL_CORES" in os.environ:
        segmentationTasksArg = int(os.environ["BS_JOB_TOTAL_CORES"])
        print "Got number of segmentation tasks from environment: %d"%segmentationTasksArg


    for arg in sys.argv[5:]:
        if arg == "--interactive":
            interactive=True
        elif arg.startswith("--contour-points="):
            m=re.search("--contour-points=(\d+)",arg)
            if m is not None:
                contourPointsArg=int(m.group(1))
            else:
                print "Invalid option: %s"%arg
                quit()
        elif arg.startswith("--tolerance="):
            m=re.search("--tolerance=([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",arg)
            if m is not None:
                toleranceArg=float(m.group(1))
            else:
                print "Invalid option: %s"%arg
                quit()
        elif arg.startswith("--image-mask"):
            m=re.search("--image-mask=(\d+),(\d+),(\d+),(\d+)",arg)
            if m is not None:
                imageMaskArg=(int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)))
            else:
                print "Invalid option: %s"%arg
                quit()
        elif arg.startswith("--frame-range"):
            m=re.search("--frame-range=(\d+)-(\d+)",arg)
            if m is not None:
                frameRangeArg=(int(m.group(1)),int(m.group(2)))
            else:
                print "Invalid option: %s"%arg
                quit()
        elif arg.startswith("--segmentation_tasks="):
            m=re.search("--segmentation_tasks=(\d+)",arg)
            if m is not None:
                segmentationTasksArg=int(m.group(1))
            else:
                print "Invalid option: %s"%arg
                quit()
        elif arg.startswith('-o='):
            m = re.match(r'-o=(\S+)', arg)
            if m:
                outputFilename = m.group(1) + '.pkl'
        elif arg.startswith('--suffix='):
            m = re.match(r'--suffix=(\S+)', arg)
            if m:
                outputName,outputExt = os.path.splitext(outputFilename)
                outputFileName = outputName + m.group(1) + outputExt
        else:
            print "Unknown option: %s"%arg
            quit()

    # Configure Spark
    conf = SparkConf().setAppName("Yeast Segmentation")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc, imageFilename, indexFilename, referenceImageFilename, outputFilename, contourPointsArg, toleranceArg, imageMaskArg, frameRangeArg, validateOptions, segmentationTasksArg, interactive)
