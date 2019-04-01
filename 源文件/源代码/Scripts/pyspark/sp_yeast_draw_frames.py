## Spark Application - execute with spark-submit, e.g.
## hdfs dfs -rm -r -f /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out && spark-submit --master yarn-client --num-executors 4 --driver-memory 2g --executor-memory 2g --executor-cores 1 --jars ~/Work/Lab/Git/roberts-lab/Code/Hadoop/lib/robertslab-hadoop.jar ~/Work/Lab/Git/roberts-lab/Scripts/pyspark/sp_align_frames.py /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/test2.sfile index.txt reference.png /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out/test.sfile /user/erober32/Work/Lab/Yeast_Growth/segmentation_test/out/test.txt no
##

import csv
from pyspark import SparkConf, SparkContext, StorageLevel
import sys
import re
import sys
import zlib

import matplotlib.pyplot as plt
from skimage import io
from robertslab.sfile import *
import robertslab.imaging.yeast.contour as contour
import robertslab.pbuf.NDArray_pb2 as NDArray_pb2
import robertslab.pbuf.imaging.yeast.CellContour_pb2 as CellContour_pb2

# Global variables.
globalFrameIndex=None
globalScale=None

def main(sc, framesFilename, indexFilename, contoursFilename, scale, outputDir, outputPrefix):
    
    global globalFrameIndex, globalScale

    # Set any global variables.
    globalScale=sc.broadcast(scale)

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

    # Load the contours from the sfile.
    rawContours = sc.newAPIHadoopFile(contoursFilename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")
    contours = rawContours.map(extractContour).groupByKey()
    contours.cache()

    # Get a list of all the frames with contours.
    frameList=contours.keys().collect()

    # Load the frames from the sfile.
    allFrames = sc.newAPIHadoopFile(framesFilename, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Match the frames and contours.
    framesAndContours = allFrames.filter(lambda x: filterFrame(x,frameList)).map(extractFrame).join(contours)

    images = framesAndContours.map(drawFrame).collectAsMap()
    for frame in images.keys():
        io.imsave("%s/%s%06d.png"%(outputDir,outputPrefix,frame), images[frame])
    
def filterFrame(record, frameList):

    global globalFrameIndex

    # Extract the data.
    (name,dataType)=record[0]

    # Load the global variables.
    frameIndex = globalFrameIndex.value

    # Make sure the record and and type are correct.
    m=re.search("frame-aligned-(\d+)",name)
    if m == None or dataType != "protobuf:robertslab.pbuf.NDArray":
        return False
    frameNumber=int(m.group(1))

    # Make sure the frame is in the list.
    if frameNumber not in frameList:
        return False

    # Otherwise, we will process this frame.
    return True

def extractContour(record):

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # Extract the frame number.
    m=re.match("/(\d+)/(\w+)",name)
    if m is None or dataType != "protobuf:robertslab.pbuf.imaging.yeast.CellContour":
        return None
    frameNumber=int(m.group(1))
    cellId = m.group(2)

    print "Extracting frame %d (%d)"%(frameNumber,len(data))

    # Extract the array.
    obj = CellContour_pb2.CellContour()
    obj.ParseFromString(str(data))
    cell = contour.CellContour(obj.xc, obj.yc, np.array(obj.thetas), np.array(obj.rs), None, 0, 0)
    cell.id = obj.id

    # Return the aligned image.
    return (frameNumber,cell)

def extractFrame(record):

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # Extract the frame number.
    m=re.search("frame-aligned-(\d+)",name)
    if m is None or dataType != "protobuf:robertslab.pbuf.NDArray":
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

    # Create the numpy array for the data.
    if obj.compressed_deflate:
        im=np.reshape(np.fromstring(zlib.decompress(obj.data), dtype=np.uint8), tuple(obj.shape))
    else:
        im=np.reshape(np.fromstring(obj.data, dtype=np.uint8), tuple(obj.shape))

    # Return the aligned image.
    return (frameNumber,im)

def drawFrame(record):

    global globalScale

    # Get any global variables.
    scale = globalScale.value

    # Parse the data.
    frameNumber = record[0]
    im = record[1][0]
    cells = record[1][1]

    # Setup the figure.
    fig,ax = plt.subplots()
    fig.set_dpi(100)
    fig.set_size_inches(scale*im.shape[1]/100.0, scale*im.shape[0]/100.0)

    # Plot the image.
    ax.imshow(im, vmin=0, vmax=255, cmap="gray")
    ax.set_xlim([0,im.shape[1]])
    ax.set_ylim([im.shape[0],0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Draw the cell contours.
    for cell in cells:
        #ax.scatter(cell.xc,cell.yc,color='r',marker='x')
        ax.plot(cell.xs,cell.ys,color='r')
        ax.plot([cell.xs[0],cell.xs[-1]],[cell.ys[0],cell.ys[-1]],color='r')
        ax.text(cell.bounds[2], cell.bounds[3], cell.id[:3], fontsize=12, color='r', ha='left', va='top')

    # Extract the image data from the pyplot figure.
    fig.tight_layout(pad=0.0)
    fig.canvas.draw()
    cols,rows = fig.canvas.get_width_height()
    im2 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(rows,cols,3)

    return (frameNumber,im2)


if __name__ == "__main__":
    
    if len(sys.argv) < 7:
        print "Usage: frames_sfile index_file cell_countours_sfile output_dir [options]"
        print "options: --scale=scaling_factor (e.g., 0.5)"
        print "         --prefix=filename_prefix"
        quit()
    
    framesFilenameArg = sys.argv[1]
    indexFilenameArg = sys.argv[2]
    contoursFilenameArg = sys.argv[3]
    outputDirArg = sys.argv[4]
    scaleArg = 1.0
    outputPrefixArg = ""
    for arg in sys.argv[5:]:
        if arg.startswith("--scale="):
            m=re.search("--scale=([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)",arg)
            if m is not None:
                scaleArg = float(m.group(1))
        elif arg.startswith('--prefix='):
            m = re.match(r'--prefix=(\S+)', arg)
            if m is not None:
                outputPrefixArg = m.group(1)
        else:
            print "Invalid option: %s"%arg
            quit()

    # Configure Spark
    conf = SparkConf().setAppName("Draw Frames")
    sc = SparkContext(conf=conf)

    # Execute Main functionality
    main(sc, framesFilenameArg, indexFilenameArg, contoursFilenameArg, scaleArg, outputDirArg, outputPrefixArg)
