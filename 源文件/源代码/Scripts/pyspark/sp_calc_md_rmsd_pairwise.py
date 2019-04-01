## Spark Application - execute with sparkSubmit, e.g.
## sparkSubmit rmsdNxNSpark /user/tel/test/sfile/md/3d6z_aligned_vmd_wat_ion__protein_only.sfmd ~/git/roberts-lab/Code/Python/robertslab/md/test/testData/3d6z_aligned_vmd_wat_ion.pdb
from __future__ import division

try:
    from pyspark import SparkConf, SparkContext
except:
    print "No spark libraries found, ensure you are running locally."
    from robertslab.spark.helper import SparkLocalGlobal
    from robertslab.sfile import SFile

from pathlib import Path
import mdtraj
import numpy as np
import re

import robertslab.cellio as cellio
from robertslab.md.mdIO import Frame, SFileMD
from robertslab.md.rmsd.rmsd import RMSD
from robertslab import ndarray
from robertslab.helper.parseHelper import ArgProperty, ArgGroup, ConversionFuncs, Parser

# Global variable "declarations"
g_struct = None
g_selection = None
g_sliceFrames = None
g_windowSize = None

def main_spark(sc, outputFile, trajectoryFile, structureFile, selection, windowSize, sliceFrames):

    global g_selection, g_sliceFrames, g_struct, g_windowSize

    # Broadcast the global variables.
    g_struct = sc.broadcast(mdtraj.load(structureFile))
    g_selection = sc.broadcast(selection)
    g_sliceFrames = sc.broadcast(sliceFrames)
    g_windowSize = sc.broadcast(windowSize)

    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(trajectoryFile, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Extract the frames by window.
    allFrames = allRecords.filter(isFrame).map(extractSelectionByWindow)

    # Average the frames by window.
    avgFrames = allFrames.combineByKey(createAverage,mergeIntoAverage,combineAverages).map(calculateAverage)

    # Perform the pairwise rmsd calculation.
    rmsds = avgFrames.cartesian(avgFrames).filter(isUpperTriangle).map(calculatePairwiseRmsd).collect()

    # Save the records.
    saveRecords(outputFile, rmsds)

def main_local(outputFile, trajectoryFile, structureFile, selection, windowSize, sliceFrames):

    global g_selection, g_sliceFrames, g_struct, g_windowSize

    # Broadcast the global variables.
    g_struct = SparkLocalGlobal(mdtraj.load(structureFile))
    g_selection = SparkLocalGlobal(selection)
    g_sliceFrames = SparkLocalGlobal(sliceFrames)
    g_windowSize = SparkLocalGlobal(windowSize)

    # Open the file.
    fp = SFile.fromFilename(trajectoryFile, "r")

    # Loop through the records.
    recordsByWindow = {}
    while True:
        rawRecord = fp.readNextRecord()
        if rawRecord is None: break

        # Accumulate the frames.
        sparkRecord = [(rawRecord.name,rawRecord.dataType)]
        if isFrame(sparkRecord):
            sparkRecord.append(fp.readData(rawRecord.dataSize))
            record = extractSelectionByWindow(sparkRecord)
            if record[0] not in recordsByWindow:
                recordsByWindow[record[0]] = createAverage(record[1])
            else:
                recordsByWindow[record[0]] = mergeIntoAverage(recordsByWindow[record[0]],record[1])
        else:
            fp.skipData(rawRecord.dataSize)

    # Calculate the average frames.
    avgByWindow = {}
    for windowNumber in recordsByWindow.keys():
        avgByWindow[windowNumber] = calculateAverage((windowNumber,recordsByWindow[windowNumber]))

    # Calculate the pairwise rmsds.
    rmsds = []
    for windowNumber1 in avgByWindow.keys():
        for windowNumber2 in avgByWindow.keys():
            recordPair = (avgByWindow[windowNumber1],avgByWindow[windowNumber2])
            if isUpperTriangle(recordPair):
                print "Calculating %d/%d"%(recordPair[0][0],recordPair[1][0])
                rmsds.append(calculatePairwiseRmsd(recordPair))

    saveRecords(outputFile, rmsds)

def isFrame(record):

    # Extract the data.
    (name,dataType)=record[0]

    # Make sure the record and and type are correct.
    if dataType == SFileMD.pbufDataType:
        return True
    return False

def extractSelectionByWindow(record):

    global g_struct, g_selection, g_windowSize

    # Get the global variables.
    struct = g_struct.value
    selection = g_selection.value
    windowSize = g_windowSize.value

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # Extract the frame number.
    m=re.match("/Frames/(\d+)",name)
    if m is None:
        raise ValueError("Invalid record, no frame number.")
    frameNumber=int(m.group(1))

    # Figure out the window number.
    windowNumber = frameNumber//windowSize

    # Get the full frame.
    allCoords = ndarray.deserialize(data)

    # Find the atom indices to use for the selection.
    atomIndices = struct.top.select(selection)
    atomCoords = allCoords[atomIndices]

    # Store the selection.
    return (windowNumber,atomCoords)

def createAverage(coords):
    return (1,coords)

def mergeIntoAverage(avg, coords):
    return (avg[0]+1,avg[1]+coords)

def combineAverages(avg1, avg2):
    return (avg1[0]+avg2[0],avg1[1]+avg2[1])

def calculateAverage(avgRecord):
    (windowNumber,avg) = avgRecord
    return (windowNumber,avg[1]/avg[0])

def isUpperTriangle(recordPair):
    ((windowNumber1,coords1),(windowNumber2,coords2)) = recordPair
    return windowNumber1 > windowNumber2

def calculatePairwiseRmsd(recordPair):
    ((windowNumber1,coords1),(windowNumber2,coords2)) = recordPair
    rmsd = RMSD(align=True, refFrame=Frame(coords1), frames=[Frame(coords2)])
    rmsdValue = rmsd.calculate()
    return ((windowNumber1,windowNumber2),rmsdValue*10.0)

def saveRecords(outputFilename, records):

    data={}

    # Figure out the number of windows.
    maxWindow = -1
    for ((i,j),rmsdValue) in records:
        if i > maxWindow:
            maxWindow = i
    numberWindows = maxWindow+1

    # Create the rmsd array.
    rmsd = np.zeros((numberWindows,numberWindows))

    # Fill in the rmsd array.
    for ((i,j),rmsdValue) in records:
        rmsd[i,j] = rmsdValue
        rmsd[j,i] = rmsdValue

    # Output the pdfs.
    data["/RMSD"] = rmsd
    cellio.cellsave(outputFilename, data, format="hdf5")
    print "Saved RMSD values from %d windows to %s"%(numberWindows,outputFilename)

def main():

    parser = ArgParser(description='Calculates the pairwise RMSD between every frame (or window of frames) of a molecular dynamics trajectory using pyspark.')
    args = parser.parseArguments()

    # Execute Main functionality
    if not args.local:
        conf = SparkConf().setAppName("MD Pairwise RMSD")
        sc = SparkContext(conf=conf)
        main_spark(sc, outputFile=parser.getOutputFile(), trajectoryFile=args.trajectory_sfile, structureFile=args.structure_file, selection=args.selection, windowSize=args.window_size, sliceFrames=(not args.presliced))
    else:
        main_local(outputFile=parser.getOutputFile(), trajectoryFile=args.trajectory_sfile, structureFile=args.structure_file, selection=args.selection, windowSize=args.window_size, sliceFrames=(not args.presliced))

class ArgParser(Parser):

    def genDefaultArgGroups(self):

        requiredArgs = ArgGroup('required',
                                ArgProperty('output_file',                                help='Path for a file to save the output into, in HDF5 format.'),
                                ArgProperty('trajectory_sfile',                           help='Path to an SFile containing a molecular dynamics trajectory.'),
                                ArgProperty('structure_file',                             help='Path to a structure file (e.g., .pdb) that corresponds to the trajectory file. The file must be in a format readable by the mdtraj package.'))

        SelectionConversion = ConversionFuncs.delimiterSubClosure('_', ' ')
        optionalArgs = ArgGroup('optional',
                                ArgProperty('-s','--selection', type=SelectionConversion, help='Atoms to use in the RMSD calculation, specified via the mdtraj atom selection syntax.'),
                                ArgProperty('-w','--window-size', type=int, default=1,    help='Number of frames to average for each comparison window (default=1).'),
                                ArgProperty('-ps','--presliced', action='store_true',     help='The trajectory has already been sliced and it doesn\'t need to be sliced again during the RMSD calculation.'),
                                ArgProperty('-sf','--suffix',                             help='A string to be added to the output filename, before the file extension. Useful for saving the output from multiple replicates/parameter sweeps.'),
                                ArgProperty('--local', action='store_true',               help='Execute the analysis locally instead of using spark.'))

        return [requiredArgs, optionalArgs]

    def getOutputFile(self):
        outputFile = self.args.output_file
        if self.args.suffix is not None:
            outputFile = Path(outputFile).stem + self.args.suffix + Path(outputFile).suffix
        return outputFile



if __name__ == "__main__":
    main()




