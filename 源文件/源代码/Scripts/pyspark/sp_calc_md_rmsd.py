## Spark Application - execute with spark-submit, e.g.
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars <your-robertslab-git-repo>/Code/Hadoop/lib/robertslab-hadoop.jar rmsdSpark.py <hdfs-path>/3d6z_aligned_vmd_wat_ion.sfmd <local-filesystem-path>/3d6z_aligned_vmd_wat_ion.pdb "name CA"
## spark-submit --master yarn-client --num-executors 16 --driver-memory 1g --executor-memory 1g --executor-cores 1 --jars <your-robertslab-git-repo>/Code/Hadoop/lib/robertslab-hadoop.jar rmsdSpark.py <hdfs-path>/3d6z_aligned_vmd_wat_ion__CA_only.sfmd <local-filesystem-path>/3d6z_aligned_vmd_wat_ion.pdb --presliced
## or
## spark-submit-robertslab <path-to-this-script>/rmsdSpark.py <hdfs-path>/3d6z_aligned_vmd_wat_ion.sfmd <local-filesystem-path>/3d6z_aligned_vmd_wat_ion.pdb --selection name_CA
## spark-submit-robertslab <path-to-this-script>/rmsdSpark.py <hdfs-path>/3d6z_aligned_vmd_wat_ion__CA_only.sfmd <local-filesystem-path>/3d6z_aligned_vmd_wat_ion.pdb --presliced
## or
## sparkSubmit rmsdSpark <hdfs-path>/3d6z_aligned_vmd_wat_ion__CA_only.sfmd <local-filesystem-path>/3d6z_aligned_vmd_wat_ion.pdb --presliced
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
from robertslab.helper.parseHelper import ArgProperty, ArgGroup, ConversionFuncs, Parser


# Global variables
g_struct = None
g_align = None
g_selection = None
g_sliceFrames = None

def main_spark(sc, outputFile, trajectoryFile, structureFile, selection, align, sliceFrames):

    global g_align, g_selection, g_sliceFrames, g_struct

    # Broadcast the global variables.
    g_struct = sc.broadcast(mdtraj.load(structureFile))
    g_align = sc.broadcast(align)
    g_selection = sc.broadcast(selection)
    g_sliceFrames = sc.broadcast(sliceFrames)

    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(trajectoryFile, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # Calculate the per-frame rmsd via a map and collect the results.
    mappedRecords = allRecords.filter(filterFrames).map(calculateRmsd).collect()

    saveRecords(outputFile, mappedRecords)

def main_local(outputFile, trajectoryFile, structureFile, selection, align, sliceFrames):

    global g_align, g_selection, g_sliceFrames, g_struct

    # Broadcast the global variables.
    g_struct = SparkLocalGlobal(mdtraj.load(structureFile))
    g_align = SparkLocalGlobal(align)
    g_selection = SparkLocalGlobal(selection)
    g_sliceFrames = SparkLocalGlobal(sliceFrames)

    # Open the file.
    fp = SFile.fromFilename(trajectoryFile, "r")

    # Loop through the records.
    mappedRecords = []
    while True:
        rawRecord = fp.readNextRecord()
        if rawRecord is None: break

        # Filter the record.
        sparkRecord = [(rawRecord.name,rawRecord.dataType)]
        if filterFrames(sparkRecord):
            sparkRecord.append(fp.readData(rawRecord.dataSize))
            mappedRecords.append(calculateRmsd(sparkRecord))
        else:
            fp.skipData(rawRecord.dataSize)

    saveRecords(outputFile, mappedRecords)

def filterFrames(record):

    # Extract the data.
    (name,dataType)=record[0]

    # Make sure the record and and type are correct.
    if dataType == SFileMD.pbufDataType:
        return True
    return False

def calculateRmsd(record):

    global g_align, g_selection, g_sliceFrames, g_struct

    # Get the global variables.
    align = g_align.value
    selection = g_selection.value
    sliceFrames = g_sliceFrames.value
    struct = g_struct.value

    # Parse the data.
    (name,dataType)=record[0]
    data=record[1]

    # Extract the frame number.
    m=re.match("/Frames/(\d+)",name)
    if m is None:
        raise ValueError("Invalid record, no frame number.")
    frameNumber=int(m.group(1))

    # calculate the rmsd
    rmsd = RMSD(align=align, frames=[Frame(data, frameNumber)], refFrame=struct.xyz[0], selection=selection, sliceFrames=sliceFrames, top=struct.top)
    rmsdValue = rmsd.calculate()

    # Store units in Angstroms.
    return (frameNumber,rmsdValue*10.0)

def saveRecords(outputFilename, records):

    data={}

    # Create the rmsd array.
    data["/RMSD"] = np.zeros((len(records),))

    # Convert the bins into pdfs.
    for (frameNumber,rmsdValue) in records:
        data["/RMSD"][frameNumber] = rmsdValue

    # Output the pdfs.
    cellio.cellsave(outputFilename, data, format="hdf5")
    print "Saved %d RMSD values to %s"%(len(records),outputFilename)


def main():

    parser = ArgParser(description='Calculates the RMSD of a molecular dynamics trajectory using pyspark.')
    args = parser.parseArguments()

    # Execute Main functionality
    if not args.local:
        conf = SparkConf().setAppName("MD RMSD")
        sc = SparkContext(conf=conf)
        main_spark(sc, outputFile=parser.getOutputFile(), trajectoryFile=args.trajectory_sfile, structureFile=args.structure_file, selection=args.selection, align=(not args.prealigned), sliceFrames=(not args.presliced))
    else:
        main_local(outputFile=parser.getOutputFile(), trajectoryFile=args.trajectory_sfile, structureFile=args.structure_file, selection=args.selection, align=(not args.prealigned), sliceFrames=(not args.presliced))

class ArgParser(Parser):

    def genDefaultArgGroups(self):

        requiredArgs = ArgGroup('required',
                            ArgProperty('output_file',                                help='Path for a file to save the output into, in HDF5 format.'),
                            ArgProperty('trajectory_sfile',                           help='Path to an SFile containing a molecular dynamics trajectory.'),
                            ArgProperty('structure_file',                             help='Path to a structure file (e.g., .pdb) that corresponds to the trajectory file. The file must be in a format readable by the mdtraj package.'))

        SelectionConversion = ConversionFuncs.delimiterSubClosure('_', ' ')
        optionalArgs = ArgGroup('optional',
                            ArgProperty('-s','--selection', type=SelectionConversion, help='Atoms to use in the RMSD calculation, specified via the mdtraj atom selection syntax'),
                            ArgProperty('-pa','--prealigned', action='store_true',    help='The trajectory has already been aligned and doesn\'t need to be aligned again during the RMSD calculation.'),
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
