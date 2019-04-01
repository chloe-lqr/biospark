#!/usr/bin/env python3
#PYTHON_ARGCOMPLETE_OK
import numpy as np
from operator import iadd
from pathlib import Path

from lm_anal.src.datum.oparam import OParams
from lm_anal.src.datum.hist import OParamHist, OParamHists
from lm_anal.src.datum.trajectory import SpeciesTrajectories
from lm_anal.src.datum.tiling import Tilings
from lm_anal.src.serializable.serializable import Serializable
from robertslab.helper.sparkProfileHelper import SparkProfileDecorator
from robertslab.helper.timerHelper import TimeDecoratorSpark,TimeWithSpark
from robertslab.helper.parseHelper import ArgProperty, ArgGroup
from robertslab.spark.sparkParser import SparkParser

from robertslab.helper.pysparkImportHelper import PysparkImport
pyspark = PysparkImport()
from pyspark import SparkConf, SparkContext

# Global variables.
g_compressed=None
g_oparams=None
g_skipTime=None
g_tilings=None
g_tilingIDs=None

def _Main(sc, sfilePath, lmPath, outPath, tilingIDs, partition=False, profiled=False, skip_less_than=0.0, suffix=None):
    global g_compressed,g_oparams,g_skipTime,g_tilings,g_tilingIDs

    # generate the path for the output
    if suffix is not None and suffix not in outPath:
        outPath = Path(outPath).stem + suffix + Path(outPath).suffix

    # setup oparams and tilings while bypassing lazy loading
    oparams = OParams(fPath=lmPath, lazyLoad=False)
    tilings = Tilings(fPath=lmPath, lazyLoad=False)

    # Broadcast the global variables.
    g_compressed = sc.broadcast(False)
    g_oparams = sc.broadcast(oparams)
    g_skipTime = sc.broadcast(skip_less_than)
    g_tilings = sc.broadcast(tilings)
    g_tilingIDs = sc.broadcast(tilingIDs)

    profiler = SparkProfileDecorator(sc=sc)
    if profiled:

        timeDecorator = TimeDecoratorSpark(sc=sc, count=5)
        if partition:
            timeWith = TimeWithSpark(sc=sc, names=['get broadcast values', 'alloc trajectories Data', 'alloc oparamHists Data', 'index record', 'deserialize trajectories', 'transform trajectories to oparamHists', 'sum and serialize oparamHists'])
        else:
            timeWith = TimeWithSpark(sc=sc, names=['get broadcast values', 'index record', 'alloc trajectories Data', 'deserialize trajectories', 'alloc oparamHists Data', 'transform trajectories to oparamHists', 'sum and serialize oparamHists'])

        def Agg_arrArr_Timed(arr0, arr1):
            # print(arr0[10:20,10:20])
            # print(arr1[10:20,10:20])
            return iadd(arr0, arr1)

        def Agg_arrSerializedArrClosure_Timed(dtype, shape):
            def Agg_arrSerializedArr_Timed(arr, serializedArr):
                global g_compressed

                compressed = g_compressed.value
                # print(arr[10:20,10:20])
                # print(type(serializedArr))
                return iadd(arr, Serializable.deserializeArr(arrString=serializedArr, dtype=dtype, shape=shape, compressed=compressed))
            return Agg_arrSerializedArr_Timed

        def Filter_SpeciesCounts_Timed(record):
            # Extract the data.
            (name,dataType)=record[0]

            # Make sure the record and and type are correct.
            if dataType == "protobuf:lm.io.SpeciesTimeSeries":  #dataType == "protobuf:lm.io.SpeciesCounts" or
                return True
            return False

        def Map_TransformOParamHists_Timed(record):
            global g_compressed,g_oparams,g_skipTime,g_tilings,g_tilingIDs

            with timeWith('get broadcast values') as t:
                compressed = g_compressed.value
                oparams = g_oparams.value
                skipTime = g_skipTime.value
                tilings = g_tilings.value
                tilingIDs = g_tilingIDs.value

            with timeWith('index record') as t:
                (name,dataType) = record[0]
                data = record[1]

            with timeWith('alloc trajectories Data') as t:
                specTrajs = SpeciesTrajectories()
            with timeWith('deserialize trajectories') as t:
                profiler(specTrajs.deserialize)(data, trajID='uuid')
            with timeWith('alloc oparamHists Data') as t:
                oparamHists = OParamHists()
            with timeWith('transform trajectories to oparamHists') as t:
                profiler(oparamHists.transform)(specTrajs, oparams=oparams, tilings=tilings, tilingIDs=tilingIDs)

            with timeWith('sum and serialize oparamHists') as t:
                oparamHistSumArrSerialized = Serializable.serializeArr(oparamHists.genSum().h, compressed=compressed)

            return oparamHistSumArrSerialized

        def MapPartitions_TransformOParamHists_Timed(records):
            global g_compressed,g_oparams,g_skipTime,g_tilings,g_tilingIDs

            with timeWith('get broadcast values') as t:
                compressed = g_compressed.value
                oparams = g_oparams.value
                skipTime = g_skipTime.value
                tilings = g_tilings.value
                tilingIDs = g_tilingIDs.value

            with timeWith('alloc trajectories Data') as t:
                specTrajs = SpeciesTrajectories()
            with timeWith('alloc oparamHists Data') as t:
                oparamHists = OParamHists()

            for i,record in enumerate(records):
                with timeWith('index record') as t:
                    (name,dataType) = record[0]
                    data = record[1]
                with timeWith('deserialize trajectories') as t:
                    specTrajs.deserialize(data, trajID='uuid')     #, trajID=0)

            with timeWith('transform trajectories to oparamHists') as t:
                # l = 0
                # for specTraj in specTrajs:
                #     l+=specTraj.species_count.shape[0]
                longSpecCount = np.concatenate([specTraj.species_count for specTraj in specTrajs.values()])
                specTrajs.peek().species_count = longSpecCount
                specTrajs.map = {'sum': specTrajs.peek()}
                oparamHists.transform(specTrajs, oparams=oparams, tilings=tilings, tilingIDs=tilingIDs)

            with timeWith('sum and serialize oparamHists') as t:
                oparamHistSumArrSerialized = Serializable.serializeArr(oparamHists.genSum().h, compressed=compressed)

            return [oparamHistSumArrSerialized]

    # initialize an oparamHists that will hold the summed data after the spark operations return it
    oparamHists = OParamHists()
    oparamHistSum = oparamHists.initDatum('sum', full=True)
    oparamHistSum.setTilings(oparams, tilings, tilingIDs)

    # Load the records from the sfile.
    allRecords = sc.newAPIHadoopFile(sfilePath, "robertslab.hadoop.io.SFileInputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter", valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    if profiled:
        Agg_arrSerializedArr = Agg_arrSerializedArrClosure(dtype=oparamHistSum.h.dtype, shape=oparamHistSum.h.shape)

        # Bin the species counts records and sum across all of the bins.
        if partition:
            oparamHistSumArr = allRecords.filter(profiler(Filter_SpeciesCounts)).mapPartitions(profiler(MapPartitions_TransformOParamHists)).aggregate(oparamHistSum.h_raw, profiler(Agg_arrSerializedArr), profiler(Agg_arrArr))
        else:
            oparamHistSumArr = allRecords.filter(profiler(Filter_SpeciesCounts)).map(profiler(Map_TransformOParamHists)).aggregate(oparamHistSum.h_raw, profiler(Agg_arrSerializedArr), profiler(Agg_arrArr))

        profiler.print_stats()
        print(timeDecorator)
        print(timeWith)
    else:
        Agg_arrSerializedArr = Agg_arrSerializedArrClosure(dtype=oparamHistSum.h.dtype, shape=oparamHistSum.h.shape)

        # Bin the species counts records and sum across all of the bins.
        if partition:
            oparamHistSumArr = allRecords.filter(Filter_SpeciesCounts).mapPartitions(MapPartitions_TransformOParamHists).aggregate(oparamHistSum.h_raw, Agg_arrSerializedArr, Agg_arrArr)
        else:
            oparamHistSumArr = allRecords.filter(Filter_SpeciesCounts).map(Map_TransformOParamHists).aggregate(oparamHistSum.h_raw, Agg_arrSerializedArr, Agg_arrArr)

    # Save a copy of the records, for examples purposes only here.
    #records.saveAsNewAPIHadoopFile(sys.argv[1]+".copy", "robertslab.hadoop.io.SFileOutputFormat", "robertslab.hadoop.io.SFileHeader", "robertslab.hadoop.io.SFileRecord", keyConverter="robertslab.spark.sfile.PythonToSFileHeaderConverter", valueConverter="robertslab.spark.sfile.PythonToSFileRecordConverter", conf=conf)

    oparamHistSum.setH(oparamHistSumArr)
    oparamHists.wtint(fPath=outPath)
    oparamHistSum.plot()
    oparamHistSum.savefig(outPath)

def Agg_arrArr(arr0, arr1):
    # print(arr0[10:20,10:20])
    # print(arr1[10:20,10:20])
    return iadd(arr0, arr1)

def Agg_arrSerializedArrClosure(dtype, shape):
    def Agg_arrSerializedArr(arr, serializedArr):
        global g_compressed

        compressed = g_compressed.value
        # print(arr[10:20,10:20])
        # print(type(serializedArr))
        return iadd(arr, Serializable.deserializeArr(arrString=serializedArr, dtype=dtype, shape=shape, compressed=compressed))
    return Agg_arrSerializedArr

def Filter_SpeciesCounts(record):
    # Extract the data.
    (name,dataType)=record[0]

    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.SpeciesTimeSeries":  #dataType == "protobuf:lm.io.SpeciesCounts" or
        return True
    return False

def Map_TransformOParamHists(record):
    global g_compressed,g_oparams,g_skipTime,g_tilings,g_tilingIDs

    # Get the global variables.
    compressed = g_compressed.value
    oparams = g_oparams.value
    skipTime = g_skipTime.value
    tilings = g_tilings.value
    tilingIDs = g_tilingIDs.value

    # Parse the data.
    (name,dataType) = record[0]
    data = record[1]

    specTrajs = SpeciesTrajectories()
    specTrajs.deserialize(data, trajID='uuid')
    oparamHists = OParamHists()
    oparamHists.transform(specTrajs, oparams=oparams, tilings=tilings, tilingIDs=tilingIDs)

    # sum all of the histograms generated for this chunk of data
    oparamHistSumArrSerialized = Serializable.serializeArr(oparamHists.genSum().h, compressed=compressed)

    # Return the array from the summed histogram
    return oparamHistSumArrSerialized

def MapPartitions_TransformOParamHists(records):
    global g_compressed,g_oparams,g_skipTime,g_tilings,g_tilingIDs

    # Get the global variables.
    compressed = g_compressed.value
    oparams = g_oparams.value
    skipTime = g_skipTime.value
    tilings = g_tilings.value
    tilingIDs = g_tilingIDs.value

    specTrajs = SpeciesTrajectories()
    oparamHists = OParamHists()

    for i,record in enumerate(records):
        # Parse the data.
        (name,dataType) = record[0]
        data = record[1]
        specTrajs.deserialize(data, trajID='uuid')

    # longSpecCount = np.concatenate([specTraj.species_count for specTraj in specTrajs.values()])
    # specTrajs.peek().species_count = longSpecCount
    # specTrajs.map = {'sum': specTrajs.peek()}
    oparamHists.transform(specTrajs, oparams=oparams, tilings=tilings, tilingIDs=tilingIDs)

    # sum all of the histograms generated for this chunk of data
    oparamHistSumArrSerialized = Serializable.serializeArr(oparamHists.genSum().h, compressed=compressed)

    # Return the array from the summed histogram
    return [oparamHistSumArrSerialized]

def Main():
    parser = SparkParser(description='Calculation of the stationary PDF of an order parameter for a brute force Lattice Microbes simulation')
    # group of args relating to output files
    outputArgs = ArgGroup('inputArgs',
                          ArgProperty('sfilePath',                             help='hdfs path to sfile-formatted output from a brute force Lattice Microbes simulation.'),
                          ArgProperty('lmPath',                                help='path to the latice microbes input file (.lm) containing the order parameters and tilings to be used when making the PDF histogram'),
                          ArgProperty('tilingIDs', type='intRange',            help='specify which tilings (as described in your .lm file) to use as the histogram bin edges'),
                          ArgProperty('--partition', action='store_true',      help='if set, use mapPartition instead of map funcs'),
                          ArgProperty('--profiled', action='store_true',       help='if set, print timing/profiling data for the major functions in this script'),
                          ArgProperty('-s', '--skip_less_than', type=float,    help='the "burn-in" time for each trajectory'))
    parser.addArgGroup(outputArgs)
    parser.run()

    mainArgDict = parser.getArgDict()

    # Configure Spark
    conf = SparkConf().setAppName("LM Order Parameter PDF Calculation")     #.set("spark.python.profile", "true")
    sc = SparkContext(conf=conf)    #, environment={'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp'})

    # Execute Main functionality
    _Main(sc, **mainArgDict)
    # sc.show_profiles()

if __name__ == "__main__":
    Main()
