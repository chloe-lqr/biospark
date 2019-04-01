import os
import time

class SparkLocalGlobal:
    def __init__(self, value):
        self.value = value

def newAPIHadoopFileConfig():
    conf={}
    if "BS_JOB_SPLIT_SIZE" in os.environ:
        conf["mapreduce.input.fileinputformat.split.minsize"] = os.environ["BS_JOB_SPLIT_SIZE"]
        print time.strftime("%y/%m/%d %H:%M:%S"),"INFO Biospark: Set input file split size: %s"%(conf["mapreduce.input.fileinputformat.split.minsize"])
    return conf
