from distutils.spawn import find_executable
from importlib import import_module
import numpy as np
import os,sys
from pathlib import Path
import re

from robertslab.helper.bashHelper import SourceBash
from robertslab.helper.dictHelper import DictDiffer

__all__ = ['GetSparkConfPath', 'GetSparkDefaults', 'GetSparkEnv', 'GetSparkHomePath', 'PysparkImport', 'PysparkImportDebug']

bashGarbageKeys = ['_', '__CF_USER_TEXT_ENCODING', 'PWD', 'SHLVL']

def GetSparkConfPath():
    '''
    get the path to the spark conf dir
    '''
    if 'SPARK_CONF_DIR' in os.environ:
        return Path(os.environ['SPARK_CONF_DIR'])
    else:
        for confPath in (confPath for dPath in list(Path(find_executable('spark-submit')).resolve().parents)[:2] for confPath in dPath.rglob('conf')):
            return confPath
    raise OSError('could not find the spark conf dir. No SPARK_CONF_DIR environment var was set, so tried recursively searching in: %s' % list(Path(find_executable('spark-submit')).resolve().parents)[:2])

def GetSparkDefaults(excludeFromExecEnv=None):
    '''
    get a list with all of the lines from the default spark-defaults.conf script
    excludeFromExecEnv can be a list of environment variable names. Any lines setting these env vars in spark-defaults.conf will be excluded from the return
    '''
    excludeFromExecEnv = excludeFromExecEnv if excludeFromExecEnv is not None else []

    defaultSparkDefaultsPath = GetSparkConfPath() / 'spark-defaults.conf'
    if defaultSparkDefaultsPath.exists():
        excludeFromExecEnvRes = [re.compile('spark.executorEnv.%s' % eFEE) for eFEE in excludeFromExecEnv]

        with open(str(defaultSparkDefaultsPath), 'r') as f:
            return [line for line in f if not np.any([eFEER.search(line) for eFEER in excludeFromExecEnvRes])]
    else:
        return []

def GetSparkEnv(sanitize='envi'):
    '''
    get a dict with the environment variables that are present after sourcing the default spark-env.sh script
    if sanitize=='diff', return only the variables changed by spark-env.sh
    if sanitize=='envi', use env -i to source spark-env.sh and return the whole env starting from a completely clean environemnt
    '''
    defaultSparkEnvPath = GetSparkConfPath() / 'spark-env.sh'
    if defaultSparkEnvPath.exists():
        if sanitize=='envi':
            sparkEnvKeys = [key for key in SourceBash(defaultSparkEnvPath, sanitize='envi').keys() if key not in bashGarbageKeys]
            sparkEnvDict = SourceBash(defaultSparkEnvPath)
            return {key:sparkEnvDict[key] for key in sparkEnvKeys}
        elif sanitize=='diff':
            return DictDiffer(SourceBash(defaultSparkEnvPath), os.environ).getDiff()
        else:
            return SourceBash(defaultSparkEnvPath)
    else:
        return {}

def GetSparkHomePath():
    '''
    get the path that spark internally calls SPARK_HOME
    '''
    return GetSparkConfPath().parent

def PysparkImport(subpkgs=None):
    '''
    wrapper for importing pyspark package that allows for running pyspark scripts either normally (ie via spark-submit) or in a standalone mode for debugging purposes
    '''
    try:
        import pyspark
        if subpkgs is None:
            return pyspark
        else:
            return [import_module('.' + subpkg, 'pyspark') for subpkg in subpkgs]
    except ImportError:
        return PysparkImportDebug(subpkgs=subpkgs)

def PysparkImportDebug(subpkgs=None):
    # for when a script is run in standalone/debug mode wihoutout a spark-submit (for debugging)
    import pkg_resources

    jarPath = pkg_resources.resource_filename('robertslab.jars.hadoop', 'robertslab-hadoop.jar')
    os.environ['SPARK_CLASSPATH'] = jarPath + os.environ.get('SPARK_CLASSPATH', '')

    # spark_home = os.environ.get('SPARK_HOME', None)
    # if spark_home is None:
    #     raise ValueError('could not import SparkConf,SparkContext, and SPARK_HOME environment variable is not set. Either run your program using a spark submit script or set SPARK_HOME')
    # else:
    #     spark_path = Path(spark_home)
    spark_path = GetSparkHomePath()
    for spark_component_path in [spark_path / 'python', next(spark_path.rglob('**/py4j*src.zip'))]:
        sys.path.insert(0, str(spark_component_path))
    import pyspark
    if subpkgs is None:
        return pyspark
    else:
        return [import_module('.' + subpkg, 'pyspark') for subpkg in subpkgs]