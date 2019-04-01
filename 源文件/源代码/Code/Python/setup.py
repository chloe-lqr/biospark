#!/usr/bin/env python
#PYTHON_ARGCOMPLETE_OK

# example install:
# `pip install -r requirements.txt`
# `pip install .` OR `pip install . --install-option="--deployment=xanthus"`
# `./setupTabCompletion.py`
from __future__ import print_function
import os,sys
sys.path.insert(0, '.')

from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_lib
from setuptools import Extension, find_packages, setup

from setupUtils.setupParser import SetupParser
from setupUtils.setupUtils import doDict, GetDataFileTuples

# create and use the option parser
parser = SetupParser()
parser.run()
do = doDict[parser['deployment']]

# get the paths to this script and the python site dir
thisScriptDir = os.path.dirname(os.path.realpath(__file__))
if 'PYTHONUSERBASE' in os.environ:
    siteDir = os.path.realpath(get_python_lib(prefix=os.environ['PYTHONUSERBASE']))
else:
    siteDir = os.path.realpath(get_python_lib())

# get all of the information required to install the non-python files
jars_data_files = GetDataFileTuples(relPath=('robertslab', 'jars', 'hadoop'), fsRootPath=thisScriptDir,
                                    sitePath=siteDir, fNames=('*.jar',))

mdTestData_data_files = GetDataFileTuples(relPath=('robertslab', 'md', 'test', 'testData'), fsRootPath=thisScriptDir,
                                          sitePath=siteDir, fNames=('*.dcd', '*.pdb'))

ncbi_data_files = GetDataFileTuples(relPath=('robertslab', 'ncbi'), fsRootPath=thisScriptDir,
                                    sitePath=siteDir, fNames=('*.sh', '*.sql'))

rawdata_data_files = GetDataFileTuples(relPath=('robertslab', 'rawdata'), fsRootPath=thisScriptDir,
                                       sitePath=siteDir, fNames=('*.sql',))

# the actual setup function
setup(
    # setup_requires/install_requires don't work so great in most cases, so we don't use them
    #install_requires = ['mdtraj >= 1.5 --global-option="--disable-openmp"'],
    #setup_requires = ['numpy'],

    author = 'Elijah Roberts, Max Klein',
    cmdclass = {'build_ext': build_ext},
    data_files = jars_data_files +
                 mdTestData_data_files +
                 ncbi_data_files +
                 rawdata_data_files,
    description = 'general python modules for projects in the Roberts Lab',
    entry_points={'console_scripts': ['sparkSubmit = robertslab.spark.sparkSubmit:Main',
                                      'trajectoryToSFile = robertslab.md.mdIO.trajectoryToSFile:Main']},
    ext_modules = [Extension('robertslab.rzlib',
                             language="c++",
                             sources=['./robertslab/rzlib.pyx', './robertslab/rzlib_c.cpp'],
                             **do.getExtensionOpts()
                             )],
    license = 'UOISL',
    name = 'robertslab',
    packages = find_packages(exclude=('setupUtils',)),
    #scripts = ['robertslab/md/src/rmsd/spark/rmsdSpark.py']
)