from copy import deepcopy
from distutils.errors import DistutilsExecError, DistutilsOptionError, DistutilsSetupError
import glob
import numpy as np
import os,sys

from . import setupHelper
Tupify = setupHelper.Tupify

thisScriptDir = os.path.dirname(os.path.realpath(__file__))

class DeploymentOptions(object):
    # more info at https://docs.python.org/2/distutils/apiref.html
    targetName = ''

    extensionOpts = {'define_macros':(),
                     'extra_compile_args':(),
                     'extra_link_args':(),
                     'extra_objects':(),
                     'include_dirs':(np.get_include(), './robertslab'),
                     'libraries':(),
                     'library_dirs':(),
                     'runtime_library_dirs':(),
                     'undef_macros':()}

    @classmethod
    def getExtensionOpts(cls):
        return {key:list(val) for key,val in cls.extensionOpts.items() if val is not ()}

class DefaultDO(DeploymentOptions):
    targetName = 'default'

    extensionOpts = deepcopy(DeploymentOptions.extensionOpts)
    extensionOpts['extra_compile_args'] += ('-pthread', '-fno-strict-aliasing', '-DNDEBUG', '-fwrapv', '-O3', '-Wall', '-Wstrict-prototypes', '-fPIC')
    extensionOpts['extra_link_args']    += ('-pthread',)

class XanthusDO(DeploymentOptions):
    targetName = 'xanthus'

    extensionOpts = deepcopy(DeploymentOptions.extensionOpts)
    extensionOpts['extra_compile_args'] += ('-pthread', '-fno-strict-aliasing', '-DNDEBUG', '-fwrapv', '-march=corei7-avx', '-mavx', '-O3', '-Wall', '-Wstrict-prototypes', '-fPIC')
    extensionOpts['extra_link_args']    += ('-pthread', '-shared', '-Wl,-O1', '-Wl,-Bsymbolic-functions', '-Wl,-Bsymbolic-functions', '-Wl,-z,relro')
    extensionOpts['extra_objects']      += ('/share/apps/src/zlib-cf/libz.a',)
    extensionOpts['include_dirs']       += ('/share/apps/src/zlib-cf',)

class MarccDO(DeploymentOptions):
    targetName = 'marcc'

    extensionOpts = deepcopy(DeploymentOptions.extensionOpts)
    extensionOpts['extra_compile_args'] += ('-pthread', '-fno-strict-aliasing', '-DNDEBUG', '-fwrapv', '-march=corei7-avx', '-mavx', '-O3', '-Wall', '-Wstrict-prototypes', '-fPIC')
    extensionOpts['extra_link_args']    += ('-pthread', '-shared', '-Wl,-O1', '-Wl,-Bsymbolic-functions', '-Wl,-Bsymbolic-functions', '-Wl,-z,relro')
    # extensionOpts['extra_objects']      += ('/share/apps/src/zlib-cf/libz.a',)
    # extensionOpts['include_dirs']       += ('/share/apps/src/zlib-cf',)

doDict = {cls.targetName:cls for cls in DeploymentOptions.__subclasses__()}

def GetDataFileTuples(relPath, fsRootPath, sitePath, fNames):
    # note: fs ~ file system
    relPath,fsRootPath,sitePath = Tupify(relPath),Tupify(fsRootPath),Tupify(sitePath)

    absFSDir = os.path.join(*(fsRootPath + relPath))
    absSiteDir = os.path.join(*(sitePath + relPath))

    # nested list comp deals with soft links
    return [(absSiteDir,[os.path.realpath(fPath) for fPath in glob.glob(os.path.join(absFSDir, fName))]) for fName in fNames]