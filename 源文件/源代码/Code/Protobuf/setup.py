# aspects of this build system are based on https://github.com/PySide/pyside-setup
from __future__ import print_function
import os,sys
sys.path.insert(0, '.')

from distutils.errors import DistutilsSetupError
from distutils.spawn import find_executable
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

from setupUtils import CopyDirStructure, InitInit, OptionValue, RGlob, RunProcess, WrappedMakedirs

# Globals! Hooray!
package_dir = {'': 'build/python',
               'lm': 'build/python/lm',
               'robertslab': 'build/python/robertslab'}
thisScriptDir = os.path.dirname(os.path.realpath(__file__))

protoSrcDir = os.path.join(thisScriptDir, 'src', 'protobuf')
protoBuildDir = os.path.join(thisScriptDir, 'build')
protoPythonBuildDir = os.path.join(protoBuildDir, 'python')

initInitPaths = [protoBuildDir]
initInitTemplatedPaths = [os.path.join(protoBuildDir, 'python', 'robertslab')]

if 'PROTOC' in os.environ:
    OPTION_PROTOC = os.environ['PROTOC']
else:
    OPTION_PROTOC = OptionValue("protoc")
if OPTION_PROTOC is None:
    OPTION_PROTOC = find_executable("protoc")
if OPTION_PROTOC is None or not os.path.exists(OPTION_PROTOC):
    raise DistutilsSetupError(
        "Failed to find protoc."
        " Please specify the path to protoc with --protoc parameter.")

class CustomEggInfoCommand(egg_info):
    '''
    customized egg_info command class
    deals with the fact that the build dir containing the compiled protobuf files might not exist initially
    '''
    initInitPaths = initInitPaths
    initInitTemplatedPaths = initInitTemplatedPaths
    package_dir = package_dir

    def __init__(self, dist, **kw):
        [WrappedMakedirs(dirPath) for dirPath in self.package_dir.values()]
        CopyDirStructure(protoSrcDir, protoPythonBuildDir)
        [InitInit(dirPath, recursive=True) for dirPath in self.initInitPaths]
        [InitInit(dirPath, useTemplate=True) for dirPath in self.initInitTemplatedPaths]

        egg_info.__init__(self, dist, **kw)

class CustomSetupCommand:
    '''
    customized setup command base class
    meant to be used in a subclass that also inherits either setuptools.command.install.install or .develop
    '''
    protoPythonBuildDir = protoPythonBuildDir
    protoSrcDir = protoSrcDir

    def run(self):
        self._build_extension()

    def _build_extension(self):
        print('using protoc at %s' % OPTION_PROTOC)

        WrappedMakedirs(self.protoPythonBuildDir)
        protoSrcPaths = RGlob(self.protoSrcDir, '*proto')

        # Compile protobuf files to python
        protoc_python_cmd = [
            OPTION_PROTOC,
            '--proto_path=%s' % self.protoSrcDir,
            '--python_out=%s' % self.protoPythonBuildDir
        ]
        protoc_python_cmd.extend(protoSrcPaths)

        if RunProcess(protoc_python_cmd) != 0:
            raise DistutilsSetupError("Error compiling protobuf files to Python")

class CustomDevelopCommand(CustomSetupCommand, develop):
    def run(self):
        CustomSetupCommand.run(self)
        develop.run(self)

class CustomInstallCommand(CustomSetupCommand, install):
    def run(self):
        CustomSetupCommand.run(self)
        install.run(self)

setup(
    author = 'Elijah Roberts, Max Klein',
    cmdclass = {'develop': CustomDevelopCommand,
                'egg_info': CustomEggInfoCommand,
                'install': CustomInstallCommand},
    description = 'python modules compiled from protobuf sources for projects in the Roberts Lab',
    license = 'UIOSL',
    name = "robertslab-protobuf",
    package_dir = package_dir,
    packages = find_packages(where='./build/python'),
    zip_safe = False,
)