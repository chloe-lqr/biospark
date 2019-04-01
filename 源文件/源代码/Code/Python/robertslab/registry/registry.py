from robertslab.helper.importHelper import GetPathFromModuleName, GetSubmoduleNamesFromPackageName
from robertslab.helper.singletonHelper import SingletonMixin

__all__ = ['Registry']

class RegistryMetaclass(type):
    # this metaclass sets some default values for some mutable class attrs (mcas), to avoid the classic problems with mcas
    def __new__(cls, clsname, bases, dct):
        # dict of registered python modules with {module name:module fullname}
        # eg moduleDict = {'calc_rmsd': 'robertslab.md.src.rmsd.spark.calc_rmsd}
        dct['moduleDict'] = {}

        # list of absolute paths to registered files
        # eg fileList = ['/Users/tel/git/roberts-lab/Code/Hadoop/lib/robertlab-hadoop.jar']
        dct['fileList'] = []

        #

        return super(RegistryMetaclass, cls).__new__(cls, clsname, bases, dct)

class Registry(SingletonMixin):
    # scripts placed in robertslab.spark.scripts are automatically registered
    # other scripts should be added to the dict bellow
    sparkScriptDict = {'calc_rmsd': 'robertslab.md.src.rmsd.spark.calc_rmsd',
                       'calc_rmsdNxN': 'robertslab.md.src.rmsd.spark.calc_rmsdNxN'}
    scriptPkg = 'robertslab.spark.scripts'

    def __init__(self):
        for scriptModuleName in GetSubmoduleNamesFromPackageName(self.scriptPkg):
            self.sparkScriptDict[scriptModuleName.split('.')[-1]] = scriptModuleName

    def getStems(self):
        return list(self.sparkScriptDict.keys())

    def getPathFromSparkScriptName(self, name):
        if name in self.sparkScriptDict:
            return GetPathFromModuleName(self.sparkScriptDict[name])
        else:
            return None