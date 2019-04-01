from robertslab.helper.importHelper import GetPathFromModuleName, GetSubmoduleNamesFromPackageName
from robertslab.helper.singletonHelper import SingletonMixin

__all__ = ['ScriptRegistry']

class ScriptRegistry(SingletonMixin):
    # scripts placed in robertslab.spark.scripts are automatically registered
    # other scripts should be added to the dict bellow
    sparkScriptDict = {'calc_rmsd': 'robertslab.md.rmsd.spark.calc_rmsd',
                       'calc_rmsdNxN': 'robertslab.md.rmsd.spark.calc_rmsdNxN'}
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