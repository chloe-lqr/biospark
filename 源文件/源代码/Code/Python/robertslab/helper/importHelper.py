from imp import find_module
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

__all__ = ['GetPathFromModuleName', 'GetSubmoduleNamesFromPackageName', 'NormalizeScriptExt']

def GetPathFromModuleName(name, normalizeScriptExt=True):
    nameParts = name.split('.')
    containingName, name = '.'.join(nameParts[:-1]), nameParts[-1]
    # careful! trying to import a pyspark script directly will likely cause an exception at the 'import pyspark' line, so we instead import only the module containing the script
    containingModule = import_module(containingName)
    fmOut = find_module(name, containingModule.__path__)
    # find_module opens the script as a text file, so we'll close it
    fmOut[0].close()
    return fmOut[1] if not normalizeScriptExt else NormalizeScriptExt(fmOut[1])

def GetSubmoduleNamesFromPackageName(name):
    # iter_modules() returns an iter of tuples in the form (module_loader, name, ispkg)
    return ['.'.join([name, submodTup[1]]) for submodTup in iter_modules(import_module(name).__path__)]

def NormalizeScriptExt(scriptPth, ext='.py'):
    # depending on installation details, .__path__ might point to a .pyc file instead of a .py file, so fix that
    return str(Path(scriptPth).with_suffix(ext))