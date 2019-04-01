from pathlib import Path

from robertslab.exception.registryException import InsufficientInformationException
from robertslab.helper.importHelper import GetPathFromModuleName, GetSubmoduleNamesFromPackageName

class RegFile(object):
    @property
    def fsPath(self):
        if self._fsPath is not None:
            return self._fsPath
        elif self._pyPath is not None:
            return GetPathFromModuleName(self._pyPath)
        else:
            raise InsufficientInformationException(fsPath=True, pyPath=True)

    @property
    def pyPath(self):
        if self._pyPath is not None:
            return self._pyPath
        else:
            raise InsufficientInformationException(pyPath=True)


    def __init__(self, fsPath=None, hdfsPath=None, pyPath=None):
        # init attrs from args
        self._fsPath = fsPath
        self._hdfsPath = hdfsPath
        self._pyPath = pyPath
