from argparse import ArgumentParser as AP
from robertslab.exception.robertslabException import RobertslabException

__all__ = ['RobertslabException', 'InsufficientInformationException']

class RegistryException(RobertslabException):
    '''
    base class for Exceptions relating to the registry subpackage
    '''
    pass

class InsufficientInformationException(RegistryException):
    '''
    Exception raised when when a reg object doesn't have enough information available to perform a given operation
    '''
    parser = AP()

    def __init__(self, pyPath=False, fsPath=False, hdfsPath=False, *args):
        py = 'pyPath,'*pyPath
        fs = 'fsPath,'*fsPath
        hdfs = 'hdfsPath,'*hdfsPath

        message = ('This function needs at least one of the following pieces of information: %s%s%s' % (py, fs, hdfs)).rstrip(',')
        super(self.__class__, self).__init__(message, *args)