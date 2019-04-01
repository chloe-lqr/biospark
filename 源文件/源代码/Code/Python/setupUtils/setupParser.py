#PYTHON_ARGCOMPLETE_OK
from argparse import ArgumentParser
import os,sys

try:
    import argcomplete
    doComplete = True
except ImportError:
    doComplete = False

from . import setupUtils; doDict = setupUtils.doDict

__all__=['SetupParser', 'SetupTabCompletionParser']

class BaseParser(object):
    def __init__(self, description=''):
        self.parser = ArgumentParser(description)
        if doComplete:
            argcomplete.autocomplete(self.parser)

    def __getitem__(self, key):
        return self.getArgVal(key)

    def run(self):
        # have to use .parse_known_args() since pip throws around a lot of arguments
        self.args,self.unconsumed = self.parser.parse_known_args()
        # some commands execed by pip complain if they get extra args, so remove them
        sys.argv[1:] = self.unconsumed

    def getArgVal(self, name):
        try:
            return self.args.__getattribute__(name)
        except AttributeError:
            # fix the whole error arising from the fact that ArgumentParser switches '_' for '-' in argument names
            return self.args.__getattribute__('_'.join(name.split('-')))

class SetupParser(BaseParser):
    def __init__(self, description=''):
        super(self.__class__, self).__init__(description=description)

        deploymentDefault = os.environ['ROBERTSLAB_DEPLOYMENT'] if 'ROBERTSLAB_DEPLOYMENT' in os.environ else 'default'
        deploymentFlags = ['--deployment', '-d']
        self.parser.add_argument(*deploymentFlags, choices=doDict.keys(), default=deploymentDefault, help='flag to select one of a set of predefined build options for any cython extensions')
        # self.parser.add_argument('--tab-completion', '-t', action='store_true',                          help='if you have the python package argcomplete installed, you can set this flag to enable tab completion for scripts in the robertslab package. requires sudo access, since this needs to add some lines to the system-wide .bashrc equivalent')

class SetupTabCompletionParser(BaseParser):
    def __init__(self, description=''):
        super(self.__class__, self).__init__(description=description)

        self.parser.add_argument('--uninstall', '-u', action='store_true',   help='pass this flag to completely reverse any changes this script has made to your system')
        self.parser.add_argument('--system-wide', '-s', action='store_true', help='pass this flag to install tab completions system-wide. By default they are installed on a per-user basis')

    def __getitem__(self, key):
        return self.getArgVal(key)

    def run(self):
        self.args = self.parser.parse_args()

    def getArgVal(self, name):
        try:
            return self.args.__getattribute__(name)
        except AttributeError:
            # fix the whole error arising from the fact that ArgumentParser switches '_' for '-' in argument names
            return self.args.__getattribute__('_'.join(name.split('-')))