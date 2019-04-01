#PYTHON_ARGCOMPLETE_OK
import argparse
from argparse import ArgumentParser
from collections import OrderedDict

try:
    import argcomplete
    from argcomplete.completers import ChoicesCompleter
    doComplete = True
except ImportError:
    doComplete = False

__all__ = ['ArgProperty', 'ArgGroup', 'ConversionFuncs', 'PostProcessFuncs', 'Parser']

class ArgProperty(object):
    '''
    object representing single arg to be passed to ArgumentParser.add_argument()
    '''
    # kwargs (and default values) for extra functionality that ArgProperty understands but ArgumentParser.add_argument() does not
    extraKwargs = [('completeChoices', None), ('propDict', None), ('suppress', False), ('trueDefault', None)]

    # property that generates the *args that get passed to .add_argument
    @property
    def argsForAdd(self):
        return self.flags

    # property that generates the **kwargs that get passed to .add_argument
    @property
    def kwargsForAdd(self):
        return self.kwargs

    def __init__(self, *flags, **kwargs):
        # init attrs from args
        self.flags = flags
        self.name = self.genName()

        # pop any items from kwargs that shouldn't get passed to ArgumentParser.add_argument() and handle them
        self.initAttrsFromKwargs(kwargs)

        # other initialization
        self.initDefault()
        self.initType()

    def initAttrsFromKwargs(self, kwargs):
        # handle the 'trueDefault' kwarg, if present. This can be set if we need to tell the difference between the parser receiving no argVal and the user entering a value==the default value
        for extraKwarg in self.extraKwargs:
            if extraKwarg[0] in kwargs:
                self.__setattr__(extraKwarg[0], kwargs.pop(extraKwarg[0]))
            else:
                self.__setattr__(*extraKwarg)

        self.kwargs = kwargs
        if self.propDict is not None:
            self.kwargs.update(self.propDict)

    def initDefault(self):
        if 'default' not in self.kwargs and self.suppress:
            self.kwargs['default'] = argparse.SUPPRESS

    def initPostProcess(self, tipe):
        # add a post processing func if required, as determined by tipe
        if tipe in PostProcessFuncs.__dict__:
            self.postProcess = PostProcessFuncs.__dict__[tipe].__func__

    def initType(self):
        # if 'type' in kwargs, see if we need to replace a string with a conversion function
        if 'type' in self.kwargs and isinstance(self.kwargs['type'], str) and hasattr(ConversionFuncs, self.kwargs['type']):
            tipe = self.kwargs['type']
            self.kwargs['type'] = ConversionFuncs.__dict__[tipe].__func__
            self.initPostProcess(tipe)

    def genName(self):
        # first, check for a flag without the '-' dinks, as for a positional arg
        for flag in self.flags:
            if not flag.startswith('-'):
                return flag

        # next check for a flag with the double '--', indicative of a long form arg
        for flag in self.flags:
            if flag.startswith('--'):
                return flag.lstrip('-')

        # finally, just return the first flag, sans any dinks
        return self.flags[0].lstrip('-')

class ArgGroup(object):
    '''
    object representing group of args to be passed to ArgumentParser.add_argument
    '''
    # property that generates the *args that get passed to .add_argument
    @property
    def argsForAdd(self):
        return [argProp.argsForAdd for argProp in self.values()]

    # property that generates the **kwargs that get passed to .add_argument
    @property
    def kwargsForAdd(self):
        return [argProp.kwargsForAdd for argProp in self.values()]

    @property
    def argNames(self):
        return [argProp.name for argProp in self.values()]

    # short aliases
    @property
    def argProps(self):
        return self.argProperties

    def __init__(self, name, *argProperties):
        # init attrs from args
        self.name = name
        self.argProperties = OrderedDict()

        self.addArgProperty(*argProperties)

    def __iter__(self):
        return self.argProperties.__iter__()

    # other iterators
    def keys(self):
        return self.argProperties.keys()

    def items(self):
        return self.argProperties.items()

    def values(self):
        return self.argProperties.values()

    def addArgProperty(self, *argProperties):
        for argProp in argProperties:
            self.argProperties[argProp.name] = argProp

    def popArgProperty(self, name):
        return self.argProperties.pop(name)

class Parser(object):
    def __init__(self, argGroups=None, description=''):
        # init attrs from *argProperties
        self.argGroups = OrderedDict()
        self.argProperties = OrderedDict()
        self.description = description

        argGroups = [] if argGroups is None else argGroups
        self.addArgGroup(*(argGroups + self.genDefaultArgGroups()))

        self.initParser()

    def __getitem__(self, key):
        return self.getArgVal(key)

    def initParser(self):
        # turns out this works best if it isn't called until .run() is called
        self.parser = ArgumentParser(self.description)
        for argProp in (argProp for argGroup in self.argGroups.values() for argProp in argGroup.values()):
            if argProp.completeChoices is not None and doComplete:
                self.parser.add_argument(*argProp.argsForAdd, **argProp.kwargsForAdd).completer=ChoicesCompleter(argProp.completeChoices)
            else:
                self.parser.add_argument(*argProp.argsForAdd, **argProp.kwargsForAdd)

    def addArgGroup(self, *argGroups):
        for argGroup in argGroups:
            self.argGroups[argGroup.name] = argGroup
            for argProp in argGroup.values():
                self.argProperties[argProp.name] = argProp

    def addArgProperty(self, argProp, group):
        self.argProperties[argProp.name] = argProp
        if group is not None:
            self.argGroups[group].addArgProperty(argProp)

    def genArgGroupSet(self, group=None):
        if group is not None:
            return {self.argGroups[group]}
        else:
            return set(self.argGroups.values())

    def genDefaultArgGroups(self):
        # hook for subclasses that want to define a default set of argGroups
        return []

    def getArgDict(self, group=None, exclude=None, subs=None):
        argGroupSet = self.genArgGroupSet(group)
        return OrderedDict(((argName,argVal) for argName,argVal in self.getArgTupsFromArgGroupSet(argGroupSet, exclude, subs)))

    def getArgProperty(self, name):
        try:
            return self.argProperties[name]
        except KeyError:
            return self.argProperties['-'.join(name.split('_'))]

    def getArgTupsFromArgGroupSet(self, argGroupSet, exclude=None, subs=None):
        exclude = exclude if exclude is not None else []
        subs = subs if subs is not None else []

        argTups = []
        for argName,argProp in [argPropTup for argGroup in argGroupSet for argPropTup in argGroup.items() if argPropTup[0] not in exclude and not argPropTup[1].suppress]:
            argVal = self.getArgVal(argName)

            # replace values of args as specified by the subs param
            for sub in subs:
                if argName==sub[0] or argName=='-'.join(sub[0].split('_')):
                    argVal = sub[1]

            # workaround to help tell the difference between no user input and user input==default value
            if argVal==self.getDefault(argName) and argProp.trueDefault is not None:
                argVal = argProp.trueDefault

            argTups.append((argName, argVal))
        return argTups

    def getArgVal(self, name):
        try:
            return self.args.__getattribute__(name)
        except AttributeError:
            # fix the whole error arising from the fact that ArgumentParser switches '_' for '-' in argument names
            return self.args.__getattribute__('_'.join(name.split('-')))

    def getCmdLine(self, group=None, exclude=None, subs=None):
        argGroupSet = self.genArgGroupSet(group)
        cmdLineTups = [('--' + argName, str(argVal)) for argName,argVal in self.getArgTupsFromArgGroupSet(argGroupSet, exclude, subs)]
        return [argToken for cmdLineTup in cmdLineTups for argToken in cmdLineTup]

    def getDefault(self, name):
        return self.getArgProperty(name).kwargs.get('default', None)

    def popArgProperty(self, name):
        for argGroup in self.argGroups.values():
            try:
                argGroup.popArgProperty(name)
            except KeyError:
                pass
        return self.argProperties.pop(name)

    def postProcess(self):
        for argName,argProp in (argPropTup for argPropTup in self.argProperties.items() if hasattr(argPropTup[0], 'postProcess')):
            self.setArgVal(argName, argProp.postProcess(self.getArgVal(argName)))

    def parseArguments(self, cmdLine=None):

        # if you want to parse a list of args aside from sys.argv, specify it in cmdLine
        self.initParser()

        if doComplete:
            argcomplete.autocomplete(self.parser)

        if cmdLine is not None:
            self.args = self.parser.parse_args(cmdLine.split())
        else:
            self.args = self.parser.parse_args()

        self.postProcess()

        return self.args

    def setArgVal(self, name, val):
        if hasattr(self, name):
            self.args.__setattr__(name, val)
        elif hasattr(self, '_'.join(name.split('-'))):
            # fix the whole error arising from the fact that ArgumentParser switches '_' for '-' in argument names
            self.args.__setattr__(self, '_'.join(name.split('-')), val)

class ConversionFuncs(object):
    '''
    type conversion functions for the 'type' kwarg
    '''
    @staticmethod
    def delimiterSubClosure(findDelim, replDelim):
        '''
        conversion for option values that need a single character substitution
        for example 'protein_and_not_name_H' -> 'protein and not name H'
        '''
        def delimiterSub(s):
            return replDelim.join(s.split(findDelim))
        return delimiterSub

    @staticmethod
    def intRange(s):
        '''
        conversion for option values containing ranges of integers specified as a-b,c-d,...
        for example 2,3-5,19,27,41-43 -> [2,3,4,5,19,27,41,42,43]
        '''
        vals = []
        for subS in s.split(','):
            if '-' in subS:
                # the map serves to convert everything in the list returned by .split to integers
                vals+=ConversionFuncs.rangeToEnd(*map(int, subS.split('-')))
            else:
                vals.append(int(subS))
        return vals

    @staticmethod
    def rangeToEnd(start, stop):
        '''
        version of built-in range() that includes the stop val in the return
        '''
        return range(start, stop+1)

    @staticmethod
    def strList(s):
        '''
        conversion for option values containing multiple strings specified as str0,str1...
        for example foo,bar -> ['foo', 'bar']
        '''
        return s.split(',')

    @staticmethod
    def varEqualsVal(s):
        '''
        conversion for option values containing variable names and values specified as name0=val0,name1=val1...
        for example apple=red,foo=bar -> {'apple': 'red', 'foo': 'bar'}
        '''
        varDict = {}
        for subS in s.split(','):
            if '=' in subS:
                key,val = subS.split('=')
                varDict[key] = val
            else:
                raise ValueError('malformed variable specification in option value. should be in the form name0=val0,name1=val1...\n option value: %s' % s)
        return varDict

class PostProcessFuncs(object):
    '''
    functions that can be called on argVals after initial parsing has completed. Usually for assisting with combinations of nargs='+' and a ConversionFunc
    '''
    @staticmethod
    def intRange(iR):
        '''
        if type=intRange and nargs='*' are used together, we get back a mixed list of ints and list-of-ints, so this'll flatten it
        '''
        flatIR = []
        if iR is not None:
            # it might be a single int, or it might be a list, so we'll just call it a bunch
            for intBunch in iR:
                try:
                    for i in intBunch:
                        flatIR.append(i)
                except TypeError:
                    flatIR.append(intBunch)