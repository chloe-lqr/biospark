from robertslab.helper.parseHelper import ArgProperty, ArgGroup, Parser

class SparkParser(Parser):
    def genDefaultArgGroups(self):
        # group of args relating to output files
        outputArgs = ArgGroup('outputArgs',
                              ArgProperty('--outPath', '-o', help='specify path to output'),
                              ArgProperty('--suffix',        help='added to the outPath, before the file ext. Useful for saving the output from multiple replicates/parameter sweeps'))

        return [outputArgs]