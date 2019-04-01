# fancy importing
# lovely and automatic, but screws up IDE parsing
#
# from . import frame, sfileMD, trajectoryToSFile
# mods = [frame, sfileMD, trajectoryToSFile]
# __all__, allDict = [], {}
#
# for mod in mods:
#     for var in mod.__all__:
#         allDict[var] = mod.__getattribute__(var)
#         __all__.append(var)
#
# locals().update(allDict)

# boilerplate heavy importing
from robertslab.md.mdIO.frame import Frame
from robertslab.md.mdIO.sfileMD import SFileMD
