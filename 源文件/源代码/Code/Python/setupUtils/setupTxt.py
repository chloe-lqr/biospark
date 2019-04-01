import re

__all__=['DetectTxtBlock', 'InstallTxtBlock', 'UninstallTxtBlock']

def AppendBlockToFile(fPath, startLine, endLine, txtBlock):
    oldTxt = ''
    newTxt = EnsureNewline(startLine) + EnsureNewline(txtBlock) + endLine
    try:
        with open(fPath, 'r') as f:
            oldTxt = f.read()
            oldTxt = EnsureNewline(oldTxt) + '\n'
    except IOError:
        pass
    with open(fPath, 'w') as f:
        f.write(oldTxt)
        f.write(newTxt)


def DetectTxtBlock(fPath, tag):
    blockStartBumper = GenBumper(['start', tag])
    blockStopBumper = GenBumper(['stop', tag])
    return DetectLines(fPath, blockStartBumper, blockStopBumper)

def DetectLines(fPath, startLine, endLine):
    detRe = re.compile(startLine + '.*' + endLine, re.DOTALL)
    try:
        with open(fPath, 'r') as f:
            oldTxt = f.read()
    except IOError:
        return False
    if detRe.search(oldTxt):
        return True
    else:
        return False

def EnsureNewline(line):
    line = line.rstrip()
    line = line + '\n'
    return line

def GenBumper(tags, spacer='#'*4):
    tag = '_'.join(tags).upper()
    return '%s%s%s' % (spacer, tag, spacer)

def InstallTxtBlock(fPath, tag, txtBlock):
    blockStartBumper = GenBumper(['start', tag])
    blockStopBumper = GenBumper(['stop', tag])
    if not ReplaceBtwnLines(fPath, blockStartBumper, blockStopBumper, txtBlock):
        AppendBlockToFile(fPath, blockStartBumper, blockStopBumper, txtBlock)

def ReplaceBtwnLines(fPath, startLine, endLine, txtBlock, keepBumpers=True):
    repRe = re.compile(startLine + '.*' + endLine, re.DOTALL)
    if keepBumpers:
        replTxt = EnsureNewline(startLine) + EnsureNewline(txtBlock) + endLine
    else:
        replTxt = txtBlock

    try:
        with open(fPath, 'r') as f:
            oldTxt = f.read()
    except IOError:
        return False
    if repRe.search(oldTxt):
        newTxt = repRe.sub(replTxt, oldTxt)
        with open(fPath, 'w') as f:
            f.write(newTxt)
        return True
    else:
        return False

def UninstallTxtBlock(fPath, tag):
    blockStartBumper = GenBumper(['start', tag])
    blockStopBumper = GenBumper(['stop', tag])
    ReplaceBtwnLines(fPath, blockStartBumper, blockStopBumper, '', keepBumpers=False)