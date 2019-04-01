#!/usr/bin/env python
#PYTHON_ARGCOMPLETE_OK
from os.path import expanduser
from pathlib import Path
import re

from setupUtils.setupParser import SetupTabCompletionParser
from setupUtils.setupTxt import DetectTxtBlock, InstallTxtBlock, UninstallTxtBlock

scriptsWithTabCompletion = ['sparkSubmit', 'trajectoryToSFile']

# bash code to enable use of /etc/profile.d
profileDEnablement = 'if [ -d /etc/profile.d ]; then\n  for i in /etc/profile.d/*.sh; do\n    if [ -r $i ]; then\n      . $i\n    fi\n  done\n  unset i\nfi\n'

def FindShellStartupPath(systemwide=False):
    # if we want to install system wide, we add a script to /etc/profile.d
    if systemwide:
        return Path('/etc/profile.d/robertslab.sh')

    # for per-user install, prefer .bash_profile to .bashrc
    startupNames = ['.bash_profile', '.bashrc']
    homePath = Path(expanduser('~'))
    for startupPath in (homePath/Path(sName) for sName in startupNames):
        if startupPath.exists():
            return startupPath

    # if we can't find the user's shell startup script, something weird is going on
    raise IOError("can't find user's shell startup script (eg ~/.bash_profile, ~/.bashrc)")

def InitProfileD():
    profilePath = Path('/etc/profile')
    profileDPath = Path('/etc/profile.d')
    # some systems won't have a profile.d dir to start with, so we try to create it
    try:
        profileDPath.mkdir()
        print 'created dir at %s' % profileDPath
    except OSError:
        pass
    # at this point, /etc/profile.d is guaranteed to exist, so we make sure the code to enable its use is present
    with open(str(profilePath), 'r') as f:
        profileTxt = f.read()
    if not re.search(r'/etc/profile\.d', profileTxt, re.DOTALL):
        try:
            print 'adding the following txt to /etc/profile to support the use of /etc/profile.d:'
            print profileDEnablement
            InstallTxtBlock(fPath=str(profilePath), tag='profile.d_enablement', txtBlock=profileDEnablement)
        except IOError as e:
            raise IOError('permissions error. Since you are trying to install tab completions system-wide, run this script with sudo')
    return profileDPath

def Install(parser, ssPath, tag, txtBlock):
    if parser['system-wide']:
        InitProfileD()
    if not ssPath.exists():
        print 'creating file at %s' % ssPath

    if not DetectTxtBlock(fPath=str(ssPath), tag=tag):
        print 'adding the following text to %s:' % ssPath
        print txtBlock

        try:
            InstallTxtBlock(fPath=str(ssPath), tag=tag, txtBlock=txtBlock)
        except IOError as e:
            if parser['system-wide']:
                raise IOError('permissions error. Since you are trying to install tab completions system-wide, run this script with sudo')

def UninitProfileD():
    profilePath = Path('/etc/profile')
    profileDPath = Path('/etc/profile.d')
    tag = 'profile.d_enablement'

    try:
        profileDPath.rmdir()
        print '%s was empty, removed it' % profileDPath
        if DetectTxtBlock(fPath=str(profilePath), tag=tag):
            print 'removing the following text from %s:' % profilePath
            print profileDEnablement
            UninstallTxtBlock(fPath=str(profilePath), tag=tag)
    except OSError:
        if profileDPath.exists():
            print '%s is not empty, leaving it in place' % profileDPath

def Uninstall(parser, ssPath, tag, txtBlock):
    if DetectTxtBlock(fPath=str(ssPath), tag=tag):
        print 'removing the following text from %s:' % ssPath
        print txtBlock

        UninstallTxtBlock(fPath=str(ssPath), tag=tag)

    if parser['system-wide']:
        if ssPath.exists() and ssPath.stat().st_size==0:
            print 'removing empty file at %s' % ssPath
            ssPath.unlink()
        UninitProfileD()

def Main():
    parser = SetupTabCompletionParser()
    parser.run()

    ssPath = FindShellStartupPath(systemwide=parser['system-wide'])
    tag = 'robertslab_tab_completion_activation'
    txtBlock = ''

    for script in scriptsWithTabCompletion:
        txtBlock+=('eval "$(register-python-argcomplete %s)"\n' % script)

    if not parser['uninstall']:
        Install(parser, ssPath, tag, txtBlock)
    else:
        Uninstall(parser, ssPath, tag, txtBlock)

if __name__=='__main__':
    Main()