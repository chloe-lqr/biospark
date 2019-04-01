import json
import subprocess
import sys

__all__ = ['SourceBash']

def SourceBash(fPath, sanitize=None):
    '''
    modified from http://stackoverflow.com/a/7198338/425458
    if sanitize=='envi', use env -i to source fPath in a completely clean environemnt
    '''
    bashCmdToks = ['/usr/bin/env', 'bash', '-c']
    if sanitize=='envi':
        bashCmdToks = ['env', '-i'] + bashCmdToks
    sourceCmd = 'source %s' % fPath
    # sys.executable is the path to the current python interpreter
    dumpCmd = '%s -c "from __future__ import print_function;import os, json;print(json.dumps(dict(os.environ)))"' % sys.executable

    # source a bash script (within a subprocess) and then dump the contents of the python os.environ call into a serialized dict in stdout
    pipe = subprocess.Popen(bashCmdToks + ['%s && %s' % (sourceCmd, dumpCmd)], stdout=subprocess.PIPE)
    # read the serialized environment dict back into the current python script and return it
    return json.loads(pipe.stdout.read().decode())