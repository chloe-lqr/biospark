import os,sys
from robertslab.sfile import SFile
from subprocess import PIPE, Popen
import time

from robertslab.md.mdIO.frame import Frame

__all__ = ['SFileMD']

class SFileMD(object):

    pbufRecordPattern = "/Frames/%d"
    pbufDataType = "protobuf:robertslab.pbuf.NDArray"

    @classmethod
    def isFrame(cls, record):
        return record.dataType==cls.pbufDataType

    @property
    def bytesWritten(self):
        return self.sfile.bytesWritten

    def __init__(self, sfilePath, blockSize=None, frames=None, limitToTotalBytes=True, overWrite=True, startTime=None, totalBytes=None, writeTarget=None):

        # init attrs from args
        self.sfilePath = sfilePath
        self.blockSize = blockSize
        self.startTime = startTime
        self.totalBytes = totalBytes
        self.limitToTotalBytes = False if self.totalBytes is None else limitToTotalBytes
        self.writeMode = 'w' if overWrite else 'a'
        self.writeTarget = writeTarget

        # init attr default values
        if frames is not None:
            self.frames = frames
        else:
            self.frames = []
        self.framesWritten = 0
        self.nextFrameNumber = 0
        self.progressStatus = ''
        self.targetStdOut = False

    def initStartTime(self):
        self.startTime = time.time()

    def open(self):
        if self.writeTarget == 'hdfs':
            hdfsPutArgs = ['hdfs', 'dfs', '-put', '-', self.sfilePath]
            if self.writeMode=='w':
                hdfsPutArgs.insert(3, '-f')
                #hdfsRmArgs = ['hdfs', 'dfs', '-rm', self.sfilePath]
            if self.blockSize is not None:
                hdfsPutArgs.insert(2, '-Ddfs.block.size=%d' % self.blockSize)
            p = Popen(hdfsPutArgs, stdin=PIPE)
            self.fh = p.stdin
        elif self.writeTarget == 'cat':
            if self.sfilePath == 'stdout':
                self.targetStdOut = True
                out = sys.stdout
            else:
                out = open(self.sfilePath, self.writeMode)
            print 'cat mode'
            p = Popen(['cat'], stdin=PIPE, stdout=out)
            self.fh = p.stdin
        else:
            if self.sfilePath == 'stdout':
                self.targetStdOut = True
                self.fh = sys.stdout
            else:
                self.fh = open(self.sfilePath, self.writeMode)
        self.sfile = SFile(self.fh, printProgress=False)

        # the first time this function runs .writeMode=='w', =='a' afterwards
        self.writeMode = 'a'

    def close(self):
        if self.writeTarget == 'hdfs':
            self.fh.flush()
        elif self.writeTarget == 'cat':
            self.fh.flush()
        else:
            if self.sfilePath == 'stdout':
                self.fh.flush()
            else:
                self.fh.close()

    # interacting with in-memory data
    def addFrame(self, data):
        self.frames.append(Frame(data))

    # deleting
    def clearFrames(self):
        for i in reversed(range(len(self.frames))):
            del self.frames[i]
        self.frames = []

    # reading/writing to disk
    def readFrames(self):
        """
        populate .frames from a file
        """
        self.sfile = SFile.fromFilename(self.sfilePath, 'r')
        while True:
            r = self.sfile.readNextRecord()
            if r is None:
                break
            elif SFileMD.isFrame(r):
                # if the record represents a frame, deserialize it to a Frame object
                frame = Frame(r.name, r.dataType, self.sfile.readDataRaw(r.dataSize))
                frame.setFrameNumber()
                if self.nextFrameNumber <= frame.frameNumber:
                    self.nextFrameNumber = frame.frameNumber + 1
                self.frames.append(frame)
        self.sfile.close()

    def writeFrames(self, handleFile=True):
        """
        write the contents of .frames to a file
        """
        # if we are only writing one set of frames to the file, it's fine if the write method does the file handling
        if handleFile:
            self.openFileHandle()
        totalBytesReached = False
        for frame in self.frames:
            self.sfile.writeRecord(self.pbufRecordPattern%self.nextFrameNumber, self.pbufDataType, stringData=frame.serialize())
            self.framesWritten+=1
            self.nextFrameNumber+=1
            if self.limitToTotalBytes and not self.totalBytes > self.sfile.bytesWritten:
                totalBytesReached = True
                break
        self.fh.flush()
        # show/update the progress bar
        self.showProgress()
        if handleFile:
            self.closeFileHandle()
        return totalBytesReached

    # progress bar
    def showProgress(self):
        if not self.targetStdOut and self.totalBytes is not None:
            self.progressStatus = self.update_progress(float(self.sfile.bytesWritten)/self.totalBytes, startTime=self.startTime)

    def showProgressDone(self):
        # a cheat, since .totalBytes is usually only a close estimate
        if not self.targetStdOut and self.totalBytes is not None and 'Done' not in self.progressStatus:
            self.update_progress(1, startTime=self.startTime)

    @staticmethod
    def update_progress(progress, startTime=None, timeFmt='%H:%M:%S'):
        # lifted and lightly modified from http://stackoverflow.com/a/15860757/425458
        barLength = 40 # Modify this to change the length of the progress bar
        status = ''
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = 'error: progress var must be float\r\n'
        if progress < 0:
            progress = 0
            status = 'Halt...\r\n'
        if progress >= 1:
            progress = 1
            status = 'Done...\r\n'
        block = int(round(barLength*progress))
        if startTime is not None:
            timeStr = time.strftime(timeFmt, time.gmtime(time.time() - startTime))
            text = '\rPercent: [{0}] {1:>5.2f}%, time elapsed: {2} {3}'.format('#'*block + '-'*(barLength-block), round(progress*100, 2), timeStr, status)
        else:
            text = '\rPercent: [{0}] {1:>5.2f}% {2}'.format('#'*block + '-'*(barLength-block), round(progress*100, 2), status)
        sys.stdout.write(text)
        sys.stdout.flush()
        return status