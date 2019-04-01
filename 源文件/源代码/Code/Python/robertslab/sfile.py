from six import print_
import numpy as np

class SFile:
    
    @classmethod
    def fromFilename(cls, filename, mode):
        return cls(open(filename, mode), seekable=True)
        
    def __init__(self, fp, printProgress=False, seekable=False):
        self.fp=fp
        self.printProgress = printProgress
        self.seekable=seekable
        self.RECORD_SEPARATOR = np.array([83,70,82,88,1,65,243,72,36,217,55,18,134,11,234,83], dtype=np.uint8)
        self.bytesWritten = 0
        
    def __str__(self):
        return "sfile"
        
    def readNextRecord(self):
        if self.fp != -1:
            
            # Read the record separator.
            separator = np.fromfile(self.fp, dtype=np.uint8, count=self.RECORD_SEPARATOR.size)

            # See if this was EOF.
            if separator.size == 0:
                return None
            
            # See if the the separator was correct.
            if not np.array_equal(separator, self.RECORD_SEPARATOR):
                
                # Otherwise, we need to scan for the separator.
                raise Exception("Error reading next sfile separator.")
            
            # Read the name.
            nameSize = np.fromfile(self.fp, dtype=np.uint32, count=1)
            name = self.fp.read(nameSize)
        
            # Read the type.
            dataTypeSize = np.fromfile(self.fp, dtype=np.uint32, count=1)
            dataType = self.fp.read(dataTypeSize)
            
            # Read the size.
            dataSize = np.fromfile(self.fp, dtype=np.uint64, count=1)
            
            return SFileRecord(name, dataType, dataSize)
            
    def readData(self, dataLength, recordType=None):
        if self.fp != -1:
            if recordType is not None:
                return np.fromfile(self.fp, dtype=recordType, count=dataLength/np.dtype(recordType).itemsize)
            else:
                return self.fp.read(dataLength)

    def skipData(self, length):
        if self.fp != -1:
            if self.seekable:
                self.fp.seek(long(length), 1)
            else:
                bytesToSkip = long(length)
                while bytesToSkip > 0:
                    if bytesToSkip > 4096:
                        self.fp.read(4096)
                        bytesToSkip -= 4096
                    else:
                        self.fp.read(bytesToSkip)
                        bytesToSkip -= bytesToSkip
                        
    def writeRecord(self, name, dataType, npData=None, stringData=None, dataSize=0):
        if self.fp != -1:
            
            # Write the record separator.
            self.RECORD_SEPARATOR.tofile(self.fp)
            
            # Write the name.
            length=np.ndarray((1,), dtype=np.uint32)
            length[0] = np.uint32(len(name))
            length.tofile(self.fp)
            self.fp.write(name)
            
            # Write the data type.
            length[0] = np.uint32(len(dataType))
            length.tofile(self.fp)
            self.fp.write(dataType)
            
            # Write the data size.
            length=np.ndarray((1,), dtype=np.uint64)
            
            if npData != None:
                length[0] = npData.size*npData.dtype.itemsize
                length.tofile(self.fp)  
                npData.tofile(self.fp)
            elif stringData != None:
                length[0] = len(stringData)
                length.tofile(self.fp)
                self.fp.write(stringData)
                bytesWritten = len(stringData)
                self.bytesWritten+=bytesWritten
                if self.printProgress:
                    print_("Wrote %d byte of string data" % (bytesWritten))
            else:
                length[0] = dataSize
                length.tofile(self.fp)  
        
    def close(self):
        if self.fp != -1:
            self.fp.close()
            self.fp=-1
        
        
class SFileRecord:
    def __init__(self, name, dataType, dataSize):
        self.name = name;
        self.dataType = dataType;
        self.dataSize = np.uint64(dataSize);

    def __str__(self):
        return "%s\t%d\t%s"%(self.name,self.dataSize,self.dataType)

