from __future__ import absolute_import
from __future__ import division

import numpy as np
import re
import StringIO
import sys
import zipfile
import zlib

# Try to load snappy.
try:
    import snappy
    snappyPresent = True
except:
    snappyPresent = False
    pass

# Try to load the Protocol Buffers NDArray schema.
try:
    from robertslab.pbuf.NDArray_pb2 import NDArray as NDArray
    pbufPresent = True
except:
    pbufPresent = False
    pass

# Try to load the Avro NDArrary schema.
try:
    import avro.schema
    import avro.io
    import robertslab.avro
    regex = re.compile("^(.+robertslab-avro-python.zip)/robertslab/avro/__init__.py$")
    m = regex.search(robertslab.avro.__file__)
    if m != None:
        avroZipFile = m.group(1)
        fp = zipfile.ZipFile(avroZipFile, "r")
        avroNDArraySchema = avro.schema.parse(fp.open("robertslab/avro/NDArray.avsc","r").read())
        fp.close()
        avroPresent = True
    else:
        avroPresent = False
except Exception as e:
    avroPresent = False
    pass

# Define the numpy to serialization type maps.
msgTypes = [(NDArray.int8,np.int8),(NDArray.int16,np.int16),(NDArray.int32,np.int32),(NDArray.int64,np.int64),(NDArray.uint8,np.uint8),(NDArray.uint16,np.uint16),(NDArray.uint32,np.uint32),(NDArray.uint64,np.uint64),(NDArray.float16,np.float16),(NDArray.float32,np.float32),(NDArray.float64,np.float64),(NDArray.complex64,np.complex64),(NDArray.complex128,np.complex128)]
msgToNpTypesMap = dict(msgTypes)
npToMsgTypesMap = dict(map(lambda x:x[::-1],msgTypes))
npToMsgTypesMap.update(dict([(np.dtype("uint8"),NDArray.uint8),(np.dtype("uint32"),NDArray.uint32),(np.dtype("float32"),NDArray.float32),(np.dtype("float64"),NDArray.float64)]))
avroTypes = [("int8",np.int8),("int16",np.int16),("int32",np.int32),("int64",np.int64),("uint8",np.uint8),("uint16",np.uint16),("uint32",np.uint32),("uint64",np.uint64),("float16",np.float16),("float32",np.float32),("float64",np.float64),("complex64",np.complex64),("complex128",np.complex128)]
avroToNpTypesMap = dict(avroTypes)
npToAvroTypesMap = dict(map(lambda x:x[::-1],avroTypes))
npToAvroTypesMap.update(dict([(np.dtype("uint8"),"uint8"),(np.dtype("uint32"),"uint32"),(np.dtype("float32"),"float32"),(np.dtype("float64"),"float64")]))

def deserialize(data, format="pbuf"):

    if format == "pbuf":
        msg = NDArray()
        msg.ParseFromString(str(data))
        return deserializePbufMessage(msg)
    elif format == "avro":
        return deserializeFromAvro(data)
    else:
        raise Exception("Unknown serialization format: %s"%(format))

def deserializePbufMessage(msg):

    # Convert the data to a numpy array.
    if msg.compressed_deflate:
        return np.reshape(np.fromstring(zlib.decompress(msg.data), dtype=msgToNpTypesMap[msg.data_type]), msg.shape)
    elif msg.compressed_snappy:
        return np.reshape(np.fromstring(snappy.uncompress(msg.data), dtype=msgToNpTypesMap[msg.data_type]), msg.shape)
    else:
        return np.reshape(np.fromstring(msg.data, dtype=msgToNpTypesMap[msg.data_type]), msg.shape)
        
def deserializeFromAvro(data):

    reader = avro.io.DatumReader(avroNDArraySchema)
    bytes_reader = StringIO.StringIO(data)
    decoder = avro.io.BinaryDecoder(bytes_reader)
    avroRecord = reader.read(decoder)
    if avroRecord["compressed_deflate"]:
        return np.reshape(np.fromstring(zlib.decompress(avroRecord["data"]), dtype=avroToNpTypesMap[avroRecord["data_type"]]), avroRecord["shape"])
    elif avroRecord["compressed_snappy"]:
        return np.reshape(np.fromstring(snappy.uncompress(avroRecord["data"]), dtype=avroToNpTypesMap[avroRecord["data_type"]]), avroRecord["shape"])
    else:
        return np.reshape(np.fromstring(avroRecord["data"], dtype=avroToNpTypesMap[avroRecord["data_type"]]), avroRecord["shape"])

def serialize(array, format="pbuf", compressed=True, compressionLevel=1):

    # Figure out the compression type.
    compression = None
    if type(compressed) == type(True):
        if compressed is True:
            compression = "deflate"
    elif type(compressed) == type(""):
        if compressed == "deflate":
            compression = "deflate"
        elif compressed == "snappy":
            compression = "snappy"
        else:
            raise "Unknown compression type:",compressed
    else:
        raise "Unknown compression type:",compressed
    
    # Call the appropriate serialization method for the format.
    if format == "pbuf":
        msg = NDArray()
        serializeIntoPbufMessage(msg, array, compression, compressionLevel)
        return bytearray(msg.SerializeToString())
    elif format == "avro":
        return serializeAsAvro(array, format, compression, compressionLevel)
    else:
        raise Exception("Unknown serialization format: %s"%(format))

def serializeIntoPbufMessage(msg, array, compression="deflate", compressionLevel=1):

    if array.dtype not in npToMsgTypesMap:
        print npToMsgTypesMap
        print array.dtype
        raise TypeError("Could not translate np type to NDArray type: ",array.dtype)
    msg.Clear()
    
    msg.data_type = npToMsgTypesMap[array.dtype]
    msg.shape.extend(array.shape)
    
    if compression == "deflate":
        msg.data = zlib.compress(array.tostring(),compressionLevel)
        msg.compressed_deflate = True
    elif compression == "snappy":
        msg.data = snappy.compress(array.tostring())
        msg.compressed_snappy = True
    else:
        msg.data = array.tostring()

def serializeAsAvro(array, format="pbuf", compression="deflate", compressionLevel=1):
        
    avroRecord = {"array_order": "ROW_MAJOR", "byte_order": "LITTLE_ENDIAN", "compressed_deflate": False, "compressed_snappy": False}
    if array.dtype not in npToAvroTypesMap:
        raise TypeError("Could not translate np type to NDArray type: ",array.dtype)
    avroRecord["data_type"] = npToAvroTypesMap[array.dtype]
    avroRecord["shape"] = list(array.shape)
        
    # Fill in the record data.
    if compression == "deflate":
        avroRecord["data"] =  zlib.compress(array.tostring(),compressionLevel)
        avroRecord["compressed_deflate"] = True
    elif compression == "snappy":
        avroRecord["data"] = snappy.compress(array.tostring())
        avroRecord["compressed_snappy"] = True
    else:
        avroRecord["data"] = array.tostring()
        
    # Serialize the record.
    writer = avro.io.DatumWriter(avroNDArraySchema)
    bytes_writer = StringIO.StringIO()
    encoder = avro.io.BinaryEncoder(bytes_writer)
    writer.write(avroRecord, encoder)
    return bytes_writer.getvalue()
    
