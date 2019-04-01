import numpy as np
import os
import re
import scipy.io
import h5py

def exists(filename, indices=None, format="matlab"):

    # If we got indices, construct the filename
    if indices is not None:

        # Use the filename as a directory.
        dirname = filename

        # Make the directory.
        if not os.path.exists(dirname): os.makedirs(dirname)

        # Generate the filename.

        filename="%s/"%dirname
        for index in indices:
            filename+="%05d."%(index+1)

        if format == "matlab":
            filename+="mat"
        elif format == "hdf5":
            filename+="h5"

    return os.path.exists(filename)


def cellsave(filename, celldata, indices=None, format="matlab"):

    # If we got indices, construct the filename
    if indices is not None:

        # Use the filename as a directory.
        dirname = filename

        # Make the directory.
        if not os.path.exists(dirname): os.makedirs(dirname)

        # Generate the filename.

        filename="%s/"%dirname
        for index in indices:
            filename+="%05d."%(index+1)

        if format == "matlab":
            filename+="mat"
        elif format == "hdf5":
            filename+="h5"

    # Save the cell.
    if format == "matlab":
        scipy.io.savemat(filename, {"celldata":celldata}, oned_as='column')
    elif format == "hdf5":
        if isinstance(celldata, dict):
            fp = h5py.File(filename, "a")
            for key in celldata.keys():
                fp.create_dataset(key, data=celldata[key], compression="gzip")
            fp.close()
        else:
            raise Exception("Unknown type for cellio hdf5 file format: %s"%type(celldata))
    else:
        raise Exception("Unknown cellio file format: %s"%format)


def cellload(filename, specificIndices=None):

    celldata = {}

    # If this is a directory, load all of the files.
    if os.path.isdir(filename):
        dirname = filename
        for filename in os.listdir(dirname):
            if re.match("^[0-9\.]+\.h5$",filename) is not None:
                indices = ()
                for indexString in re.findall("(\d+)\.",filename):
                    indices += (int(indexString)-1,)
                if len(indices) > 0 and (specificIndices is None or indices == specificIndices):
                    fp = h5py.File(dirname+"/"+filename, "r")
                    datasetNames=[]
                    fp.visititems(lambda x,y: cellload_h5py_callback(x,y,datasetNames))
                    data={}
                    for name in datasetNames:
                        data[name] = np.array(fp[name])
                    fp.close()
                    celldata[indices] = data

    return celldata

def cellload_h5py_callback(name, object, datasetNames):
    if isinstance(object, h5py.Dataset):
        datasetNames.append(str("/"+name))