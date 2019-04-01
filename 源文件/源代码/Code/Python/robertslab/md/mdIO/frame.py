import numpy as np

import robertslab.ndarray as ndarray

__all__ = ['Frame']

class Frame(object):

    def __init__(self, data, frameNumber=None):

        self.frameNumber = frameNumber

        if isinstance(data, str) or isinstance(data, bytearray):
            # extract the xyz data from a byte string
            self.xyz = ndarray.deserialize(data)
        else:
            # assume that data is already a numpy array, or that the user knows what they're doing
            self.xyz = data

        # Make sure the data is consistent.
        if len(self.xyz.shape) != 2 or self.xyz.shape[1] != 3:
            raise ValueError("Invalid array shape: ",self.xyz.shape)
        if self.xyz.dtype != np.float32:
            raise ValueError("Invalid array data type: ",self.xyz.dtype)

    def serialize(self):
        return ndarray.serialize(self.xyz, compressed=False)

