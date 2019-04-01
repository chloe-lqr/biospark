
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "rzlib_c.h":
    int decompress_nonzero_c(unsigned char* in_array, size_t in_size, size_t expected_decompressed_size, int* indices_array, unsigned char* particles_array, size_t indices_size)

# create the wrapper code, with numpy type annotations
def decompress_nonzero(np.ndarray[np.uint8_t, ndim=1, mode="c"] in_array not None, int expected_decompressed_size, np.ndarray[np.int32_t, ndim=1, mode="c"] indices_array not None, np.ndarray[np.uint8_t, ndim=1, mode="c"] particles_array not None):
    return decompress_nonzero_c(<unsigned char*> np.PyArray_DATA(in_array), in_array.shape[0], expected_decompressed_size, <int*> np.PyArray_DATA(indices_array), <unsigned char*> np.PyArray_DATA(particles_array), indices_array.shape[0])