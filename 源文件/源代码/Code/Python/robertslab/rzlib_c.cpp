#include <stdio.h>
#include <string.h>

#include "rzlib_c.h"

#include "rzlib_c_tinfl.cpp"


static int decompress_nonzero_c_callback(const void* pBuf, int len_c, void *pUser)
{
    unsigned long long int* out_chunk_i = (unsigned long long int*)pBuf;
    int len_i = len_c/sizeof(out_chunk_i[0]);

    void** ptrs = (void**)pUser;
    size_t* total_decompressed_size = (size_t*)ptrs[0];
    int* indices_array = (int*)ptrs[1];
    unsigned char* particles_array = (unsigned char*)ptrs[2];
    size_t* indices_size = (size_t*)ptrs[3];
    size_t* indices_next_position = (size_t*)ptrs[4];

    int i=0;
    while (true)
    {
        // Skip until we have a nonzero value or we have processed the full chunk.
        while (out_chunk_i[i] == 0 && i < len_i) i++;

        // If we are finished, break the loop.
        if (i >= len_i) break;

        // Otherwise, there must have been nonzero data, find the nonzero byte.
        unsigned char* out_chunk_c = (unsigned char*)&(out_chunk_i[i]);
        for (int j=0; j<(int)sizeof(out_chunk_i[0]); j++)
        {
            if (out_chunk_c[j] != 0)
            {
                // Add the position to the indices list.
                if ((*indices_next_position) < (*indices_size))
                {
                    indices_array[(*indices_next_position)] = (int)((*total_decompressed_size)+i*sizeof(sizeof(out_chunk_i[0]))+j);
                    particles_array[(*indices_next_position)] = out_chunk_c[j];
                }
                (*indices_next_position)++;
            }
        }

        // We finished with this i, increment it.
        i++;
    }

    // Update the total decompressed size.
    (*total_decompressed_size) += len_c;

    return 1;
}

int decompress_nonzero_c(unsigned char* in_array, size_t in_size, size_t expected_decompressed_size, int* indices_array, unsigned char* particles_array, size_t indices_size)
{
    size_t total_decompressed_size=0;
    size_t indices_next_position=0;

    void* ptrs[5];
    ptrs[0] = &total_decompressed_size;
    ptrs[1] = indices_array;
    ptrs[2] = particles_array;
    ptrs[3] = &indices_size;
    ptrs[4] = &indices_next_position;
    size_t in_buf_size = in_size;
    int status = tinfl_decompress_mem_to_callback(in_array, &in_buf_size, decompress_nonzero_c_callback, (void*)ptrs, TINFL_FLAG_PARSE_ZLIB_HEADER);
    if (!status)
    {
        printf("tinfl_decompress_mem_to_callback() failed with status %i!\n", status);
        return -1;
    }
    if (total_decompressed_size != expected_decompressed_size) return -4;

    // Make sure we decompressed the expected number of bytes.
    return (int)indices_next_position;
}
//test
#define CHUNK 32*1024

//int decompress_nonzero_c(unsigned char* in_array, size_t in_size, size_t expected_decompressed_size, int* indices_array, unsigned char* particles_array, size_t indices_size)
//{
//    int ret;
//    unsigned have;
//    unsigned i;
//    z_stream strm;
//    unsigned char out_chunk[CHUNK];
//
//    // The current position in the input buffer.
//    size_t in_next_position=0;
//    size_t indices_next_position=0;
//    size_t remaining=in_size;
//    size_t total_decompressed_size=0;
//
//    /* allocate inflate state */
//    strm.zalloc = Z_NULL;
//    strm.zfree = Z_NULL;
//    strm.opaque = Z_NULL;
//    strm.avail_in = 0;
//    strm.next_in = Z_NULL;
//    ret = inflateInit(&strm);
//    if (ret != Z_OK)
//        return -2;
//
//    /* decompress until deflate stream ends */
//    do {
//
//        if (remaining == 0) break;
//
//        // Set the position and size for the next input chunk.
//        strm.avail_in = (remaining>CHUNK)?(CHUNK):(remaining);
//        remaining -= strm.avail_in;
//        strm.next_in = in_array+in_next_position;
//        in_next_position += strm.avail_in;
//
//        /* run inflate() on input until output buffer not full */
//        do {
//
//            strm.avail_out = CHUNK;
//            strm.next_out = out_chunk;
//
//            ret = inflate(&strm, Z_NO_FLUSH);
//            switch (ret) {
//            case Z_NEED_DICT:
//            case Z_STREAM_ERROR:
//            case Z_DATA_ERROR:
//            case Z_MEM_ERROR:
//                (void)inflateEnd(&strm);
//                return -3;
//            }
//
//            // Figure out how many bytes are in the chunk.
//            have = CHUNK - strm.avail_out;
//
//            // Go through the decompressed bytes.
//            /*for (i=0; i<have; i++)
//            {
//                // See if the byte is nonzero.
//                if (out_chunk[i] != 0)
//                {
//                    // Add the position to the indices list.
//                    if (indices_next_position < indices_size)
//                    {
//                        indices_array[indices_next_position] = total_decompressed_size+i;
//                        particles_array[indices_next_position] = out_chunk[i];
//                    }
//                    indices_next_position++;
//                }
//            }*/
//            total_decompressed_size += have;
//
//        } while (strm.avail_out == 0);
//
//        /* done when inflate() says it's done */
//    } while (ret != Z_STREAM_END);
//
//    /* clean up and return */
//    (void)inflateEnd(&strm);
//
//    // Make sure we decompressed the expected number of bytes.
//    if (total_decompressed_size != expected_decompressed_size) return -4;
//
//    return ret == Z_STREAM_END ? indices_next_position : -1;
//}





