#ifndef HIGH_PASS_CUH
#define HIGH_PASS_CUH

#include <cufft.h>

void cudaCallComplexArrayToFloat(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        float *out_data,
        const unsigned int length);
void cudaCallHighPass(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *out_data,
        const unsigned int length);
#endif
