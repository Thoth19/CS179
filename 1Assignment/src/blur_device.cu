/* 
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 */

#include <cstdio>

#include <cuda_runtime.h>

#include "blur_device.cuh"


__global__
void cudaBlurKernel(const float *raw_data, const float *blur_v, float *out_data,
    int n_frames, int blur_v_size) {

    // TODO: Fill in the implementation for the GPU-accelerated convolution. 
    //
    // It may be helpful to use the information in the lecture slides, as well
    // as the CPU implementation, as a reference.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_index < n_frames)
    {
        unsigned int i = thread_index;
        for (int j = 0; j <= min((unsigned int) blur_v_size, i); j++)
        {
            out_data[i] += raw_data[i - j] * blur_v[j]; 
        }
        thread_index += (blockDim.x * gridDim.x);
    }
}


void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int n_frames,
        const unsigned int blur_v_size) {
        
    cudaBlurKernel<<<blocks, threadsPerBlock>>> (raw_data, blur_v, out_data,
        n_frames, blur_v_size);
}
