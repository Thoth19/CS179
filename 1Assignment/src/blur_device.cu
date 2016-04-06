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

    // Set up the GPU's version of the i variable.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i;
    while(thread_index < (unsigned int)n_frames)
    {
        i = thread_index;
        // Condense the two for loops into one if block
        // In the worst case, this causes divergence in only one warp
        if (i < (unsigned int) blur_v_size)
	    {
            for (unsigned int j = 0; j <= thread_index; j++)
                out_data[i] += raw_data[i - j] * blur_v[j];
	    }
        else 
	    {	
            for (unsigned int j = 0; j < (unsigned int)blur_v_size; j++)
                out_data[i] += raw_data[i - j] * blur_v[j]; 
	    }
        // Update the thread index.
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
