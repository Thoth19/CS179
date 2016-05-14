/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* Preforms the same computation that the CPU implementation
did. It stores the answers in the newDisplacement. */

 __global__
void cudaOneDimWaveKernel(const float *old, const float *curr, float *new,
    int n_Nodes) {

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(0 < thread_index && thread_index <= ((unsigned int)n_Nodes -2))
    {
        new[thread_index] = 
                2*curr[thread_index] - old[thread_index]
                + dev_courantSquared * (curr[thread_index+1]
                        - 2*curr[thread_index] 
                        + curr[thread_index-1]);
        
        // Update the thread index.
        thread_index += (blockDim.x * gridDim.x);
    }

// Calls the kernel by piping in the blocks and threads/block
// correctly.
void callOneDimWave(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *old
        const float *curr,
        float *new,
        const unsigned int n_Nodes) {
        
    cudaOneDimWaveKernel<<<blocks, threadsPerBlock>>> (old,curr,new,n_Nodes);
}
