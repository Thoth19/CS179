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
void cudaOneDimWaveKernel(const float *old, const float *curr, float *new_d,
    int n_Nodes, float cour) {

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    // Need to check this here so that we still get multiples of (blockDim*gridDim
         if (thread_index == 0)
{

        thread_index += (blockDim.x * gridDim.x);
}
   while(thread_index < ((unsigned int)n_Nodes -1))
    {
        new_d[thread_index] = 
                (1-cour) * 2*curr[thread_index] - old[thread_index]
                + cour * curr[thread_index+1]
                + cour * curr[thread_index-1];
        
        // Update the thread index.
        thread_index += (blockDim.x * gridDim.x);
    }
}
// Calls the kernel by piping in the blocks and threads/block
// correctly.
void callOneDimWave(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *old,
        const float *curr,
        float *new_d,
        const unsigned int n_Nodes,
        float cour) {
        
    cudaOneDimWaveKernel<<<blocks, threadsPerBlock>>> (old,curr,new_d,n_Nodes, cour);
}
