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
       
    for (unsigned int a =  1 + blockIdx.x * blockDim.x + threadIdx.x; a <= n_Nodes - 2; a += threadIdx.x*gridDim.x){
        new_d[a] = 
                2*curr[a] - old[a]
                + cour * (curr[a+1]
                        - 2*curr[a] 
                        + curr[a-1]);
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
