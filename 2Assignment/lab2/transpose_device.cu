#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        /*
        There are memory coalescing issues with respect to writing the
        output. n is a multiple of 64. We are using one cache line per
        float on the output. This is very inefficient.
        */
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.


    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    int xOffset = blockIdx.x * 64;
    int yOffset = blockIdx.y * 64;

    __shared__ float data[64*65]; // We was an extra bank for padding

    for (; j < end_j; j++)
        data[(j-yOffset) + 64 * (i - xOffset)+threadIdx.x] = input[i + n * j];
    __syncthreads();

    int k = 4*threadIdx.y;
    const int end_k = k + 4;
    int xIdx, yIdx;
    for (; k < end_k; k ++)
    {
        xIdx = threadIdx.x + (64*blockIdx.y);
        yIdx = (k+blockIdx.x * 64);
        output[n*(yIdx)+(xIdx)] = data[threadIdx.x+65*k];
    }
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.


    // Moved the variable declarations to the top of the function
    // in order to reduce dependency isues.
    int k = 4*threadIdx.y;
    // We don't need these two, but they make the next definition
    // make sense.
    // int nyIdx = n*(k+blockIdx.x * 64);
    // int xIdx = threadIdx.x + (64*blockIdx.y);
    int nyIdxPlusxIdx = n*(k+blockIdx.x * 64) + threadIdx.x + (64*blockIdx.y);
    int data_idx = threadIdx.x+65*k;
    
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;

    int xOffset = blockIdx.x * 64;
    int yOffset = blockIdx.y * 64;

    int data_field = (j-yOffset) + 64 * (i - xOffset)+threadIdx.x;
    int input_field = i + n * (j);
    __shared__ float data[64*65]; // We was an extra bank for padding

    // Unrolled the loop. Used precomputed values.
    data[data_field] = input[input_field];
    data[data_field+1] = input[input_field + n];
    data[data_field+2] = input[input_field + 2*n];
    data[data_field+3] = input[input_field + 3*n];
    __syncthreads();


    // Unrolled loop. Only need to compute xIdx once.
    // Using precomputed values that are at top of function

    output[nyIdxPlusxIdx]       = data[data_idx];
    output[nyIdxPlusxIdx +   n] = data[data_idx];
    output[nyIdxPlusxIdx + 2*n] = data[data_idx];
    output[nyIdxPlusxIdx + 3*n] = data[data_idx];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
