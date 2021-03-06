/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data, int padded_length) {

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex padded_factor = make_cuFloatComplex(1./padded_length, 0.0);

    /* Point-wise multiplication and scaling for the
    FFT'd input and impulse response.  

    Also remember to scale by the padded length of the signal

    Resilient to varying numbers of threads.
    */

    // We already did the forward DFT on the input and impulse in-place
    // so we don't need to do that here.
    while (thread_index < (unsigned int) padded_length)
    {
    /*    
    out_data[thread_index].x = ((raw_data[thread_index].x * impulse_v[thread_index].x) 
        - (raw_data[thread_index].y * impulse_v[thread_index].y)) 
        / (padded_length);
    out_data[thread_index].y = ((raw_data[thread_index].x * impulse_v[thread_index].y)
        - (raw_data[thread_index].y * impulse_v[thread_index].x)) 
        / (padded_length); 
    */
       out_data[thread_index] = cuCmulf(cuCmulf(raw_data[thread_index], 
        impulse_v[thread_index]), padded_factor);
       thread_index += (blockDim.x * gridDim.x); 
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* Maximum-finding and subsequent
    normalization (dividing by maximum). 

    Uses AtomicMax function above. 

    Explanation of approach to the reduction:

    Sequential Addressing. It is a reduction strategy that fixes the warp
    divergence and bank conflicts from binary tree reduction. It also
    takes advantage of the GPU's multiple SM's. It also reduces the number
    of atomic operations necessary because they must be done serially because
    they access shared memory and are thus blocking.
    */

    /* Note that this kernel doesn't seem to work.
    I'm not entirely sure why, but it seems that either the shared_out_data
    or out_data array indices become too large and cause segfaults. OOM
    errors on Haru made it rather difficult to debug and find which
    one was causing the problem. The main ideas of this code should work however.
    Each thread compares a data value and another one that is sharedsize/shareddiv
    away. We do this for all of the threads.
    Then, we increase shared div. This means we coalesce the data into one block.
    Then we can atomicMax the values from each block to get one real answer
    which can be acquire dby the host.
    Note that we used shared memory so that all of the threads in one block
    can communicate and for faster speed. 
    */

    // Put the data into shared memory.
    unsigned int thread_index_start = threadIdx.x;
    unsigned thread_index = thread_index_start;
    unsigned int shared_size = sizeof(float)*padded_length/(gridDim.x - 1);
    extern __shared__ float shared_out_data[];
    // Which shared memory block am I?
    unsigned int shmem_block = blockIdx.x * shared_size;
    unsigned int shared_div = 2;
    while (thread_index < shared_size)
    {
        shared_out_data[thread_index] = out_data[thread_index + shmem_block].x;
        thread_index += (blockDim.x); 
    }
    __syncthreads();

    // reset thread_index to the beginning
    thread_index = thread_index_start;
    // set shared_div to 2, but making it at the beginning for optimization.
    while(shared_size / shared_div > 0)
    {
        while(thread_index < shared_size / shared_div)
        {
            // don't need to be atomic because no other thread needs to touch it
            shared_out_data[thread_index] = 
                max(fabs(shared_out_data[thread_index]), 
                    fabs(shared_out_data[thread_index + shared_size / shared_div]));
            thread_index += (blockDim.x);
        }
        if (threadIdx.x == 0 && shared_size / shared_div % 2 == 1)
        {
            /* Then we have an extra element that has no mate. */
            shared_out_data[0] = 
                max(fabs(shared_out_data[thread_index]), 
                    fabs(shared_out_data[0]));
        }
        __syncthreads();
        shared_div *= 2;
    }
    // at the end of everything atomic max with maxabsval so that our threads
    // can communicate with one another
    atomicMax(max_abs_val, shared_out_data[0]);
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* Division kernel. Divides all
    data by the value pointed to by max_abs_val. 
    */

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex div_factor = make_cuFloatComplex(1./(* max_abs_val), 0.0);
    while (thread_index < (unsigned int) padded_length)
    {
        out_data[thread_index] = cuCmulf(out_data[thread_index], div_factor);
        thread_index += (blockDim.x * gridDim.x);
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>> (raw_data, impulse_v, 
        out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock, sizeof(float)*padded_length/(blocks - 1)>>> (out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>> (out_data, max_abs_val, 
        padded_length);
}
