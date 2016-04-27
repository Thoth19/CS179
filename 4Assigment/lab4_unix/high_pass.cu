#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "high_pass.cuh"

__global__
void
cudaComplexArrayToFloat(const cufftComplex *raw_data, 
    float *out_data, int length) 
/* Converts an array of cufftComplex's to the real parts */
{

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (thread_index < (unsigned int) length)
    {
       out_data[thread_index] = raw_data[thread_index].x;
       thread_index += (blockDim.x * gridDim.x); 
    }
}
__global__
void
cudaHighPassOdd(const cufftComplex *raw_data,
    cufftComplex *out_data, int length)
/* Implements a High Pass filter on an odd length array
Since we are in the Frequency domain, we scale based on the position in the
array.*/
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int len_helper = ((length - 1) /2);
    int dist_from_center;
    int scale_factor;
    // Note: This isn't fully optimized because it would make the code
    // unclean and the assignment doesn't ask for it.
    // We could compute the scale_factor and apply it in one step.
    while (thread_index < (unsigned int) length)
    {
        // Compute distance from len / 2. 
        if (thread_index > (unsigned int)len_helper)
        {
            dist_from_center = thread_index - len_helper; // positive
            scale_factor = dist_from_center / len_helper; // positive
        }
        else if (thread_index == (unsigned int) len_helper)
        {
            scale_factor = 0;
        }
        else
        {
            dist_from_center = len_helper - thread_index; // positive
            scale_factor = dist_from_center / len_helper; // positive
        }

        out_data[thread_index].x = (raw_data[thread_index].x) * scale_factor;
        out_data[thread_index].y = (raw_data[thread_index].y) * scale_factor;

       thread_index += (blockDim.x * gridDim.x); 
    }
}

__global__
void
cudaHighPassEven(const cufftComplex *raw_data,
    cufftComplex *out_data, int length)
/* Implements a High Pass filter on an even length array
Since we are in the Frequency domain, we scale based on the position in the
array.
Even arrays don't actually have a true center, so two elements will be scaled to 0*/
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int len_helper = ((length /2) - 1); // Element to left of middle
    int dist_from_center;
    int scale_factor;
    // Note: This isn't fully optimized because it would make the code
    // unclean and the assignment doesn't ask for it.
    // We could compute the scale_factor and apply it in one step.
    while (thread_index < (unsigned int) length)
    {
        // Compute distance from len / 2. 
        if (thread_index <= (unsigned int)len_helper)
        {
            dist_from_center = thread_index - len_helper; // >= 0
            scale_factor = dist_from_center / len_helper; // positive
        }
        else
        {
            dist_from_center = (len_helper + 1) - thread_index; // >= 0
            scale_factor = dist_from_center / len_helper; // positive
            // We still use len_helper on this part because we are using
            // it to scale based on being half of the length of the list,
            // not its position in the array
        }

        out_data[thread_index].x = (raw_data[thread_index].x) * scale_factor;
        out_data[thread_index].y = (raw_data[thread_index].y) * scale_factor;

       thread_index += (blockDim.x * gridDim.x); 
    }
}


void cudaCallComplexArrayToFloat(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        float *out_data,
        const unsigned int length) {
        
    /* Call the element-wise conversion to float kernel. */
    cudaComplexArrayToFloat<<<blocks, threadsPerBlock>>> (raw_data, 
        out_data, length);
}
void cudaCallHighPass(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *out_data,
        const unsigned int length)
{
    // We have two different kernels to reduce the number of if statements
    // coming from even vs. odd lengths. The difference comes from finding
    // the center element.
    if (length % 2 == 0)
    {
        cudaHighPassEven<<<blocks, threadsPerBlock>>> (raw_data,
            out_data, length);
    }
    else
    {
        cudaHighPassOdd<<<blocks, threadsPerBlock>>> (raw_data,
            out_data, length);
    }
}