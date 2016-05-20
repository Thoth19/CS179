/* 
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 */

#include <cstdio>

#include <cuda_runtime.h>

#include "partTwo.cuh"


__global__
void cudaGillKernel(const float b,
        const float g,
        const float kon,
        const float koff,
        const float* random_distrib,
        const int len_rand,
        int* state,
        float* concentration
        float* times) {

    // Set up the GPU's version of the i variable.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float t;
    while(thread_index< (unsigned int)len_rand)
    {
        // We are going to use thread_index for getting the timestep
        // and thread_index + 1 for the probability of changing state.
        times[random_distrib[thread_index]] = -1 * log(random_distrib[thread_index]) / (b + g);
        float temp_rand = random_distrib[(thread_index + 1) % len_rand];
        if (state)
        {
            // ON
            if (temp_rand < (koff / (koff + b +concentration[thread_index] * g)))
            {
                /* Change to off state */
                state[thread_index] = 0;
            }
            else if (temp_rand < ((koff + b) / (koff + b +concentration[thread_index] * g)))
            {
                /* If we weren't less than koff, but are less than koff + b
                then we must be in b, and thus add a point to concentration*/
                concentration[thread_index] += 1;
            }
            else
            {
                // We must have lost a concentration if we get this far.
                // Ensure that we never go negative on concentration
                concentration[thread_index] = max(1, concentration[thread_index] + 1);
            }
        }
        else
        {
            // OFF
            if (temp_rand < (kon / (kon + concentration[thread_index] * g)))
            {
                /* Then we need to change to the on state */
                state[thread_index] = 1;
            }
            else
            {
                // There are only two options and they are mutually exclusive
                // Ensure that we never go negative on concentration
                concentration[thread_index] = max(1, concentration[thread_index] + 1);
            }
        }
        thread_index += (blockDim.x * gridDim.x);
    }
}

// Calls the Gill Kernel with its arguments.
void cudaCallGillKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float b,
        const float g,
        const float kon,
        const float koff,
        const float* random_distrib,
        const int len_rand,
        int* state,
        float* concentration
        float* times) {
        
    cudaGillKernel<<<blocks, threadsPerBlock>>> (b,g,kon,koff,random_distrib, state
        len_rand, concentration, times);
}

__global__
void cudaUniKernel(float* times, const float step, int n_times)
{
    // Set up the GPU's version of the i variable.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_index < (unsigned int)n_times)
    {
        times[thread_index] = min(times[thread_index], step);

        thread_index += (blockDim.x * gridDim.x);
    }


}

void cudaCallUniKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float* times
        const float step
        int n_times) {
    cudaCallUniKernel<<<blocks, threadsPerBlock>>> (times, step, n_times);
}


__global__
void cudaStatKernel(float* times, int n_times, float* mean, float* varience, const int n_instant)
{
    // Set up the GPU's version of the i variable.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_index < (unsigned int) n_instant)
    {
        // Compute the mean
        for (int i = 0; i < n_times; ++i)
        {
            mean[thread_index] += times[i][thread_index];
            varience[thread_index];

        }
        mean[thread_index] /= n_instant;

        // Compute the varience
        for (int i = 0; i < n_times; ++i)
        {
            varience[thread_index] += ((times[i][thread_index] - mean[thread_index])
                * (times[i][thread_index] - mean[thread_index]));
        }
        varience[thread_index] /= n_instant;

        thread_index += (blockDim.x * gridDim.x);
    }
}

void cudaCallStatKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float* times
        int n_times
        float* mean
        float* varience
        const int num_instant){

    cudaCallUniKernel<<<blocks, threadsPerBlock>>> (times, n_times, mean,
        varience, num_instant);
}