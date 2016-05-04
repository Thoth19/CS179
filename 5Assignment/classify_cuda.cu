#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format is the last element is -1 if not a restaurant and 1 if it is.
            elements 0-49 are the bag of words values
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM (50).
 * error: Pointer to a single int used to describe the error for the batch.
 *         An output variable for the kernel. It is the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(
    float **data,
    int batch_size,
    float step_size,
    float *weights,
    int *error)
{
    //* error = 11;
    /* There is some error related to getting the data to save to *error.
    Suppose we set * error as above.
    If we return early at position 1, then the host sees 11 as intended.
    If we return at position 2, then the host will see the default value
    at position 3. Changing that default value proves that it does not change.
    However, I can't find any code between position 1 and 2 that should affect
    error. */
    extern __shared__ float sh_weights[];
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float dot_prod;
    float grad_coeff;
    float y_dot_prod[REVIEW_DIM]={};
    float grad[REVIEW_DIM];
    int i;
    int misclass;
    float weight_delta[REVIEW_DIM];
    for(i = 0; i < REVIEW_DIM; i++)
        weight_delta[i] = 0;
    // Pos 1
    while(thread_index < REVIEW_DIM)
    {
        sh_weights[thread_index] = weights[thread_index];
        thread_index += blockDim.x;
    }
    // Pos 2
    __syncthreads();
    thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_index < batch_size)
    {
        dot_prod = 0;
        // compute the dot_prod
        i = 0;
        while(i < REVIEW_DIM)
        {
            dot_prod += weights[i] * data[thread_index][i];
            i += 1;
        }

        //compute the number of misclassifications by comparing the signs
        // of y_n and the weight * data
        y_dot_prod[thread_index] = data[thread_index][REVIEW_DIM] * dot_prod;
        if (y_dot_prod < 0)
        {
            misclass += 1;
        }

        // compute gradient/updated weight values
        i = 0;
        grad_coeff = (-1/ batch_size) / (1 + exp(y_dot_prod[thread_index]));
        while (i < REVIEW_DIM)
        {
            grad[i] = grad_coeff * (data[thread_index][REVIEW_DIM] * data[thread_index][i]);
            weight_delta[i] -= step_size * grad[i];
        }
        thread_index += (blockDim.x * gridDim.x);
    }
    // Do the atomic adds
    thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_index < batch_size)
    {
        for (i = 0; i < REVIEW_DIM; ++i)
        {
            atomicAdd(&weights[i], - weight_delta[i]);
        }
        atomicAdd(error, misclass);
        thread_index += (blockDim.x * gridDim.x);
    }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float **data,
    int batch_size, 
    float step_size,
    float *weights, 
    cudaStream_t stream)
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = sizeof(float) * REVIEW_DIM;

    int *d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    cudaMemset(d_errors, 0, sizeof(int));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    int h_errors = 0;
    // position 3
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}
