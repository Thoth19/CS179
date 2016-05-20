#ifndef partTwo_CUH
#define partTwo_CUH

void cudaCallGillKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float b,
        const float g,
        const float kon,
        const float koff,
        const float* random_distrib,
        const int len_rand,
        int* state;
        float* concentration
        float* times);

void cudaCallUniKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float* times
        const float step
        int n_times);

void cudaCallStatKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float* times
        int n_times
        float* mean
        float* varience
        const int num_instant);
#endif