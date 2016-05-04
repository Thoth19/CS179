#ifndef CUDA_CLASSIFY_CUH
#define CUDA_CLASSIFY_CUH

#define REVIEW_DIM 50
#define STEP_SIZE 1

float cudaClassify(
    float **data,
    int batch_size, 
    float step_size,
    float *weights, 
    cudaStream_t stream);

#endif
