
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)

Modified by Jordan Bonilla and Matthew Cedeno (2016)
*/

#include <cstdio>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
#include "ta_utilities.hpp"

#define PI 3.14159265358979


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

void cudaCallComplexArrayToFloat(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        float *out_data,
        const unsigned int length);
void cudaCallHighPass(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *out_data,
        const unsigned int length);

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
        cufftComplex *out_data,
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


int main(int argc, char** argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Begin timer and check for the correct number of inputs
    time_t start = clock();
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input sinogram text file's name > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output text file's name >\n");
        exit(EXIT_FAILURE);
    }






    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    cudaMalloc((void **) &output_dev, size_result * sizeof(float));
    cudaMalloc((void **) &dev_sinogram_cmplx, sinogram_width*nAngles*sizeof(cufftComplex));
    cudaMalloc((void **) &dev_sinogram_float, sinogram_width*nAngles*sizeof(float));

    cudaMemcpy(dev_sinogram_cmplx, sinogram_host, sinogram_width*nAngles*sizeof(cufftComplex), cudaMemcpyHostToDevice);

    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    cufftHandle plan;
    int batch = 1;
    cufftPlan1d(&plan, sinogram_width*nAngles*sizeof(cufftComplex), CUFFT_C2C, batch);

    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);

    // Do frequency scaling
    cudaCallHighPass(nBlocks, threadsPerBlock, dev_sinogram_cmplx, dev_sinogram_cmplx,
        sinogram_width*nAngles);

    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);
    // Remove the imaginary parts
    cudaCallComplexArrayToFloat(nBlocks, threadsPerBlock, 
            dev_sinogram_cmplx, dev_sinogram_float,
            sinogram_width * nAngles);

    // Clean up plan and obsolete variable
    cufftDestroy(plan);
    cudaFree(dev_sinogram_cmplx);



    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    // Free remaining memory
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);

    
    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);
    printf("CT reconstruction complete. Total run time: %f seconds\n", (float) (clock() - start) / 1000.0);
    return 0;
}



