/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH


/* This is a CUDA header file.
Declaration of kernels.*/
 
void callOneDimWave(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *old
        const float *curr,
        float *new,
        const unsigned int n_Nodes);

void cudaOneDimWaveKernel(const float *old, const float *curr, float *new,
    int n_Nodes);

#endif // CUDA_1D_FD_WAVE_CUDA_CUH
