/* Part Two of Assignment Six */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>


#include <cuda_runtime.h>
#include <algorithm>

#include "partTwo_cuda.cuh"
#include "ta_utilities.hpp"

#ifndef N_INSTANCES
#define N_INSTANCES 100
#endif
#ifndef N_TIMES
#define N_TIMES 1000
#endif

int main(int argc, char* argv[]) {
  // These functions allow you to select the least utilized GPU
  // on your system as well as enforce a time limit on program execution.
  // Please leave these enabled as a courtesy to your fellow classmates
  // if you are using a shared computer. You may ignore or remove these
  // functions if you are running on your local machine.
  TA_Utilities::select_least_utilized_GPU();
  int max_time_allowed_in_seconds = 40;
  TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

  if (argc < 3){
      printf("Usage: (threads per block) (max number of blocks)\n");
      exit(-1);
  }

  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);

  // Part One

  // Random timesteps
  float* dev_points;
  cudaMalloc((void **) &dev_points, N_INSTANCES * sizeof(float));

  // Need to copy the concentration.
  float concen[N_INSTANCES];
  float* dev_concen;
  cudaMalloc((void **) &dev_concen, N_INSTANCES * sizeof(float));
  cudaMemset(dev_concen, 1, N_INSTANCES * sizeof(float));

  int state[N_INSTANCES];
  int* dev_state;
  cudaMalloc((void **) &dev_state, N_INSTANCES * sizeof(int));
  cudaMemset(dev_state, 0, N_INSTANCES * sizeof(int));

  float times[N_INSTANCES];
  float* dev_times;
  cudaMalloc((void **) &dev_times, N_INSTANCES * sizeof(float));
  cudaMemset(dev_times, 0, N_INSTANCES * sizeof(float));
  
  // Generates 100 random floats between 0 and 1 to run the kernel on.
  curandGenerator_t gen;
  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, dev_points, N_INSTANCES); 

  // Call kernel
  cudaCallGillKernel(maxBlocks, threadsPerBlock, 10, 1, 0.1, 0.9, dev_points,
   N_INSTANCES, dev_state, dev_concen, dev_times);

  // Copy memory back.
  cudaMemcpy(concen, dev_concen, N_INSTANCES * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(state, dev_state, N_INSTANCES * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(times, dev_times, N_INSTANCES * sizeof(float), cudaMemcpyDeviceToHost);


  // There's nothing in the directions asking us to do anything with the output.
  // I don't know what format it should be printed/saved to file with so, ..
  printf("First Concentration: %f\n", concen[0]);
  printf("First State: %d\n", state[0]);
  printf("First timestep: %f\n", times[0]);

  // Free variables
  cudaFree(dev_points);
  cudaFree(dev_state);
  cudaFree(dev_times);

  // Part Two 
  // We want to renormalize the data from part One, so we remove the code that
  // frees the data. and continue from there.
  // We didn't remove the Memcpy's because we still want to print the data
  cudaCallUniKernel(maxBlocks, threadsPerBlock, 10, 1, 0.1, 0.9, dev_points,
   N_INSTANCES, dev_state, dev_concen, dev_times);

    // Random timesteps
  float* dev_points;
  cudaMalloc((void **) &dev_points, N_INSTANCES * sizeof(float));

  // Need to copy the concentration.
  float concen[N_TIMES][N_INSTANCES];
  float* dev_concen;
  cudaMalloc((void **) &dev_concen, N_INSTANCES * sizeof(float));
  cudaMemset(dev_concen, 1, N_INSTANCES * sizeof(float));

  int state[N_TIMES][N_INSTANCES];
  int* dev_state;
  cudaMalloc((void **) &dev_state, N_INSTANCES * sizeof(int));
  cudaMemset(dev_state, 0, N_INSTANCES * sizeof(int));

  float times[N_TIMES][N_INSTANCES];
  float* dev_times;
  cudaMalloc((void **) &dev_times, N_INSTANCES * sizeof(float));
  cudaMemset(dev_times, 0, N_INSTANCES * sizeof(float));
  
  // Generates 100 random floats between 0 and 1 to run the kernel on.
  curandGenerator_t gen;
  curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);

  time_counter = 0;
  while(10 * time_counter < N_TIMES)
  {
        // new set of random data
      curandGenerateUniform(gen, dev_points, N_INSTANCES); 
      // Call kernel
      cudaCallGillKernel(maxBlocks, threadsPerBlock, 10, 1, 0.1, 0.9, dev_points,
       N_INSTANCES, dev_state, dev_concen, dev_times);
      cudaCallUniKernel(maxBlocks, threadsPerBlock, dev_times, time_counter, N_INSTANCES);
      
      // Copy memory back.
      cudaMemcpy(concen[time_counter], dev_concen, N_INSTANCES * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(state[time_counter], dev_state, N_INSTANCES * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(times[time_counter], dev_times, N_INSTANCES * sizeof(float), cudaMemcpyDeviceToHost);

      // update the end condition
      time_counter += .1;
      for (int i = 0; i < N_INSTANCES; ++i)
      {
          /* Find the slowest moving simulation*/
        time_stop_cond = min(time_stop_cond, times[time_counter][i]);
      }
  }

  // Part Three
  // Now we want to compute mean and varience.
  float mean[N_INSTANCES];
  float* dev_mean;
  cudaMalloc((void **) &dev_mean, N_INSTANCES * sizeof(float));
  cudaMemset(dev_mean, 0, N_INSTANCES * sizeof(float));

  float varience[N_INSTANCES];
  float* dev_varience;
  cudaMalloc((void **) &dev_varience, N_INSTANCES * sizeof(float));
  cudaMemset(dev_varience, 0, N_INSTANCES * sizeof(float));

  float* dev_full_times[N_TIMES];
  for (int i = 0; i < N_TIMES; ++i)
  {
      cudaMalloc((void **) &dev_full_times, N_INSTANCES * sizeof(float));
      cudaMemcpy(dev_full_times[i], times[i], N_INSTANCES * sizeof(float), cudaMemcpyHostToDevice);
  }

  cudaCallStatKernel(maxBlocks, threadsPerBlock, dev_full_times, N_TIMES,
    dev_mean, dev_varience, N_INSTANCES);

  cudaMemcpy(mean, dev_mean, N_INSTANCES * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(varience, dev_varience, N_INSTANCES * sizeof(float), cudaMemcpyDeviceToHost);

  // We print out the first ones as a sample
  printf("Mean: %f\n", mean[0]);
  printf("Varience: %f\n", varience[0]);

      
      // Free variables
      cudaFree(dev_points);
      cudaFree(dev_state);
      cudaFree(dev_times);
      cudaFree(dev_mean);
      cudaFree(dev_varience);

}