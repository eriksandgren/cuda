
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N (1024 * 64)

__global__ void add(int* a, int* b, int* c) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N) 
  {
    c[tid] = a[tid] + b[tid];
    tid += blockDim.x * gridDim.x;
  }
}


int main()
{
  int a[N];
  int b[N];
  int c[N];
  int* dev_a;
  int* dev_b;
  int* dev_c;
  // Allocate memory for the GPU arrays
  cudaMalloc(&dev_a, N * sizeof(int));
  cudaMalloc(&dev_b, N * sizeof(int));
  cudaMalloc(&dev_c, N * sizeof(int));

  // Fill a and b with some "random" numbers
  for (int i = 0; i < N; i++)
  {
    a[i] = i;
    b[i] = i * i;
  }
  // Copy a and b to the gpu
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // Perform the addition
  add<<<128, 128>>>(dev_a, dev_b, dev_c);

  // Copy back the result to host
  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  // Verify calculation
  bool success = true;
  for (int i = 0; i < N; i++)
  {
    if ((a[i] + b[i]) != c[i])
    {
      printf("Error at index: %d, %d + %d != %d\n", i, a[i], b[i], c[i]);
      success = false;
    }
  }

  if (success)
  {
    printf("Vector addition successful!\n");
  }

  //free memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}