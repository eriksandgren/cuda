
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
  cudaDeviceProp prop;

  int count;
  cudaGetDeviceCount(&count);
  printf("Number of devices: %d\n", count);

  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop, i);

    printf("\n  --- General Information for Device %d ---  \n", i);
    printf("\tName: %s\n", prop.name);
    printf("\tCompute capability : %d.%d\n", prop.major, prop.minor);
    printf("\tClock Rate : %d\n", prop.clockRate);
    printf("\tDevice copy overlap: %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
    printf("\tKernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

    printf("\n  --- Memory Information for Device %d ---  \n", i);
    printf("\tTotal global mem : %ld bytes\n", prop.totalGlobalMem);
    printf("\tTotal constant mem : %ld bytes\n", prop.totalConstMem);
    printf("\tMax mem pitch : %ld bytes\n", prop.memPitch);
    printf("\tTexture alignment : %ld\n", prop.textureAlignment);

    printf("\n  --- Multiprocessor Information for Device %d ---  \n", i);
    printf("\tMultiprocessor count: %d\n", prop.multiProcessorCount);
    printf("\tShared memory per mp: %ld bytes\n", prop.sharedMemPerMultiprocessor);
    printf("\tRegisters per mp : %d\n", prop.regsPerBlock);
    printf("\tThreads in warp : %d\n", prop.warpSize);
    printf("\tMax threads per block %d \n", prop.maxThreadsPerBlock);
    printf("\tMax thread dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0],
                                                       prop.maxThreadsDim[1],
                                                       prop.maxThreadsDim[2]);
    printf("\tMax grid dimensions : (%d, %d, %d)\n", prop.maxGridSize[0],
                                                     prop.maxGridSize[1],
                                                     prop.maxGridSize[2]);
    
    printf("\n");

  }

    return 0;
}