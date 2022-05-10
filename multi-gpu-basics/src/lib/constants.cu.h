#ifndef CONSTANTS_H
#define CONSTANTS_H

#define ITERATIONS 25
#define BLOCKSIZE 256
#define EPSILON 1e-5
#define ELEMS_PER_THREAD 9
#define DEBUG_INFO true
#define WARP 32
#define lgWARP 5

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

typedef unsigned int uint32_t;
typedef int           int32_t;

__device__ uint32_t counter = 0;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                               \
        }                                                                                   \
    }


int gpuaAssert(cudaError_t code) {
  if (code != cudaSuccess) {

            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                  __LINE__, __FILE__, cudaGetErrorString(code), code); \
            return( code );                                                             \
    }
  return 0;
}


void EnablePeerAccess(){
  int Device;
  cudaGetDevice(&Device);
  int DeviceCount;
  cudaGetDeviceCount(&DeviceCount);
  for(int i = 0; i < DeviceCount; i++){
    cudaSetDevice(i);
    for(int j = 0; j < DeviceCount; j++){
      if(i == j){
        continue;
      } else {
        int canAccess = 0;
        if(cudaDeviceCanAccessPeer(&canAccess, i, j)){
          cudaDeviceEnablePeerAccess(j,0);
        }
      }
    }
  cudaSetDevice(Device);
  }
}

uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

void initHwd() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_BLOCK = prop.maxThreadsPerBlock;
    MAX_SHMEM = prop.sharedMemPerBlock;

    if (DEBUG_INFO) {
        printf("Device name: %s\n", prop.name);
        printf("Number of hardware threads: %d\n", MAX_HWDTH);
        printf("Max block size: %d\n", MAX_BLOCK);
        printf("Shared memory size: %d\n", MAX_SHMEM);
        puts("====");
    }
}

#endif