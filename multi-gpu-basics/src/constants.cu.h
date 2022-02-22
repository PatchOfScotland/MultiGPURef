#ifndef CONSTANTS_H
#define CONSTANTS_H

#define ITERATIONS 25
#define BLOCKSIZE 256
#define EPSILON 1e-5

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
  cudaSetDevice(0);
  }
}


#endif