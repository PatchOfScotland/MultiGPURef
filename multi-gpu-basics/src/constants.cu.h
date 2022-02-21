#ifndef CONSTANTS_H
#define CONSTANTS_H

#define ITERATIONS 25
#define BLOCKSIZE 256
#define EPSILON 1e-5

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int gpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
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