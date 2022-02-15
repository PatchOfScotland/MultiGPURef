#ifndef CONSTANTS_H
#define CONSTANTS_H

#define ITERATIONS 25
#define BLOCKSIZE 256
#define EPSILON 1e-5

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int gpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

void LogHardware(char* filename){
  std::ofstream logging;
    logging.open(filename);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    logging << "Number of devices: " << deviceCount << "\n";
    for (int i = 0; i < deviceCount; i++){
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, i);
        logging << "Device " << i << " name: " << properties.name << "\n";
        logging << "Device can use Unified Memory:" << properties.unifiedAddressing << "\n";
    }
    for (int i = 0; i < deviceCount; i++){
        for(int j = 0; j < deviceCount; j++){
            if (i==j) continue;
            int canAccessPeer = 0;
            cudaDeviceCanAccessPeer(&canAccessPeer, i,j);
            if (canAccessPeer){
                logging << "Device "<< i << " can access Device " << j << "\n";
            } else {
                logging << "Device "<< i << " cannot access Device " << j << "\n";
            }
        }
    }
}


#endif