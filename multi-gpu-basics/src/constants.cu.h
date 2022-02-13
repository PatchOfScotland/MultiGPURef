#ifndef CONSTANTS_H
#define CONSTANTS_H

#define BLOCKSIZE 256
#define EPSILON 1e-5

#include <iostream>

int gpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}


#endif