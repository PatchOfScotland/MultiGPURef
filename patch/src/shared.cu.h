#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

// Checking Cuda Call
#define CCC(call)                                       \
    {                                                   \
        cudaError_t cudaStatus = call;                  \
        if (cudaSuccess != cudaStatus) {                \
            std::cerr << "ERROR: CUDA RT call \""       \
                      << #call                          \
                      << "\" in line "                  \
                      << __LINE__                       \
                      << " of file "                    \
                      << __FILE__                       \
                      << " failed with "                \
                      << cudaGetErrorString(cudaStatus) \
                      << " ("                           \
                      << cudaStatus                     \
                      <<").\n",                         \
            exit(cudaStatus);                           \
        }                                               \
    }

#endif