#ifndef CONSTANTS_CU_H
#define CONSTANTS_CU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      fprintf(stderr,                                             \
        "ERROR: CUDA RT call in line %d of file %s failed "       \
        "with "                                                   \
        "%s (%d).\n",                                             \
        __LINE__, __FILE__, msg, result);                           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
        const char* msg = nvrtcGetErrorString(result);            \
      fprintf(stderr,                                             \
        "ERROR: CUDA RT call in line %d of file %s failed "       \
        "with "                                                   \
        "%s (%d).\n",                                             \
        __LINE__, __FILE__, msg, result);                         \
    }                                                             \
  } while(0)

#endif