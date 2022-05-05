#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "nvrtcHelpers.cu.h"
#include "constants.cu.h"


int main(int argc, char** argv){
  cuInit(0);

  CUdevice dev;
  CUcontext ctx;
  CUDA_SAFE_CALL(cuDeviceGet(&dev, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&ctx, dev));

  int* arr = (int*)malloc(10*sizeof(int));
  for(int i = 0; i < 10; i++){
    arr[i] = i + 1;
  }
  CUdeviceptr mem_in, mem_out;

  CUDA_SAFE_CALL(cuMemAllocManaged(&mem_in, sizeof(int)*10, CU_MEM_ATTACH_GLOBAL));
  CUDA_SAFE_CALL(cuMemAllocManaged(&mem_out, sizeof(int), CU_MEM_ATTACH_GLOBAL));

  CUDA_SAFE_CALL(cuMemcpyHtoD(mem_in, arr, sizeof(int)*10));
  CUDA_SAFE_CALL(cuMemcpyDtoD(mem_out, mem_in + sizeof(int), sizeof(int)));
  CUDA_SAFE_CALL(cuMemcpyDtoH(arr, mem_out, sizeof(int)));

  fprintf(stderr, "%d\n", arr[0]);

  free(arr);
  CUDA_SAFE_CALL(cuMemFree(mem_in));
  CUDA_SAFE_CALL(cuMemFree(mem_out));

  return 0;
}