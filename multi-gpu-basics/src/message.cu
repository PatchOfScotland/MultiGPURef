#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"

typedef uint8_t flag;

#define FLAG_REJECT 0x0
#define FLAG_ACCEPT 0x1

__global__ void kernel(int *shared, int *internal, flag *flags, const int devID, const int DeviceCount, cudaEvent_t *events, cudaStream_t *streams)
{
  for (int i = 0; i < DeviceCount; i++)
  {
    internal[i + DeviceCount * devID] = i + 1;
  }

  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    shared[devID] = internal[DeviceCount - 1];
    if (0 < devID){
      flag flagAcc = FLAG_ACCEPT;
    }
  }

  __syncthreads();

  for (int i = 0; i < DeviceCount; i++)
  {
    internal[i] += acc;
  }
}

int main(int argv, char *const argc)
{
  int DeviceCount;
  cudaGetDeviceCount(&DeviceCount);

  int *shared;
  int *arr;
  flag *flags;
  cudaMallocManaged(&arr, sizeof(int) * DeviceCount * DeviceCount);
  cudaMallocManaged(&shared, sizeof(int) * DeviceCount);
  cudaMallocManaged(&flags, sizeof(flag) * DeviceCount);
  cudaMemset(shared, 0, sizeof(int) * DeviceCount);

  for (int devID = DeviceCount - 1; devID >= 0; devID--)
  {
    cudaSetDevice(devID);
    kernel<<<1, 1, 0, streams[devID]>>>(shared, arr, devID, DeviceCount, events, streams);
  }

  syncronize();

  for (int i = 0; i < DeviceCount; i++)
  {
    std::cout << "Shared:" << shared[i] << "\nArray:";
    int *data = arr[i];
    for (int j = 0; j < DeviceCount; j++)
    {
      (j == DeviceCount - 1) ? std::cout << data[j] << "\n" : std::cout << data[j] << ", ";
    }
  }
}