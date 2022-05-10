#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include "constants.cu.h"


template<class T>
void AllocateDeviceArray(T** data, size_t elements){
    int Device, DeviceCount;
    cudaGetDevice(&Device);
    cudaGetDeviceCount(&DeviceCount);
    for(int devID = 0; devID < DeviceCount; devID++){
        cudaSetDevice(devID);
        CUDA_RT_CALL(cudaMalloc(data + devID, elements*sizeof(T)));
    }
    cudaSetDevice(Device);
}

template<class T>
void hint1D(T* arr, int blockSize, size_t N){
  int DeviceCount;
  cudaGetDeviceCount(&DeviceCount);

  size_t total_num_blocks = (N + blockSize - 1) / blockSize;
  size_t block_per_device = (total_num_blocks + DeviceCount - 1) / DeviceCount;
  size_t offset = 0;
  for(int devID = 0; devID < DeviceCount; devID++){
    int count = min(block_per_device, N - offset);
    CUDA_RT_CALL(cudaMemAdvise(
      arr + offset,
      count * sizeof(T),
      cudaMemAdviseSetPreferredLocation,
      devID));
    offset += count;
  }
}

template<class T>
void hint2DWithBorder(
        T* arr,
        const int borderSize,
        const int blockSize,
        const int h,
        const int w
    ){

    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    const int rowBlocks = (h % blockSize == 0) ? h / blockSize : h / blockSize + 1;

    int rows_per_device_low  = rowBlocks / DeviceCount;
    int rows_per_device_high = rows_per_device_low + 1;
    int highRows = rowBlocks % DeviceCount;

    int64_t offset = 0;

    const int64_t FilterRowSize = w * borderSize;

    for(int devID = 0; devID < DeviceCount; devID++){
        const int64_t elems_main_block = (devID < highRows) ?
            rows_per_device_high * w * blockSize :
            rows_per_device_low * w * blockSize;

        CUDA_RT_CALL(cudaMemAdvise(
            arr + offset,
            elems_main_block * sizeof(T),
            cudaMemAdviseSetPreferredLocation,
            devID
        ));

        // Border
        if (devID != 0 && borderSize > 0)  CUDA_RT_CALL(cudaMemAdvise(
            arr + offset - FilterRowSize,
            FilterRowSize * sizeof(T),
            cudaMemAdviseSetAccessedBy,
            devID
        ));

        offset += elems_main_block;

        if (devID != DeviceCount - 1 && borderSize > 0) CUDA_RT_CALL(cudaMemAdvise(
            arr + offset,
            FilterRowSize * sizeof(float),
            cudaMemAdviseSetAccessedBy,
            devID
        ));
    }
}

template<class T, uint8_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
  #pragma unroll
  for (int8_t i=0; i<CHUNK; i++) {
    const int16_t loc_ind = threadIdx.x + i * blockDim.x;
    const uint32_t glb_ind = glb_offs + loc_ind;
    T elm = ne;

    if(glb_ind < N) {
      elm = d_inp[glb_ind];
    }

    shmem_inp[loc_ind] = elm;
  }
  __syncthreads();
}

template<class T, uint8_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_red
) {
  #pragma unroll
  for (int8_t i = 0; i < CHUNK; i++) {
    const int16_t loc_ind = threadIdx.x + i * blockDim.x;
    const uint32_t glb_ind = glb_offs + loc_ind;

    if (glb_ind < N) {
      T elm = const_cast<const T&>(shmem_red[loc_ind]);
      d_out[glb_ind] = elm;
    }
  }
  __syncthreads();
}

#endif
