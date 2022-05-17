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

template <class T>
void NaiveFetch(T arr, int64_t N){
  int device;
  cudaGetDevice(&device);
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  size_t N_per_device = N / deviceCount;

  int offset = 0;

  for(int devID = 0; devID < deviceCount; devID++){
    CUDA_RT_CALL(cudaSetDevice(devID));
    CUDA_RT_CALL(cudaMemPrefetchAsync(arr + offset, N_per_device * sizeof(T), devID, NULL));
  }
  cudaSetDevice(device);
}

template<class T>
void NaiveHint(T arr, size_t N) {
  int device;
  cudaGetDevice(&device);
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  size_t data_per_device = N / deviceCount;
  size_t offset = 0;
  size_t left = N;
  for (int devID = 0; devID < ctx->device_count; devID++) {
    CUDA_RT_CALL(cudaSetDevice(devID));
    if (devID != 0) {
      CUDA_RT_CALL(cudaMemAdvise(arr, offset * sizeof(T), CU_MEM_ADVISE_SET_ACCESSED_BY, devID));
    }
    CUDA_RT_CALL(cudaMemAdvise(arr + offset,
                                    data_per_device * sizeof(T), CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                                    devID));
    offset += data_per_device;
    left   -= data_per_device;
    if (devID != deviceCount -1) {
      CUDA_SUCCEED_FATAL(cudaMemAdvise(arr + offset, left * sizeof(T), CU_MEM_ADVISE_SET_ACCESSED_BY, devID));
    }
  }
}



template<class T>
void hintReadOnly(T arr, size_t N){
  // Device Argument ignore for Read mostly advice
  CUDA_RT_CALL(cudaMemAdvise(arr, N*sizeof(T), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
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
