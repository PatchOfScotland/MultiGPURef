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
        CUDA_RT_CALL(cudaMalloc(&data[devID], elements*sizeof(T)));
    }
    cudaSetDevice(Device);
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
        if (devID != 0) CUDA_RT_CALL(cudaMemAdvise(
            arr + offset - FilterRowSize,
            FilterRowSize * sizeof(T),
            cudaMemAdviseSetAccessedBy,
            devID
        ));

        offset += elems_main_block;
                
        if (devID != DeviceCount - 1) CUDA_RT_CALL(cudaMemAdvise(
            arr + offset,
            FilterRowSize * sizeof(float),
            cudaMemAdviseSetAccessedBy,
            devID
        ));
    }
}


#endif