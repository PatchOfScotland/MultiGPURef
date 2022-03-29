#ifndef _GNU_SOURCE // Avoid possible double-definition warning.
#define _GNU_SOURCE
#endif

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

#define ARRAY_LENGTH 1e8
#define GPU_RUNS 25
#define DEBUG
#define UNIFIED 1


typedef int functype;

const char* program = "extern \"C\" __global__ void mapFunction(                  \n\
        int *mem_in,                                                              \n\
        int *mem_out,                                                             \n\
        const size_t arr,                                                         \n\
        const int offset ){                                                       \n\
    int idx = blockDim.x*gridDim.x*offset + blockDim.x * blockIdx.x + threadIdx.x;\n\
    if(idx < arr){                                                                \n\
        mem_out[idx] = mem_in[idx] + 1;                                           \n\
    }                                                                             \n\
}\n";






int main(int argc, char** argv){
    

    cuInit(0);
    int DeviceCount;
    cuDeviceGetCount(&DeviceCount);

    size_t N = ARRAY_LENGTH;
    const u_int BlockSize = 1024;
    const u_int NumBlocks = (N + BlockSize - 1) / BlockSize;
    const int BlocksPerDevice = NumBlocks / DeviceCount + 1;
    const size_t bufferSize = N*sizeof(int);

    CUmodule module;
    CUcontext CurrentContext;
    CUdevice* devices   = (CUdevice*)malloc(sizeof(CUdevice)*DeviceCount);
    CUcontext* contexts = (CUcontext*)malloc(sizeof(CUcontext)*DeviceCount);
    CUstream* streams   = (CUstream*)malloc(sizeof(CUstream)*DeviceCount);
    CUevent*  BenchmarkEvents = (CUevent*)malloc(sizeof(CUevent)*DeviceCount*2);

    cuCtxGetCurrent(&CurrentContext);

    if (CurrentContext != NULL) printf("Current Context is not null\n");

    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuDeviceGet(&devices[devID], devID));

        CUDA_SAFE_CALL(cuCtxCreate(&contexts[devID], CU_CTX_SCHED_AUTO, devices[devID]));
        CUDA_SAFE_CALL(cuStreamCreate(&streams[devID], CU_STREAM_DEFAULT));
        CUDA_SAFE_CALL(cuEventCreate(&BenchmarkEvents[devID*2] ,CU_EVENT_DEFAULT)); // Start Event
        CUDA_SAFE_CALL(cuEventCreate(&BenchmarkEvents[devID*2 + 1] ,CU_EVENT_DEFAULT)); // Stop Event
    }

    cuCtxGetCurrent(&CurrentContext);

    if (CurrentContext == NULL) printf("Current Context is null\n");
    
    if (CurrentContext != contexts[DeviceCount - 1]) printf("Current context is not equal to last created context\n");


    CUfunction* Kernels = malloc(sizeof(CUfunction)); 
    char** functionNames = (char**)malloc(sizeof(char*));
    char* functionName_1 = "mapFunction";
    functionNames[0] = functionName_1;

    compileFunctions(program, functionNames, Kernels, 1, &module);

    CUdeviceptr mem_in, mem_out;
    #if UNIFIED
    CUDA_SAFE_CALL(cuMemAllocManaged(&mem_in,  bufferSize, CU_MEM_ATTACH_GLOBAL));
    CUDA_SAFE_CALL(cuMemAllocManaged(&mem_out, bufferSize, CU_MEM_ATTACH_GLOBAL));
    #else
    CUDA_SAFE_CALL(cuMemAlloc(&mem_in, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&mem_out, bufferSize));
    #endif

    int* destData = (int*)malloc(N*sizeof(int));    

    #if UNIFIED
    int* hostData = (int*)mem_in;
    for(int i = 0; i < N; i++){
        hostData[i] = i;
    }
    #else
    int* hostData = (int*)malloc(N*sizeof(int));
    for(int i = 0; i < N; i++){
        hostData[i] = i;
    }
    CUDA_SAFE_CALL(cuMemcpyHtoD(mem_in, hostData, N*sizeof(int)));
    #endif


    #if UNIFIED
    for(int devID = 0; devID < DeviceCount; devID++){
        const size_t ElemsPerDevice = BlocksPerDevice*BlockSize;
        const size_t offset = ElemsPerDevice * devID;
        const size_t ElementsToPrefetch = (offset + ElemsPerDevice < N) ? ElemsPerDevice : N - offset;
        CUDA_SAFE_CALL(cuMemPrefetchAsync(mem_in + offset, ElementsToPrefetch*sizeof(int), devices[devID], streams[devID]));
    } 
    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuStreamSynchronize(streams[devID]));
    }

    #endif


    for(int run = 0; run < GPU_RUNS; run++){
        
    for(int devID = 0; devID < DeviceCount; devID++){
            void *args[] = {&mem_in, &mem_out, &N, &devID};
            CUDA_SAFE_CALL(cuEventRecord(BenchmarkEvents[devID*2], streams[devID]));
            CUDA_SAFE_CALL(cuLaunchKernel(Kernels[0], 
                BlocksPerDevice, 1, 1, 
                BlockSize, 1 ,1 , 
                0, streams[devID], 
                args, 0
            ));
            CUDA_SAFE_CALL(cuEventRecord(BenchmarkEvents[devID*2 + 1], streams[devID]));
        }
        float runTimes[DeviceCount];
        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_SAFE_CALL(cuStreamSynchronize(streams[devID]));
            CUDA_SAFE_CALL(cuEventElapsedTime(&runTimes[devID], BenchmarkEvents[devID*2], BenchmarkEvents[devID*2 + 1]));
            printf("%f\n", runTimes[devID]);
        }
    }

    CUDA_SAFE_CALL(cuMemcpyDtoH(destData, mem_out, N*sizeof(int)));

    for(int i = 0; i < N; i++){
        if(destData[i] != hostData[i] + 1){
            printf("Error at Index :%d\n", i);
        }
    }

    
    // Free data
    cuModuleUnload(module);

    for(int devID = 0; devID < DeviceCount; devID++){
        cuEventDestroy(BenchmarkEvents[devID*2]);
        cuEventDestroy(BenchmarkEvents[devID*2 + 1]);
        cuStreamDestroy(streams[devID]); //Destroy Streams first
        cuCtxDestroy(contexts[devID]);
    }


    
    free(Kernels);
    free(functionNames);

    free(streams);
    free(contexts);
    free(devices);



}