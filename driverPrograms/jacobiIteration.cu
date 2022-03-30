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

#define X 4096
#define Y 4096  
#define GPU_RUNS 25
#define MAX_ITER 1000
#define TOL 1e-8
#define DEBUG
#define UNIFIED 1


typedef int functype;

const char* program = "extern \"C\"                                                        \n\
        __global__ void init_boundaries(                                                   \n\
                float* __restrict__ const a1,                                              \n\
                const float pi,                                                            \n\
                const int h,                                                               \n\
                const int w                                                                \n\
            ){                                                                             \n\
            const long long idx = blockDim.x * blockIdx.x + threadIdx.x;                   \n\
            if(idx < h) {                                                                  \n\
                const float y_value = sin(2.0 * pi * idx / (h - 1));                       \n\
                a1[idx * w] = y_value;                                                     \n\
                a1[idx * w + (w - 1)] = y_value;                                           \n\
            }                                                                              \n\
        }                                                                                  \n\
                                                                                           \n\
        extern \"C\" __global__ void jacobiKernel(                                                      \n\
            float* src,                                                                    \n\
            float* dst,                                                                    \n\
            float* norm,                                                                   \n\
            const int h,                                                                   \n\
            const int w,                                                                   \n\
            const int offRows                                                              \n\
        ){                                                                                 \n\
            __shared__ float scanMem[32*32];                                               \n\
                                                                                           \n\
            const long long x = blockIdx.x * blockDim.x + threadIdx.x;                       \n\
            const long long y = blockDim.y * offRows + blockIdx.y * blockDim.y + threadIdx.y;\n\
                                                                                           \n\
            const float xp1 = (x + 1 < w) ? src[y * w + x + 1] : 0;                        \n\
            const float xm1 = (x - 1 >= 0) ? src[y * w + x - 1] : 0;                       \n\
            const float yp1 = (y + 1 < h) ? src[(y + 1) * w + x] : 0;                      \n\
            const float ym1 = (y - 1 >= 0) ? src[(y - 1) * w + x] : 0;                     \n\
                                                                                           \n\
            const float newValue = (xp1 + xm1 + yp1 + ym1) / 4;                            \n\
            dst[y * w + x] = newValue;                                                     \n\
            const float local_norm = powf(src[y * w + x] - newValue, 2);                   \n\
                                                                                           \n\
            scanMem[threadIdx.y * blockDim.x + threadIdx.x] = local_norm;                  \n\
            if(threadIdx.x == 0 && threadIdx.y == 0){                                      \n\
                atomicAdd_system(norm, scanMem[blockDim.x * blockDim.y - 1]);              \n\
            }                                                                              \n\
        }\n";


void swap(void* x, void* y){
    void* temp;
    temp = x;
    x = y;
    y = temp;
}

int main(int argc, char** argv){
    cuInit(0);
    int DeviceCount;
    cuDeviceGetCount(&DeviceCount);

    uint x = X;
    uint y = Y;
    const uint BlockSize = 32;
    
    uint Rows = (x % 32 == 0) ? x / BlockSize : x / BlockSize + 1;
    uint Cols = (y % 32 == 0) ? y / BlockSize : x / BlockSize + 1;

    uint highRows = Rows % DeviceCount;
    uint rows_per_device_low = Rows / DeviceCount;
    uint rows_per_device_high = rows_per_device_low + 1;

    const size_t bufferSize = x*y*sizeof(float);
    
    CUmodule* modules   = (CUmodule*)malloc(sizeof(CUmodule)*DeviceCount);
    CUdevice* devices   = (CUdevice*)malloc(sizeof(CUdevice)*DeviceCount);
    CUcontext* contexts = (CUcontext*)malloc(sizeof(CUcontext)*DeviceCount);
    CUstream* streams   = (CUstream*)malloc(sizeof(CUstream)*DeviceCount);
    CUevent*  BenchmarkEvents = (CUevent*)malloc(sizeof(CUevent)*DeviceCount*2);
    CUevent*  ComputeDoneEvents = (CUevent*)malloc(sizeof(CUevent)*DeviceCount*2);
    CUfunction* Kernels = (CUfunction*)malloc(sizeof(CUfunction)*DeviceCount); 
    CUfunction* inits   = (CUfunction*)malloc(sizeof(CUfunction)*DeviceCount); 

    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuDeviceGet(&devices[devID], devID));
        CUDA_SAFE_CALL(cuCtxCreate(&contexts[devID], CU_CTX_SCHED_AUTO, devices[devID])); // This Automaticly set the device
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
        CUDA_SAFE_CALL(cuStreamCreate(&streams[devID], CU_STREAM_DEFAULT));
        CUDA_SAFE_CALL(cuEventCreate(&ComputeDoneEvents[devID*2] ,CU_EVENT_DEFAULT)); // Start Event
        CUDA_SAFE_CALL(cuEventCreate(&ComputeDoneEvents[devID*2 + 1] ,CU_EVENT_DEFAULT)); // Stop Event
    }

    CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[0]));
    CUDA_SAFE_CALL(cuEventCreate(&BenchmarkEvents[0] , CU_EVENT_DEFAULT)); // Start Event
    CUDA_SAFE_CALL(cuEventCreate(&BenchmarkEvents[1],CU_EVENT_DEFAULT)); // Stop Event

    const int FunctionCount = 2;
    char** functionNames = (char**)malloc(sizeof(char*)*FunctionCount);
    char functionName_1[] = "jacobiKernel";
    char functionName_2[] = "init_boundaries";
    functionNames[0] = functionName_1;
    functionNames[1] = functionName_2;
    
    compileFunctions(program, functionNames, Kernels, 2, modules, contexts, DeviceCount);
    
    

    
    CUdeviceptr mem_1, mem_2;
    CUdeviceptr* norms = (CUdeviceptr*)malloc(sizeof(CUdeviceptr)*DeviceCount);

    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
        CUDA_SAFE_CALL(cuMemAlloc(&norms[devID], sizeof(float)));
    }
    
    CUDA_SAFE_CALL(cuMemAllocManaged(&mem_1, bufferSize, CU_MEM_ATTACH_GLOBAL));
    CUDA_SAFE_CALL(cuMemAllocManaged(&mem_2, bufferSize, CU_MEM_ATTACH_GLOBAL));
    
    float pi = 3.14159265359;

    for(int run = 0; run < GPU_RUNS; run++){    
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[0]));
        void* initArgs_1[] = {&mem_1, &pi, &y, &x };
        CUDA_SAFE_CALL(cuLaunchKernel(Kernels[1], Cols, Rows, 1 , BlockSize, BlockSize, 1, 0, streams[0], initArgs_1, 0)); // Initialize Boundaries
        void* initArgs_2[] = {&mem_2, &pi, &y, &x };
        CUDA_SAFE_CALL(cuLaunchKernel(Kernels[1], Cols, Rows, 1 , BlockSize, BlockSize, 1, 0, streams[0], initArgs_2, 0)); // Initialize Boundaries
        CUDA_SAFE_CALL(cuCtxSynchronize());
        CUDA_SAFE_CALL(cuEventRecord(BenchmarkEvents[0], streams[0]));
        // Jacobi Iteration
        float normH = 1;
        uint iter = 0;
        while(normH > TOL && iter < MAX_ITER){

            for(int devID=0; devID < DeviceCount; devID++){
                const int top = devID > 0 ? devID - 1 : DeviceCount - 1; 
                const int bottom = (devID + 1) % DeviceCount;
                size_t brows = (devID < highRows) ? rows_per_device_high : rows_per_device_low;
                int offset = (devID < highRows) ? devID * rows_per_device_high : (devID - highRows)*rows_per_device_low + highRows * rows_per_device_high;
                void* args_even[] =  { &mem_1, &mem_2, &norms[devID], &y, &x, &offset};
                void* args_ueven[] = { &mem_2, &mem_1, &norms[devID], &y, &x, &offset};
                

                CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
                CUDA_SAFE_CALL(cuMemsetD32Async(norms[devID], 0, 1, streams[devID]));
                CUDA_SAFE_CALL(cuStreamWaitEvent(streams[devID], ComputeDoneEvents[top*2 + (iter % 2)],0));
                CUDA_SAFE_CALL(cuStreamWaitEvent(streams[devID], ComputeDoneEvents[bottom*2 + (iter % 2)],0));

                if (iter % 2) {
                    CUDA_SAFE_CALL(cuLaunchKernel(Kernels[devID*2], // Jacobi Kernel
                        Cols, brows, 1, 
                        BlockSize, BlockSize ,1 , 
                        BlockSize*BlockSize*sizeof(float), streams[devID], 
                        args_ueven, 0
                    ));
                } else {
                    CUDA_SAFE_CALL(cuLaunchKernel(Kernels[devID*2], // Jacobi Kernel
                        Cols, brows, 1, 
                        BlockSize, BlockSize ,1 , 
                        BlockSize*BlockSize*sizeof(float), streams[devID], 
                        args_even, 0
                    ));
                }
                
                CUDA_SAFE_CALL(cuEventRecord(ComputeDoneEvents[devID*2 + ((iter + 1) % 2)], streams[devID]));
            }
            
            float normTemps[DeviceCount];
            normH = 0;
            for(int devID = 0; devID < DeviceCount; devID++){
                CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
                cuMemcpyDtoHAsync(normTemps + devID, norms[devID], sizeof(float), streams[devID]);
            }

            for(int devID = 0; devID < DeviceCount; devID++){
                CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
                CUDA_SAFE_CALL(cuStreamSynchronize(streams[devID]));
            }

            for(int idx = 0; idx < DeviceCount; idx++){
                normH += normTemps[idx];
            }

            normH = sqrt(normH);
            
            iter++;   
        }

        for(int devID = 0; devID < DeviceCount; devID++){
            CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
            CUDA_SAFE_CALL(cuStreamSynchronize(streams[devID]));
        }

        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[0]));
        CUDA_SAFE_CALL(cuEventRecord(BenchmarkEvents[1], streams[0]));
        float runTime;
        
        CUDA_SAFE_CALL(cuEventSynchronize(BenchmarkEvents[1]));
        CUDA_SAFE_CALL(cuEventElapsedTime(&runTime, BenchmarkEvents[0], BenchmarkEvents[1]));
        printf("%f\n", runTime);
    }

    
    
    // Free data
    /*
    CUDA_SAFE_CALL(cuMemFree(mem_1));
    CUDA_SAFE_CALL(cuMemFree(mem_2));

    CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[0]));
    CUDA_SAFE_CALL(cuEventDestroy(BenchmarkEvents[0]));
    CUDA_SAFE_CALL(cuEventDestroy(BenchmarkEvents[1]));
        
    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
        CUDA_SAFE_CALL(cuMemFree(norms[devID]));
        CUDA_SAFE_CALL(cuModuleUnload(modules[devID]));
    }
    free(norms); 

    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
        CUDA_SAFE_CALL(cuEventDestroy(ComputeDoneEvents[devID*2]));
        CUDA_SAFE_CALL(cuEventDestroy(ComputeDoneEvents[devID*2 + 1]));
        CUDA_SAFE_CALL(cuStreamDestroy(streams[devID])); //Destroy Streams first
        CUDA_SAFE_CALL(cuCtxDestroy(contexts[devID]));
    }


    
    free(Kernels);
    free(functionNames);
    free(streams);
    free(contexts);
    free(devices);
    */


}