#ifndef NVRTC_HELPERS_CU_H
#define NVRTC_HELPERS_CU_H

#include "cuda.h"
#include <cuda_runtime.h>
#include "nvrtc.h"
#include "constants.cu.h"

void compileFunctions(
        const char* program, 
        char** functionNames, 
        CUfunction* functions,
        int functionCount,
        CUmodule* modules, // List of CUmodules for each device
        CUcontext* contexts, // List of Contexts to be assoc with the module
        int DeviceCount
    ){
    nvrtcProgram kernelProgram; 
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&kernelProgram, program, NULL, 0, NULL, NULL));
    const char* opts[] = {"-arch=compute_61"};
    NVRTC_SAFE_CALL(nvrtcCompileProgram(kernelProgram, 1, opts)); 
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(kernelProgram, &logSize));

    if(logSize > 1){    
        char *log = (char*)malloc(logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(kernelProgram, log));
        printf("%s\n", log);
        free(log);
        exit(1);
    }

    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(kernelProgram, &ptxSize));
    char* ptx = (char*)malloc(ptxSize);
    
    NVRTC_SAFE_CALL(nvrtcGetPTX(kernelProgram, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&kernelProgram));
    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
        CUDA_SAFE_CALL(cuModuleLoadData(modules + devID, ptx));
    }

    free(ptx);
    for(int devID = 0; devID < DeviceCount; devID++){
        CUDA_SAFE_CALL(cuCtxSetCurrent(contexts[devID]));
        for(int funcIdx = 0; funcIdx < functionCount; funcIdx++){
            CUDA_SAFE_CALL(cuModuleGetFunction(&functions[devID*functionCount + funcIdx], modules[devID], functionNames[funcIdx]));
        }   
    }
    
}




#endif