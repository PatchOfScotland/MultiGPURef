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
        CUmodule* module
    ){
    nvrtcProgram kernelProgram; 
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&kernelProgram, program, NULL, 0, NULL, NULL));
    NVRTC_SAFE_CALL(nvrtcCompileProgram(kernelProgram, 0, NULL)); 
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(kernelProgram, &logSize));

    if(logSize > 1){
        
        char *log = (char*)malloc(logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(kernelProgram, log));
        printf("%s\n", log);
        free(log);
    }

    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(kernelProgram, &ptxSize));
    char* ptx = (char*)malloc(ptxSize);
    
    NVRTC_SAFE_CALL(nvrtcGetPTX(kernelProgram, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&kernelProgram));
    CUDA_SAFE_CALL(cuModuleLoadData(module, ptx));


    free(ptx);
    for(int funcIdx = 0; funcIdx < functionCount; funcIdx++){
        CUDA_SAFE_CALL(cuModuleGetFunction(&functions[funcIdx], *module, functionNames[funcIdx]));
    }
}




#endif