#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "map.cu"



#define ENABLEPEERACCESS 1

typedef int funcType;


int main(int argc, char* argv[]){

    std::ofstream output;

    if (argc == 2){
        output.open(argv[1]);
    } else if (argc > 2) {
        std::cout << "Usage filename\n";
        exit(1);
    } else {
        output.open("/dev/null");
    }
    
    
    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif

    int Device; 
    cudaGetDevice(&Device);
    int Devices;
    cudaGetDeviceCount(&Devices);
    cudaStream_t streams[Devices];

    for(int devID = 0; devID < Devices; devID++){
        cudaSetDevice(devID);
        cudaStreamCreate(&streams[devID]);
    }

    cudaError_t e;
    funcType* h_in;
    funcType* d_in[Devices];
    funcType* d_out[Devices];

    size_t dataSize = ARRAY_LENGTH*sizeof(funcType);

    h_in = (funcType*)malloc(dataSize);
    init_array_cpu< funcType >(h_in, 1337, ARRAY_LENGTH);

    int ArrayPerDevice = (ARRAY_LENGTH + Devices - 1) / Devices;

    for(int devID = 0; devID < Devices; devID++){
        int offset = ArrayPerDevice * devID;
        CUDA_RT_CALL(cudaMalloc(&d_in[devID], ArrayPerDevice*sizeof(funcType)));
        CUDA_RT_CALL(cudaMalloc(&d_out[devID], ArrayPerDevice*sizeof(funcType)));
        CUDA_RT_CALL(cudaMemcpy(d_in[devID], h_in + offset, ArrayPerDevice*sizeof(funcType), cudaMemcpyDefault));
    }
    
    

    for(int run = 0; run < ITERATIONS + 1; run++){
        cudaEvent_t start_event, stop_event;

        CUDA_RT_CALL(cudaEventCreate(&start_event));
        CUDA_RT_CALL(cudaEventCreate(&stop_event));

        CUDA_RT_CALL(cudaEventRecord(start_event));
        e = multiGPU::ApplyMapNonUnified< MapBasic<funcType> >(d_in, d_out, ARRAY_LENGTH);
        CUDA_RT_CALL(e);
        syncronize();
        CUDA_RT_CALL(cudaEventRecord(stop_event));
        CUDA_RT_CALL(cudaEventSynchronize(stop_event));

        float ms;
        CUDA_RT_CALL(cudaEventElapsedTime(&ms, start_event, stop_event));
        if(run != 0) output << ms << "\n";
    }

    free(h_in);
    for(int devID = 0; devID < Devices; devID++){
        cudaFree(d_in[devID]);
        cudaFree(d_out[devID]);    
    }

    
}