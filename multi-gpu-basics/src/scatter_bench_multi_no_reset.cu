#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "scatter.cu"

#define ENABLEPEERACCESS 1

typedef int64_t funcType;

template<class T>
void reset(T* host, T* dev, size_t len){
    for(int i = 0; i < len; i++){
        host[i] = dev[i];
    }
}

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
    
    funcType* host_data = (funcType*)malloc(DATA_LENGTH*sizeof(funcType));
    funcType* data;
    funcType* idxs;
    funcType* data_idx;

    cudaError_t e;

    CUDA_RT_CALL(cudaMallocManaged(&data,     DATA_LENGTH * sizeof(funcType)));
    CUDA_RT_CALL(cudaMemcpy(host_data, data, DATA_LENGTH*sizeof(funcType), cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMallocManaged(&idxs,     IDX_LENGTH  * sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&data_idx, IDX_LENGTH  * sizeof(funcType)));
    
    init_array_cpu< funcType >(data, 1337, DATA_LENGTH);

    init_idxs(DATA_LENGTH, 420, idxs, IDX_LENGTH);
    init_array_cpu< funcType >(data_idx, 69, IDX_LENGTH);
    

    for(int run = 0; run < ITERATIONS + 1; run++){
        cudaEvent_t start_event, stop_event;

        CUDA_RT_CALL(cudaEventCreate(&start_event));
        CUDA_RT_CALL(cudaEventCreate(&stop_event));

        CUDA_RT_CALL(cudaEventRecord(start_event));
        e = multiGPU::scatterUM< funcType >(data, idxs, data_idx, DATA_LENGTH, IDX_LENGTH);
        CUDA_RT_CALL(e);
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventRecord(stop_event));
        CUDA_RT_CALL(cudaEventSynchronize(stop_event));

        float ms;
        CUDA_RT_CALL(cudaEventElapsedTime(&ms, start_event, stop_event));
        output << ms << "\n";

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    cudaFree(data);
    cudaFree(idxs);
    cudaFree(data_idx);
}