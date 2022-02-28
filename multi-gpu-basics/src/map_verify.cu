#include <iostream>
#include <fstream>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "map.cu"

#define LOGGING 0
#define ENABLEPEERACCESS 1

typedef float funcType;

int main(int argc, char* argv[]){
    #if LOGGING
    LogHardware("HWINFO.log");
    #endif

    #if ENABLEPEERACCESS
        EnablePeerAccess();
    #endif

    int Device;
    int Devices;
    cudaGetDevice(&Device);
    cudaGetDeviceCount(&Devices);

    size_t N = 1e6;
    size_t data_size = N * sizeof(float);

    cudaStream_t streams[Devices];

    for(int devID = 0; devID < Devices; devID++){
        cudaSetDevice(devID);
        cudaStreamCreate(&streams[devID]);
    }
    cudaSetDevice(Device);

    funcType* d_in; 
    funcType* d_multi;
    funcType* d_single;
    funcType* d_prefetch;
    funcType* d_streams;
    funcType* d_in_noUnified[Devices];
    funcType* d_out_noUnified[Devices];

    CUDA_RT_CALL(cudaMallocManaged((void**)&d_in, data_size));
    CUDA_RT_CALL(cudaMallocManaged((void**)&d_single, data_size));
    CUDA_RT_CALL(cudaMallocManaged((void**)&d_multi, data_size));
    CUDA_RT_CALL(cudaMallocManaged((void**)&d_prefetch, data_size));
    CUDA_RT_CALL(cudaMallocManaged((void**)&d_streams, data_size));

    init_array_cpu< funcType >(d_in, 1337, N);

    size_t NpD = (N + Devices - 1) / Devices;
    for(int devID = 0; devID < Devices; devID++){
        cudaMalloc(&d_in_noUnified[devID], NpD);
        cudaMalloc(&d_out_noUnified[devID], NpD);
        cudaMemcpy(d_in_noUnified[devID] , d_in + NpD*devID, NpD*sizeof(funcType), cudaMemcpyDefault);
    }





    singleGPU::ApplyMap< MapBasic<funcType> >(d_in, d_single, N);
    multiGPU::ApplyMap< MapBasic<funcType> >(d_in, d_multi, N);
    multiGPU::ApplyMapPrefetchAdvice< MapBasic<funcType > >(d_in, d_prefetch, N);
    multiGPU::ApplyMapStreams< MapBasic <funcType >>(d_in, d_streams, N, streams);
    multiGPU::ApplyMapNonUnified< MapBasic< funcType >>(d_in_noUnified, d_in_noUnified, N);
    cudaDeviceSynchronize();

    std::string multiRes = (compare_arrays_nummeric<funcType>(d_single, d_multi, N)) ? "MULTI-GPU VALID RESULTS!\n" : "MULTI-GPU INVALID RESULTS!\n" ;
    std::string prefetchRes = (compare_arrays_nummeric<funcType>(d_single, d_prefetch, N)) ? "PREFETCH-GPU VALID RESULTS!\n" : "PREFETCH-GPU INVALID RESULTS!\n" ; ;
    std::string streamsRes = (compare_arrays_nummeric<funcType>(d_single, d_streams, N)) ? "STREAMS-GPU VALID RESULTS!\n" : "STREAMS-GPU INVALID RESULTS!\n" ;;
    //std::string noUnified = (compare_arrays_nummeric<funcType>(d_single, d_multi, N)) ? "NoUnified-GPU VALID RESULTS!\n" : "NoUnified-GPU  INVALID RESULTS!\n" ;;

    std::cout << multiRes ;
    std::cout << prefetchRes ;
    std::cout << streamsRes ;
    //std::cout <<


    for(int devID = 0; devID < Devices; devID++){
        cudaStreamDestroy(streams[devID]);
        cudaFree(d_in_noUnified[devID]);
        cudaFree(d_out_noUnified[devID]);
    }

    cudaFree(d_in);
    cudaFree(d_multi);
    cudaFree(d_single);
    cudaFree(d_prefetch);    
    cudaFree(d_streams);



    return 0;
}

