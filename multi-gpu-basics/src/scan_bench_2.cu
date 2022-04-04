#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "scan.cu"
#include "scan_voidArgs.cu"


#define DEFAULT_N 1e9

typedef int funcType;

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    int64_t N = get_argval<int>(argv, argv + argc, "-x", DEFAULT_N);
    const std::string outputFile = get_argval<std::string>(argv, argv + argc, "-output", "data/scan_bench.csv");
    const size_t bufferSize = N*sizeof(int);
    int pageSize = sysconf(_SC_PAGESIZE);

    std::ofstream File(outputFile);

    initHwd();
    EnablePeerAccess();

    int Device; 
    cudaGetDevice(&Device);
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    funcType* data_in;
    funcType* data_out;
    funcType* data_tmp;

    CUDA_RT_CALL(cudaMallocManaged(&data_in, bufferSize));
    CUDA_RT_CALL(cudaMallocManaged(&data_out, bufferSize));
    CUDA_RT_CALL(cudaMallocManaged(&data_tmp, pageSize*DeviceCount));
    
    cudaEvent_t* syncEvent = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)* DeviceCount);
    cudaEvent_t scan1blockEvent;

    cudaEventCreateWithFlags(&scan1blockEvent, cudaEventDisableTiming);

    for(int devID = 0; devID < DeviceCount; devID++){
      cudaSetDevice(devID);
      cudaEventCreateWithFlags(&syncEvent[devID], cudaEventDisableTiming);
    }
    cudaSetDevice(Device);

    init_array_cpu<funcType>(data_in, 1337, N);
    DeviceSyncronize();    

    funcType* correctData = (funcType*)malloc(N*sizeof(funcType));

    funcType accum = 0;
    for(long i = 0; i < N; i++){
        accum += data_in[i];
        correctData[i] = accum;
    }

    float* scan_single_ms = (float*)malloc(sizeof(float)*ITERATIONS);
    float* scan_MD_NoPS = (float*)malloc(sizeof(float)*ITERATIONS);
    float* scan_MD_PS = (float*)malloc(sizeof(float)*ITERATIONS);

    { //Single Core
        std::cout << "Starting Single\n";
        unsigned int blockSize = 1024;
        void *args[] = {&blockSize, &N, &data_out, &data_in, &data_tmp};
        cudaError_t (*function)(void**) = &scanIncVoidArgs<Add <funcType> >;
        benchmarkFunction(function, args,scan_single_ms,ITERATIONS);
    }

    { //Multi Core
        std::cout << "Starting MultiCore\n";
        void *args[] = {&N, &data_out, &data_in, &data_tmp, &syncEvent, &scan1blockEvent};
        cudaError_t (*function)(void**) = &scanIncVoidArgsMD<Add <funcType>>;
        benchmarkFunction(function, args,scan_MD_NoPS, ITERATIONS);
    }
    
    { //Multi Core Page
        std::cout << "Starting Paged Multicore\n";
        void *args[] = {&N, &data_out, &data_in, &data_tmp, &syncEvent, &scan1blockEvent, &pageSize};
        cudaError_t (*function)(void**) = &scanIncVoidArgsMDPS<Add <funcType>>;
        benchmarkFunction(function, args,scan_MD_PS, ITERATIONS);
    }
   
    for(int run = 0; run < ITERATIONS; run++){
        File << scan_single_ms[run] << ", " << scan_MD_NoPS[run] << ", " << scan_MD_PS[run] << "\n";
    }


    File.close();

    CUDA_RT_CALL(cudaFree(data_in));
    CUDA_RT_CALL(cudaFree(data_tmp));
    CUDA_RT_CALL(cudaFree(data_out));


    return 0;
}