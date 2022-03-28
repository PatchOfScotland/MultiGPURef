#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "constants.cu.h"
#include "helpers.cu.h"
#include "scan.cu"


#define DEFAULT_N 5e8
#define DEFAULT_OUTPUTFILE "data/scan_bench.csv"

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
    const int64_t N = get_argval<int>(argv, argv + argc, "-x", DEFAULT_N);
    const std::string outputFile = get_argval<std::string>(argv, argv + argc, "-output", DEFAULT_OUTPUTFILE);
 
    std::ofstream File(outputFile);

    initHwd();
    EnablePeerAccess();

    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    funcType* data_in;
    funcType* data_out_single;
    funcType* data_out_multi_device;
    funcType* data_tmp;
    funcType* data_tmp_multi_device;

    cudaMallocManaged(&data_in, N*sizeof(funcType));
    cudaMallocManaged(&data_out_single, N*sizeof(funcType));
    cudaMallocManaged(&data_out_multi_device, N*sizeof(funcType));
    cudaMallocManaged(&data_tmp, MAX_BLOCK*sizeof(funcType));
    cudaMallocManaged(&data_tmp_multi_device, MAX_BLOCK*sizeof(funcType));

    funcType* device_data_in;
    funcType* device_data_out;
    funcType* aggregates;
    funcType* inc_prefix;
    uint8_t*  flags;
    
    cudaEvent_t syncEvent[DeviceCount];
    cudaEvent_t scan1blockEvent;

    cudaEventCreateWithFlags(&scan1blockEvent, cudaEventDisableTiming);

    for(int devID = 0; devID < DeviceCount; devID++){
      cudaSetDevice(devID);
      cudaEventCreateWithFlags(&syncEvent[devID], cudaEventDisableTiming);
    }


    cudaMalloc(&device_data_in, N*sizeof(funcType));
    cudaMalloc(&device_data_out, N*sizeof(funcType));
    AllocateFlagArray<Add<funcType> >(&flags, &aggregates, &inc_prefix, N);

    init_array_cpu<funcType>(data_in, 1337, N);
    cudaMemcpy(device_data_in, data_in, N*sizeof(funcType), cudaMemcpyDefault);

    DeviceSyncronize();    

    for(int run = 0; run < ITERATIONS + 1; run++){
        float ms_single, ms_2pass, ms_MD;
        
        cudaEvent_t start_single;
        cudaEvent_t stop_single;

        CUDA_RT_CALL(cudaEventCreate(&start_single));
        CUDA_RT_CALL(cudaEventCreate(&stop_single));

        CUDA_RT_CALL(cudaEventRecord(start_single));
        scanInc< Add < funcType > >(1024, N, data_out_single, data_in, data_tmp);
        CUDA_RT_CALL(cudaEventRecord(stop_single));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_single, start_single, stop_single));

        cudaEventDestroy(start_single);
        cudaEventDestroy(stop_single);

        cudaEvent_t start_2pass;
        cudaEvent_t stop_2pass;

        CUDA_RT_CALL(cudaEventCreate(&start_2pass));
        CUDA_RT_CALL(cudaEventCreate(&stop_2pass));

        CUDA_RT_CALL(cudaEventRecord(start_2pass));
        scanWrapper< Add <funcType> >(device_data_out, device_data_in, N, flags, aggregates, inc_prefix);
        CUDA_RT_CALL(cudaEventRecord(stop_2pass));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_2pass, start_2pass, stop_2pass));

        cudaEventDestroy(start_2pass);
        cudaEventDestroy(stop_2pass);

        cudaEvent_t start_MD;
        cudaEvent_t stop_MD;

        CUDA_RT_CALL(cudaEventCreate(&start_MD));
        CUDA_RT_CALL(cudaEventCreate(&stop_MD));

        CUDA_RT_CALL(cudaEventRecord(start_MD));
        scanInc_multiDevice< Add < funcType > >(1024, N, data_out_single, data_in, data_tmp, syncEvent, scan1blockEvent);
        CUDA_RT_CALL(cudaEventRecord(stop_MD));
        DeviceSyncronize();
        CUDA_RT_CALL(cudaEventElapsedTime(&ms_MD, start_MD, stop_MD));

        cudaEventDestroy(start_MD);
        cudaEventDestroy(stop_MD);

        if(run != 0) File << ms_single << ", " << ms_2pass << ", " << ms_MD << "\n";
    }

    File.close();

    cudaFree(data_in);
    cudaFree(data_out_single);
    cudaFree(data_out_multi_device);
    cudaFree(data_tmp);
    cudaFree(data_tmp_multi_device);
    cudaFree(device_data_in);
    cudaFree(aggregates);
    cudaFree(inc_prefix);
    cudaFree(flags);


    return 0;
}