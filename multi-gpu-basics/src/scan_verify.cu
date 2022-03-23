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

#define DEFAULT_N 1e8
#define DEFAULT_OUTPUTFILE "data/SCAN_VERIFY.csv"

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
    int*  blockCounter; 

    initHwd();

    funcType* data_host_d = (funcType*)malloc(N*sizeof(funcType));
    funcType* data_host = (funcType*)malloc(N*sizeof(funcType));
    
    funcType* data_in;
    funcType* data_out;
    funcType* aggregates;
    funcType* inc_prefix;
    uint8_t*  flags;
    
    funcType* data_in_d;
    funcType* data_out_d;
    funcType* aggregates_d;
    funcType* inc_prefix_d;
    uint8_t*  flags_d;

    //funcType* data_simple_um_in;
    funcType* data_simple_um_out;
    funcType* data_simple_um_tmp;

    //cudaMallocManaged(&data_simple_um_in, N*sizeof(funcType));
    cudaMallocManaged(&data_simple_um_out, N*sizeof(funcType));
    cudaMallocManaged(&data_simple_um_tmp, MAX_BLOCK*sizeof(funcType));

    const int EmulatedDeviceCount = 3;

    funcType* data_emulated;
    funcType** emulatedAggregates  = (funcType**)malloc(sizeof(funcType*)*EmulatedDeviceCount);
    funcType** emulatedIncPrefixes = (funcType**)malloc(sizeof(funcType*)*EmulatedDeviceCount);
    uint8_t** emulatedFlags       = (uint8_t**)malloc(sizeof(funcType*)*EmulatedDeviceCount);  
    int**      emulatedDCounter    = (int**)malloc(sizeof(int*)*EmulatedDeviceCount);
    funcType*  emulatedGlobalAgg;

    funcType* data_simple_um_out_emulated;
    funcType* data_simple_um_tmp_emulated;

    funcType* data_simple_um_out_MD;
    funcType* data_simple_um_tmp_MD;

    cudaMallocManaged(&data_simple_um_out, N*sizeof(funcType));
    cudaMallocManaged(&data_simple_um_tmp, EmulatedDeviceCount*MAX_BLOCK*sizeof(funcType));

    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);

    cudaMallocManaged(&data_simple_um_out_MD, N*sizeof(funcType));
    cudaMallocManaged(&data_simple_um_tmp_MD, DeviceCount*MAX_BLOCK*sizeof(funcType));


    cudaMallocManaged(&data_simple_um_out_emulated, N*sizeof(funcType));
    cudaMallocManaged(&data_simple_um_tmp_emulated, EmulatedDeviceCount*MAX_BLOCK*sizeof(funcType));

    
    cudaMalloc(&data_in_d, N*sizeof(funcType));
    cudaMalloc(&data_out_d, N*sizeof(funcType));
    cudaMallocManaged(&blockCounter, sizeof(int)); 
    cudaMallocManaged(&data_in, N*sizeof(funcType)); 
    cudaMallocManaged(&data_out, N*sizeof(funcType));
    cudaMallocManaged(&data_emulated, N*sizeof(funcType));
    cudaMallocManaged(&emulatedGlobalAgg, sizeof(funcType)*EmulatedDeviceCount);
    AllocateFlagArray<Add<funcType> >(&flags, &aggregates, &inc_prefix, N);
    AllocateFlagArray<Add<funcType> >(&flags_d, &aggregates_d, &inc_prefix_d, N);

    AllocateFlagArrayMultiDevice<Add<funcType> >(emulatedFlags, emulatedAggregates, emulatedIncPrefixes, N, EmulatedDeviceCount);

    for(int devID = 0; devID < EmulatedDeviceCount; devID++){
        cudaMalloc(&emulatedDCounter[devID], sizeof(int));
    }

    init_array_cpu<funcType>(data_in, 1337, N);

    cudaMemcpy(data_in_d, data_in, sizeof(funcType)*N, cudaMemcpyDefault);
    //cudaMemcpy(data_simple_um_in, data_in, sizeof(funcType)*N, cudaMemcpyDefault);

    //CPU Scan
    int accum = 0;
    for(int64_t i = 0; i < N; i++){
        accum += data_in[i];
        data_host[i] = accum;
    }


    
    scanWrapper<Add < funcType > >(
        data_out, 
        data_in, 
        N, 
        flags, 
        aggregates, 
        inc_prefix
    ); 
    DeviceSyncronize();

    scanInc<Add < funcType > >(1024, N, data_simple_um_out, data_in, data_simple_um_tmp);
    DeviceSyncronize();



    scanWrapper<Add < funcType > >(
        data_out_d, 
        data_in_d, 
        N, 
        flags_d, 
        aggregates_d, 
        inc_prefix_d
    ); 

    DeviceSyncronize();
    cudaMemcpy(data_host_d, data_out_d, N*sizeof(funcType), cudaMemcpyDeviceToHost);

    DeviceSyncronize();
    scanInc_emulated<Add< funcType > >(1024, N, data_simple_um_out_emulated, data_in, data_simple_um_tmp_emulated, EmulatedDeviceCount);
    DeviceSyncronize();

    scanInc_multiDevice<Add< funcType > >(1024, N, data_simple_um_out_MD, data_in, data_simple_um_tmp_MD);



    if(compare_arrays<funcType>(data_host, data_out, N)){
        std::cout << "Managed 2-Pass SingleGPU and Host Agree\n";
    } else {
        std::cout << "Managed 2-Pass SingleGPU and Host Disagree\n";
    }
    if(compare_arrays<funcType>(data_host, data_host_d, N)){
        std::cout << "Device 2-Pass and Host Agree\n";
    } else {
        std::cout << "Device 2-Pass and Host Disagree\n";
    }
    if(compare_arrays<funcType>(data_host, data_simple_um_out, N)){
        std::cout << "Simple and Host agree\n";
    } else {
        std::cout << "Simple and Host disagree\n";
    }
    if(compare_arrays<funcType>(data_host, data_simple_um_out_emulated, N)){
        std::cout << "Emulated Simple and Host agree\n";
    } else {
        std::cout << "Emulated Simple and Host disagree\n";
    }
    if(compare_arrays<funcType>(data_host, data_simple_um_out_MD, N)){
        std::cout << "Multi Device Simple and Host agree\n";
    } else {
        std::cout << "Multi Device Simple and Host disagree\n";
    }


    cudaFree(data_in);
    cudaFree(data_out);

    free(data_host);
    free(emulatedAggregates);

    return 0;
}