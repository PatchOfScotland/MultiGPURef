#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/map.cu"
#include "lib/MemoryManagement.cu"

#define ENABLEPEERACCESS 1
#define ARRAY_LENGTH 1e9
#define OUTPUT_FILE_PATH "data/map.csv"
#define STREAMS 3

typedef int funcType;

template<class T>
class MapP2 {
    public:
        typedef T InpElTp;
        typedef T RedElTp;

        static __device__ __host__ RedElTp apply(const InpElTp i) {return i+2;};
};

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


int main(int argc, char** argv){
    int64_t N = get_argval<int64_t>(argv, argv + argc, "-n", ARRAY_LENGTH);
    int num_streams = get_argval<int>(argv, argv + argc, "-s", STREAMS);
    int iterations = get_argval<int>(argv, argv + argc, "-iter", ITERATIONS);
    std::string OutputFile = get_argval<std::string>(argv, argv + argc, "-output", OUTPUT_FILE_PATH);


    std::ofstream File(OutputFile);


    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif

    int Device;
    cudaGetDevice(&Device);
    int Devices;
    cudaGetDeviceCount(&Devices);
    cudaStream_t* streams = (cudaStream_t*)calloc(num_streams*Devices, sizeof(cudaStream_t)) ;

    for(int devID = 0; devID < Devices; devID++){
        cudaSetDevice(devID);
        for(int streamID = devID * num_streams; streamID < (devID + 1) * num_streams; streamID++){
            cudaStreamCreate(&streams[streamID]);
        }
    }

    funcType* in;
    funcType* out;
    funcType* correct = (funcType*)calloc(N, sizeof(funcType));

    CUDA_RT_CALL(cudaMallocManaged(&in, N*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&out, N*sizeof(funcType)));

    init_array_cpu< funcType >(in, 1337, N);
    for(int i = 0; i < N; i++){
        correct[i] = MapP2<funcType>::apply(in[i]);
    }

    float* single_gpu_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_gpu_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_streams_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_gpu_hinted_ms = (float*)calloc(iterations, sizeof(float));

    { // Single GPU
        void* args[] = { &in, &out, &N };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMap< MapP2 < funcType > >;
        benchmarkFunction(function, args, single_gpu_ms, iterations);
        if(compare_arrays(correct, out, N)){
            std::cout << "Single GPU map is correct\n";
        } else {
            std::cout << "Single GPU map is incorrect\n";
        }
    }
    { // MultiGPU - No hints
        void* args[] = { &in, &out, &N };
        cudaError_t (*function)(void**) = &multiGPU::ApplyMap< MapP2 < funcType > >;
        benchmarkFunction(function, args, multi_gpu_ms, iterations);
        if(compare_arrays(correct, out, N)){
            std::cout << "Multi GPU map is correct\n";
        } else {
            std::cout << "Multi GPU map is incorrect\n";
        }
    }
    {   // MultiGPU - Streams - No hints
        void* args[] = {&in, &out, &N, &streams, &num_streams};
        cudaError_t (*function)(void**) = &multiGPU::ApplyMapStreams<MapP2 < funcType > >;
        benchmarkFunction(function, args, multi_streams_ms, iterations);
        if(compare_arrays(correct, out, N)){
            std::cout << "Multi GPU Streams map is correct\n";
        } else {
            std::cout << "Multi GPU Streams map is incorrect\n";
        }
    }
    hint1D<funcType>(in, 1024, N);
    hint1D<funcType>(out, 1024, N);

    { // MultiGPU
        void* args[] = { &in, &out, &N };
        cudaError_t (*function)(void**) = &multiGPU::ApplyMap< MapP2 < funcType > >;
        benchmarkFunction(function, args, multi_gpu_hinted_ms, iterations);
        if(compare_arrays(correct, out, N)){
            std::cout << "Multi GPU map is correct\n";
        } else {
            std::cout << "Multi GPU map is incorrect\n";
        }
    }


    for(int run = 0; run < iterations; run++){
        File << single_gpu_ms[run]
             << ", " << multi_gpu_ms[run]
             << ", " << multi_streams_ms[run]
             << ", " << multi_gpu_hinted_ms[run] << "\n";
    }
    File.close();


    free(correct);
    cudaFree(in);
    cudaFree(out);
}