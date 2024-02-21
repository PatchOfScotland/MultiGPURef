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

typedef float funcType;

template<class T>
class MapPlusX {
    public:
        typedef T InputElement;
        typedef T ReturnElement;
        typedef T X;

        static __device__ __host__ ReturnElement apply(const InputElement i, const X x) {
            return i+x;
        };
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

    std::cout << "Running array of length " << N << " (" << ((N*2*sizeof(funcType))/1e9) <<"GB)\n";

    bool validating = true;
    if (N > 1e9) {
        std::cout << "Skipping output validations...\n";
        validating = false;
    }

    std::ofstream File(OutputFile);


    #if ENABLEPEERACCESS
    EnablePeerAccess();
    #endif

    int Device;
    cudaGetDevice(&Device);
    int Devices;
    cudaGetDeviceCount(&Devices);

    LogHardware();

    cudaStream_t* streams = (cudaStream_t*)calloc(num_streams*Devices, sizeof(cudaStream_t)) ;

    for(int devID = 0; devID < Devices; devID++){
        cudaSetDevice(devID);
        for(int streamID = devID * num_streams; streamID < (devID + 1) * num_streams; streamID++){
            cudaStreamCreate(&streams[streamID]);
        }
    }

    funcType* in;
    funcType* out;
    funcType* correct;
    
    if (validating) {
        correct = (funcType*)calloc(N, sizeof(funcType));
    }

    CUDA_RT_CALL(cudaMallocManaged(&in, N*sizeof(funcType)));
    CUDA_RT_CALL(cudaMallocManaged(&out, N*sizeof(funcType)));

    init_array_cpu< funcType >(in, 1337, N);
    if (validating) {
        for(int i = 0; i < N; i++){
            correct[i] = MapPlusX<funcType>::apply(in[i], 5);
        }
    }
    
    float* single_gpu_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_gpu_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_streams_ms = (float*)calloc(iterations, sizeof(float));
    float* multi_gpu_hinted_ms = (float*)calloc(iterations, sizeof(float));

    funcType* x;
    CUDA_RT_CALL(cudaMallocManaged(&x, sizeof(funcType)));
    *x = 5;

    { // Single GPU
        std::cout << "*** Benchmarking single GPU map ***\n";
        void* args[] = { &in, &out, &x, &N };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMap< MapPlusX < funcType > >;
        benchmarkFunction(function, args, single_gpu_ms, iterations, N, 1);
        if (validating) {
            if(compare_arrays(correct, out, N)){
                std::cout << "Single GPU map is correct\n";
            } else {
                std::cout << "Single GPU map is incorrect\n";
            }
        }
    }
    { // MultiGPU - No hints
        std::cout << "*** Benchmarking multi GPU map without hints ***\n";
        void* args[] = { &in, &out, &x, &N };
        cudaError_t (*function)(void**) = &multiGPU::ApplyMap< MapPlusX < funcType > >;
        benchmarkFunction(function, args, multi_gpu_ms, iterations, N, 1);
        if (validating) {
            if(compare_arrays(correct, out, N)){
                std::cout << "Multi GPU map is correct\n";
            } else {
                std::cout << "Multi GPU map is incorrect\n";
            }
        }
    }
    {   // MultiGPU - Streams - No hints
        std::cout << "*** Benchmarking multi GPU stream map without hints ***\n";
        void* args[] = {&in, &out, &x, &N, &streams, &num_streams};
        cudaError_t (*function)(void**) = &multiGPU::ApplyMapStreams<MapPlusX < funcType > >;
        benchmarkFunction(function, args, multi_streams_ms, iterations, N, 1);
        if (validating) {
            if(compare_arrays(correct, out, N)){
                std::cout << "Multi GPU Streams map is correct\n";
            } else {
                std::cout << "Multi GPU Streams map is incorrect\n";
            }
        }
    }
    hint1D<funcType>(in, 1024, N);
    hint1D<funcType>(out, 1024, N);
    { // MultiGPU
        std::cout << "*** Benchmarking multi GPU map with hints ***\n";
        void* args[] = { &in, &out, &x, &N };
        cudaError_t (*function)(void**) = &multiGPU::ApplyMap< MapPlusX < funcType > >;
        benchmarkFunction(function, args, multi_gpu_hinted_ms, iterations, N, 1);
        if (validating) {
            if(compare_arrays(correct, out, N)){
                std::cout << "Multi GPU map is correct\n";
            } else {
                std::cout << "Multi GPU map is incorrect\n";
            }
        }
    }

    for(int run = 0; run < iterations; run++){
        File << single_gpu_ms[run]
             << ", " << multi_gpu_ms[run]
             << ", " << multi_streams_ms[run]
             << ", " << multi_gpu_hinted_ms[run] << "\n";
    }
    File.close();

    if (validating) {
        free(correct);
    }
    cudaFree(in);
    cudaFree(out);
}