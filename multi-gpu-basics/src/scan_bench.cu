#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/scan.cu"


#define DEFAULT_N 1e9

typedef int funcType;

template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    static __device__ __host__ inline T mapFun(const T& el)           { return el;      }
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
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

int main(int argc, char* argv[]) {
    int64_t N = get_argval<int64_t>(argv, argv + argc, "-x", DEFAULT_N);
    size_t iterations = get_argval<size_t>(argv, argv + argc, "-iter", ITERATIONS);
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

    funcType* correctData = (funcType*)calloc(N,sizeof(funcType));

    funcType accum = 0;
    for(long i = 0; i < N; i++){
        accum += data_in[i];
        correctData[i] = accum;
    }

    float* scan_single_ms = (float*)calloc(iterations,sizeof(float));
    float* scan_MD_NoPS = (float*)calloc(iterations,sizeof(float));
    float* scan_MD_PS = (float*)calloc(iterations,sizeof(float));

    { //Single Core
        std::cout << "*** Benchmarking single GPU scan ***\n";
        unsigned int blockSize = 1024;
        void *args[] = {&blockSize, &N, &data_out, &data_in, &data_tmp};
        cudaError_t (*function)(void**) = &singleGPU::scanInc<Add <funcType> >;
        benchmarkFunction(function, args,scan_single_ms, iterations, 1, 1);
        if (compare_arrays<funcType>(data_out, correctData, N)){
            std::cout << "Single GPU scan is valid\n";
        } else {
            std::cout << "Single GPU scan is invalid\n";
        }
    }

    { //Multi GPU
        std::cout << "*** Benchmarking multi GPU scan ***\n";
        void *args[] = {&N, &data_out, &data_in, &data_tmp, &syncEvent, &scan1blockEvent};
        cudaError_t (*function)(void**) = &multiGPU::scanIncVoidArgsMD<Add <funcType>>;
        benchmarkFunction(function, args,scan_MD_NoPS, ITERATIONS, 1, 1);
        if (compare_arrays<funcType>(data_out, correctData, N)){
            std::cout << "Multi GPU scan is valid\n";
        } else {
            std::cout << "Multi GPU scan is invalid\n";
        }
    }

    { //Multi GPU Page
        std::cout << "*** Benchmarking multi GPU scan with page sizing ***\n";
        void *args[] = {&N, &data_out, &data_in, &data_tmp, &syncEvent, &scan1blockEvent, &pageSize};
        cudaError_t (*function)(void**) = &multiGPU::scanIncVoidArgsMDPS<Add <funcType>>;
        benchmarkFunction(function, args,scan_MD_PS, ITERATIONS, 1, 1);
        if (compare_arrays<funcType>(data_out, correctData, N)){
            std::cout << "Multi GPU scan with page size is valid\n";
        } else {
            std::cout << "Multi GPU scan With page size is invalid\n";
        }
    }

    for(int run = 0; run < iterations; run++){
        File << scan_single_ms[run] << ", " << scan_MD_NoPS[run] << ", " << scan_MD_PS[run] << "\n";
    }

    File.close();

    CUDA_RT_CALL(cudaFree(data_in));
    CUDA_RT_CALL(cudaFree(data_tmp));
    CUDA_RT_CALL(cudaFree(data_out));


    return 0;
}