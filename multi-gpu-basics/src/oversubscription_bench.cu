
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
#include "lib/map.cu"



typedef int funcType;

void BenchMapOverSubscriptionCPU(int iterations, float overSubscriptionFactor, cudaError_t (*function)(void**), bool hint){
    size_t freeMem, totalMem;
    CUDA_RT_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    size_t capacity = freeMem / sizeof(funcType);
    size_t arrSize = capacity / 2 * overSubscriptionFactor;
    size_t bufferSize = arrSize * sizeof(funcType);

    int Device;
    CUDA_RT_CALL(cudaGetDevice(&Device));

    funcType* inputMem;
    funcType* outputMem;

    CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
    CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));

    if(hint){
        CUDA_RT_CALL(cudaMemAdvise(inputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemAdvise(outputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
    }
    init_array_cpu<funcType>(inputMem, 1337, arrSize);

    void* args[] = {&inputMem, &outputMem, &arrSize};

    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    CUDA_RT_CALL(cudaEventCreate(&start_event));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync));

    for(int run = 0; run < iterations; run++){
        float runtime;

        CUDA_RT_CALL(cudaMemPrefetchAsync(outputMem, bufferSize, 0));
        CUDA_RT_CALL(cudaMemPrefetchAsync(inputMem, bufferSize, 0));

        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaEventRecord(start_event));
        function(args);
        CUDA_RT_CALL(cudaEventRecord(stop_event));
        CUDA_RT_CALL(cudaEventSynchronize(stop_event));
        CUDA_RT_CALL(cudaEventElapsedTime(&runtime, start_event, stop_event));

        std::cout << runtime;
        (run != iterations -1) ? std::cout << ", " : std::cout << "\n";
    }


    CUDA_RT_CALL(cudaEventDestroy(start_event));
    CUDA_RT_CALL(cudaEventDestroy(stop_event));

    CUDA_RT_CALL(cudaFree(inputMem));
    CUDA_RT_CALL(cudaFree(outputMem));
}


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
    int iterations = get_argval<int>(argv, argv + argc, "-iter", ITERATIONS);

    float overSubFactors[] = {0.95, 1.0, 1.05, 1.2, 1.5};
    int factors = 5;
    // No Hint - Random Order
    for(int run = 0; run < factors; run++){
        BenchMapOverSubscriptionCPU(iterations, overSubFactors[run], &singleGPU::ApplyMap<MapP2< funcType > >, false);
    }
    // No Hints - fixed Order
    for(int run = 0; run < factors; run++){
        BenchMapOverSubscriptionCPU(iterations, overSubFactors[run], &singleGPU::ApplyMapChunks<MapP2< funcType > >, false);
    }

    // Hint - Random Order
    for(int run = 0; run < factors; run++){
        BenchMapOverSubscriptionCPU(iterations, overSubFactors[run], &singleGPU::ApplyMap<MapP2< funcType > >, true);
    }
    // Hints - fixed Order
    for(int run = 0; run < factors; run++){
        BenchMapOverSubscriptionCPU(iterations, overSubFactors[run], &singleGPU::ApplyMapChunks<MapP2< funcType > >, true);
    }
    return 0;
}