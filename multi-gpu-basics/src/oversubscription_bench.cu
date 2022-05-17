

#include "lib/constants.cu.h"
#include "lib/helpers.cu.h"
#include "lib/map.cu"



typedef int funcType;

int main(){
    // Run times definitions
    float* runtimes_map_95 = (float*)malloc(sizeof(float)*ITERATIONS);
    float* runtimes_map_100 = (float*)malloc(sizeof(float)*ITERATIONS);
    float* runtimes_map_105 = (float*)malloc(sizeof(float)*ITERATIONS);

    float* runtimes_map_95_hinted = (float*)malloc(sizeof(float)*ITERATIONS);
    float* runtimes_map_100_hinted = (float*)malloc(sizeof(float)*ITERATIONS);
    float* runtimes_map_105_hinted = (float*)malloc(sizeof(float)*ITERATIONS);

    initHwd();
    size_t freeMem, totalMem;

    int Device;
    cudaGetDevice(&Device);

    cudaMemGetInfo(&freeMem, &totalMem);


    size_t capacity = freeMem / sizeof(funcType);

    // Map

    { // over Subscription: 0.95
        size_t arrSize = capacity / 2 * 0.95;
        size_t bufferSize = arrSize*sizeof(funcType);

        funcType* inputMem;
        funcType* outputMem;
        CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
        CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));

        init_array_cpu<funcType>(inputMem, 1337, arrSize);

        CUDA_RT_CALL(cudaGetLastError());

        void* args[] = { &inputMem, &outputMem, &arrSize };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMapVoidArgs< MapP2 <funcType>>;

        benchmarkFunction(function, args, runtimes_map_95, ITERATIONS);

        CUDA_RT_CALL(cudaFree(inputMem));
        CUDA_RT_CALL(cudaFree(outputMem));
    }

    { // over Subscription: 1
        size_t arrSize = capacity / 2;
        size_t bufferSize = arrSize*sizeof(funcType);

        funcType* inputMem;
        funcType* outputMem;
        CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
        CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));

        init_array_cpu<funcType>(inputMem, 1337, arrSize);

        CUDA_RT_CALL(cudaGetLastError());

        void* args[] = { &inputMem, &outputMem, &arrSize };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMapVoidArgs< MapP2 <funcType>>;

        benchmarkFunction(function, args, runtimes_map_100, ITERATIONS);

        CUDA_RT_CALL(cudaFree(inputMem));
        CUDA_RT_CALL(cudaFree(outputMem));
    }

    { // over Subscription: 1.05
        size_t arrSize = capacity / 2 * 1.05;
        size_t bufferSize = arrSize*sizeof(funcType);

        funcType* inputMem;
        funcType* outputMem;
        CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
        CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));

        init_array_cpu<funcType>(inputMem, 1337, arrSize);

        CUDA_RT_CALL(cudaGetLastError());

        void* args[] = { &inputMem, &outputMem, &arrSize };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMapVoidArgs< MapP2 <funcType>>;

        benchmarkFunction(function, args, runtimes_map_105, ITERATIONS);

        CUDA_RT_CALL(cudaFree(inputMem));
        CUDA_RT_CALL(cudaFree(outputMem));
    }

    // Map

    { // over Subscription Hinted: 0.95
        size_t arrSize = capacity / 2 * 0.95;
        size_t bufferSize = arrSize*sizeof(funcType);

        funcType* inputMem;
        funcType* outputMem;
        CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
        CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));



        init_array_cpu<funcType>(inputMem, 1337, arrSize);
        CUDA_RT_CALL(cudaMemAdvise(inputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemAdvise(outputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemPrefetchAsync(inputMem, bufferSize, Device));
        CUDA_RT_CALL(cudaDeviceSynchronize());


        void* args[] = { &inputMem, &outputMem, &arrSize };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMapVoidArgs< MapP2 <funcType>>;

        benchmarkFunction(function, args, runtimes_map_95_hinted, ITERATIONS);

        CUDA_RT_CALL(cudaFree(inputMem));
        CUDA_RT_CALL(cudaFree(outputMem));
    }

    { // over Subscription Hinted: 1
        size_t arrSize = capacity / 2;
        size_t bufferSize = arrSize*sizeof(funcType);

        funcType* inputMem;
        funcType* outputMem;
        CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
        CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));

        init_array_cpu<funcType>(inputMem, 1337, arrSize);
        CUDA_RT_CALL(cudaMemAdvise(inputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemAdvise(outputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemPrefetchAsync(inputMem, bufferSize, Device));
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaGetLastError());

        void* args[] = { &inputMem, &outputMem, &arrSize };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMapVoidArgs< MapP2 <funcType>>;

        benchmarkFunction(function, args, runtimes_map_100_hinted, ITERATIONS);

        CUDA_RT_CALL(cudaFree(inputMem));
        CUDA_RT_CALL(cudaFree(outputMem));
    }

    { // over Subscription hinted: 1.05
        size_t arrSize = capacity / 2 * 1.05;
        size_t bufferSize = arrSize*sizeof(funcType);

        funcType* inputMem;
        funcType* outputMem;
        CUDA_RT_CALL(cudaMallocManaged(&inputMem, bufferSize));
        CUDA_RT_CALL(cudaMallocManaged(&outputMem, bufferSize));

        init_array_cpu<funcType>(inputMem, 1337, arrSize);
        CUDA_RT_CALL(cudaMemAdvise(inputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemAdvise(outputMem, bufferSize, cudaMemAdviseSetPreferredLocation, Device));
        CUDA_RT_CALL(cudaMemPrefetchAsync(inputMem, bufferSize, Device));
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaGetLastError());

        void* args[] = { &inputMem, &outputMem, &arrSize };
        cudaError_t (*function)(void**) = &singleGPU::ApplyMapVoidArgs< MapP2 <funcType>>;

        benchmarkFunction(function, args, runtimes_map_105_hinted, ITERATIONS);

        CUDA_RT_CALL(cudaFree(inputMem));
        CUDA_RT_CALL(cudaFree(outputMem));
    }

    for(int i = 0; i < ITERATIONS; i++){
            std::cout << runtimes_map_95[i] << ", " << runtimes_map_100[i] << ", " << runtimes_map_105[i] << ", ";
            std::cout << runtimes_map_95_hinted[i] << ", " << runtimes_map_100_hinted[i] << ", " << runtimes_map_105_hinted[i] << "\n";
        }


    //Freeing Stuff


}