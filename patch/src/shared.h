#ifndef SHARED_H
#define SHARED_H

#include <thread>

const auto processor_count = std::thread::hardware_concurrency();

void init_array(float* arr, size_t n) {
    srand(5454);
    for(int i=0; i<n; i++){
        arr[i] = (float)rand() / RAND_MAX;
    }
}

void benchmarkFunction(float (*function)(float, float), float* input_array, const float constant, float* output_array, float array_len) {
    
}

#endif