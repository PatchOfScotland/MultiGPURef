cmake_minimum_required(VERSION 3.10)

enable_language(CUDA) 

project(multi-gpu-Primitives)

add_executable(map_test src/map_test.cu)
add_executable(mmm_test src/mmm_test.cu)
add_executable(iota_test src/iota_test.cu)
add_executable(multiCoreTest src/multiCoreTest.cu)