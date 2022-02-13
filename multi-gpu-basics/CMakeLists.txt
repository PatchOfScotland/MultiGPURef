cmake_minimum_required(VERSION 3.10)

enable_language(CUDA) 

project(multi-gpu-Primitives)

add_executable(map_test src/map_test.cu)