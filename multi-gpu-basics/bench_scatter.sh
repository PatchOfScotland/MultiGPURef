#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=9
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanrtx:4

nvcc src/scatter_bench.cu -o build/scatter_bench -O3 -std=c++11 -arch=sm_61

./build/scatter_bench -output data/scatter_bench_4.csv

echo "DONE!"
