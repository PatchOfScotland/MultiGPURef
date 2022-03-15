#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=9
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:2

nvcc src/filters_stencil_bench.cu -o build/filters_stencil_bench -O3 -std=c++11 -arch=sm_61

./build/filters_stencil_bench data/filters_stencil_2.csv

echo "DONE!"
