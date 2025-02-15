#!/bin/bash
#SBATCH --job-name=MGPUGaussian
#SBATCH --ntasks=9
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanx:4

mkdir -p build data

nvcc src/filters_stencil_bench.cu -o build/filters_stencil_bench -O3 -std=c++11 -arch=sm_61

./build/filters_stencil_bench data/filters_stencil_2.csv

echo "DONE!"
