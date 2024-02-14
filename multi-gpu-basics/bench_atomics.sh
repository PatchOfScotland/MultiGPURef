#!/bin/bash
#SBATCH --job-name=MGPUAtomics
#SBATCH --ntasks=3
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:titanx:4

mkdir -p build data

nvcc src/atomic_bench.cu -o build/atomic_bench -O3 -std=c++11 -arch=sm_61

./build/atomic_bench data/atomic_results.csv

echo "DONE!"
