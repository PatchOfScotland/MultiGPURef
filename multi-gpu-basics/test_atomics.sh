#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=5
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:gtx1080:3
nvcc src/atomic_verify.cu -o build/atomic_verify -O3 -std=c++11 -arch=sm_61
nvcc src/atomic_bench.cu -o build/atomic_bench -O3 -std=c++11 -arch=sm_61

./build/atomic_verify
./build/atomic_bench data/atomic_results.csv

echo "DONE!"