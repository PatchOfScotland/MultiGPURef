#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=7
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

nvcc src/mmm_bench_multi.cu -o build/mmm_bench_multi -std=c++11 -O3
nvcc src/mmm_bench_single.cu -o build/mmm_bench_single -std=c++11 -O3
nvcc src/mmm_bench_multi_prefetch.cu -o build/mmm_bench_prefetch -O3

./build/mmm_bench_multi data/mmm_bench_multi_3.csv
./build/mmm_bench_single data/mmm_bench_single.csv
./build/mmm_bench_prefetch data/mmm_bench_prefetch.csv

echo "DONE"
