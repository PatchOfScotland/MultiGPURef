#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=9
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

nvcc src/scatter_bench_single.cu -o build/scatter_bench_single -O3
nvcc src/scatter_bench_multi.cu -o build/scatter_bench_multi -O3
nvcc src/scatter_bench_single_no_reset.cu -o build/scatter_bench_single_no_reset -O3
nvcc src/scatter_bench_multi_no_reset.cu -o build/scatter_bench_multi_no_reset -O3

./build/scatter_bench_single data/scatter_bench_single.csv
./build/scatter_bench_multi data/scatter_bench_multi.csv
./build/scatter_bench_single_no_reset data/build/scatter_bench_single_no_reset.csv
./build/scatter_bench_multi_no_reset data/build/scatter_bench_multi_no_reset.csv

echo "DONE!"
