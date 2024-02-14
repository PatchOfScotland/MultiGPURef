#!/bin/bash
#SBATCH --job-name=MGPUScan
#SBATCH --ntasks=9
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanx:2

mkdir -p build data

nvcc src/scan_bench.cu -o build/scan_bench -O3 -std=c++11 -arch=sm_61

./build/scan_bench data/scan_bench_2.csv

echo "DONE!"
