#!/bin/bash
#SBATCH --job-name=MGPUMmm
#SBATCH --ntasks=7
#SBATCH --time=01:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanx:4

mkdir -p build data

nvcc src/mmm_bench.cu -o build/mmm_bench -O3 -arch=sm_61

./build/mmm_bench data/mmm_1.csv

echo "DONE"
