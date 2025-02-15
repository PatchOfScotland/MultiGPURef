#!/bin/bash
#SBATCH --job-name=MGPUMap
#SBATCH --ntasks=3
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanx:2

mkdir -p build data

nvcc src/map_bench.cu -o build/map_bench -O3

./build/map_bench data/map_bench.csv -output data/map_bench_2.csv -n 1000000000
echo "DONE"
