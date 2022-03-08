#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=11
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:2

nvcc src/map_bench_nonUnified.cu  -o build/map_bench_nonUnified -O3 -std=c++11
nvcc src/map_bench_prefetch.cu -o build/map_bench_prefetch -O3 -std=c++11
nvcc src/map_bench_streams.cu -o build/map_bench_streams -O3 -std=c++11
nvcc src/map_bench.cu -o build/map_bench -O3 -std=c++11
nvcc src/map_bench_single.cu -o build/map_bench_single -O3 -std=c++11

./build/map_bench_nonUnified data/map_bench_nonUnified.csv
./build/map_bench_prefetch data/map_bench_prefetch.csv
./build/map_bench_streams data/map_bench_streams.csv
./build/map_bench data/map_bench.csv
./build/map_bench_single data/map_bench_single.csv 
echo "DONE"
