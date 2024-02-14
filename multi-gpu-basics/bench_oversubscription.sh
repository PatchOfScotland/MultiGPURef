#!/bin/bash
#SBATCH --job-name=MGPUOversubscription
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=32000m
#SBATCH -p gpu --gres=gpu:titanrtx:1

mkdir -p build data

nvcc src/oversubscription_bench.cu -o build/oversubscription_bench -O3 -std=c++11

./build/oversubscription_bench data/oversubscription_bench.csv
echo "DONE"
