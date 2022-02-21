#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

nvcc src/multiCoreTestCudaTiming.cu -o build/multiCoreTestCudaTiming -std=c++11 -O3

./build/multiCoreTestCudaTiming data/multiCoreBenchCudaTiming.csv
