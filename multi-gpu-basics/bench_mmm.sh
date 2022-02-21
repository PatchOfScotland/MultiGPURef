#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

nvcc src/multiCoreTest.cu -o multiCoreTest -std=c++11 -O3
nvcc src/multiCoreTestCudaTiming.cu -o multiCoreTestCudaTiming -std=c++11 -O3

./multiCoreTest multiCoreBench.csv
./multiCoreTestCudaTiming multiCoreBenchCudaTiming.csv
