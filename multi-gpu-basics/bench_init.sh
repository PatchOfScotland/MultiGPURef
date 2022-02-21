#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=8
#SBATCH --time=00:10:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

nvcc src/initial_test_advice.cu -o build/initial_test_advice  -O3
nvcc src/initial_test_prefetch.cu -o build/initial_test_prefetch  -O3
nvcc src/initial_test.cu -o build/initial_test -O3
nvcc src/initial_test_both.cu -o build/initial_test_both -O3

./build/initial_test_both data/initial_test_both.csv
./build/initial_test_prefetch data/initial_test_prefetch.csv
./build/initial_test_advice data/initial_test_advice.csv
./build/initial_test data/initial_test.csv

