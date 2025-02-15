#!/bin/bash
#SBATCH --job-name=MGPUInit
#SBATCH --ntasks=15
#SBATCH --time=00:20:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanx:4

mkdir -p build data

nvcc src/initial_test_advice.cu -o build/initial_test_advice  -O3 -std=c++11
nvcc src/initial_test_prefetch.cu -o build/initial_test_prefetch  -O3 -std=c++11
nvcc src/initial_test.cu -o build/initial_test -O3 -std=c++11
nvcc src/initial_test_both.cu -o build/initial_test_both -O3 -std=c++11
nvcc src/initial_test_standard.cu -o build/initial_test_standard -O3 -std=c++11
nvcc src/initial_test_multi.cu -o build/initial_test_multi -O3 -std=c++11
nvcc src/initial_test_cuRand.cu -o build/initial_test_cuRand -O3 -std=c++11

./build/initial_test_both data/initial_test_both.csv
./build/initial_test_prefetch data/initial_test_prefetch.csv
./build/initial_test_advice data/initial_test_advice.csv
./build/initial_test data/initial_test.csv
./build/initial_test_standard data/initial_test_standard.csv
./build/initial_test_multi data/initial_test_multi.csv
./build/initial_test_cuRand data/initial_test_cuRand.csv

echo "DONE!"
