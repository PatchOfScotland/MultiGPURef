#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:gtx10803
nvcc src/scatter_verify.cu -o build/scatter_verify -O3

./build/scatter_verify
