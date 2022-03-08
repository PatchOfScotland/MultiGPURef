#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:gtx10803
nvcc src/stencil_verify.cu -o build/stencil_verify -O3 -std=c++11

./build/stencil_verify
