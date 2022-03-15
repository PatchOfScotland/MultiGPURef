#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:gtx1080:3
nvcc src/filters_stencil_verify.cu -o build/filters_stencil_verify -O3 -std=c++11 -arch=sm_61

./build/filters_stencil_verify
