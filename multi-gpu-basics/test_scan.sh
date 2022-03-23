#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:gtx1080:2
nvcc src/scan_verify.cu -o build/scan_verify -O3 -std=c++11

./build/scan_verify
