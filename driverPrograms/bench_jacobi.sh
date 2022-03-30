#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=3
#SBATCH --time=00:15:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:2

nvcc jacobiIteration.cu -lcuda -lnvrtc -o jacobiIteration -O3 -arch=sm_61

./jacobiIteration >> data/jacobiIteration_2.csv

rm jacobiIteration

