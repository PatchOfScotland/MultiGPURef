#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=3
#SBATCH --time=00:05:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:titanrtx:4

nvcc map.c -lcuda -lnvrtc -o map -O3 -std=c99 -arch=sm_61

./map >> data/mapDriver_4.csv

rm map

