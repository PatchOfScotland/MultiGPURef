#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=3
#SBATCH --time=00:05:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:2


nvcc mapp1.c -lcuda -lcuda --x c -o mapp1
cat data/i32_10.dat | ./mapp1 >> res.dat



