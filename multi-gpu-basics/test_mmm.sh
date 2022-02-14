#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

./mmm_test GPU1080NUM3MMM.csv
