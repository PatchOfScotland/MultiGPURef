#!/bin/bash
#SBATCH --job-name=MultiGPUMap
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --mem=10000m
#SBATCH -p gpu --gres=gpu:gtx1080:2

./map_test GPU1080NUM3.csv
