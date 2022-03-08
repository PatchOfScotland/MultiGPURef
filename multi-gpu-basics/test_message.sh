#!/bin/bash
#SBATCH --job-name=MultiGPUMessage
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=20000m
#SBATCH -p gpu --gres=gpu:gtx1080:3

nvcc src/message.cu -o build/message -O3 --relocatable-device-code

./build/message
