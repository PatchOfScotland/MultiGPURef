#!/bin/bash

mkdir -p build data slurm
rm slurm/*

declare -a benchmarks=(
    "bench_atomics.sh"
    "bench_gaussian.sh"
    "bench_jacobi.sh"
    "bench_map.sh"
    "bench_mmm.sh"
    "bench_oversubscription.sh"
    "bench_scan.sh"
    "bench_scatter.sh"
)

for i in "${benchmarks[@]}"
do
    sbatch -o slurm/%x_%A.out $i
done