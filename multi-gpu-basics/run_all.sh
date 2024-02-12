#!/bin/bash

mkdir -p build data

declare -a benchmarks=(
    "bench_atomics.sh"
    #"bench_gaussian" does not work currently
    "bench_jacobi.sh"
    "bench_map.sh"
    "bench_mmm.sh"
    #"bench_oversubscription" very slow
    "bench_scan.sh"
    "bench_scatter.sh"
)

for i in "${benchmarks[@]}"
do
    echo "Running: $i"
   ./$i
done