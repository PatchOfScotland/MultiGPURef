#!/bin/bash

cmake -B build/

cd build
make
echo "Build all \n"
cd ..
./build/map_test 
