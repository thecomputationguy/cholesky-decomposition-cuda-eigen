#!/bin/sh

printf "\nCleaning old files...\n"

rm *.out

printf "\nDone.\n"

printf "\nCompiling program...\n"

nvcc -std=c++17 -Xcompiler -fopenmp -I /usr/local/include/eigen3 solver.cu -lcusolver -o cholesky.out

printf "\nDone.\n"

export OMP_NUM_THREADS=6

./cholesky.out

printf "\nPlotting and saving graph.\n"

python3 plot.py

printf "\nDone.\n"
