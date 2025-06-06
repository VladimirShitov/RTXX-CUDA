#!/bin/bash

# Parameters for all runs
ITER=10       # Number of iterations for each size
CHECK=0      # Don't check correctness for benchmarking

# Print header for the first run
./bin/ata-sp 2 2 $ITER $CHECK 1 0

# Run benchmark for the depth 1
for size in 4 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096 6144 8192 12188 16384; do
    ./bin/ata-sp $size $size $ITER $CHECK 1 1
done
