#!/bin/bash

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 2048 --f 65 --T 128

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 4096 --f 65 --T 128

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 8192 --f 31 --T 64

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
  ./conv_compare --N 1024 --f 33 --T 64

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 8192 --f 35 --T 64

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 16384 --f 31 --T 64

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 16384 --f 33 --T 64

# srun -p thunder4 --exclusive --gres=gpu:1 --nodelist=v07 \
#   ./conv_compare --N 16384 --f 35 --T 64


