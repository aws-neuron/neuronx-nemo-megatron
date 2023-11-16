#!/usr/bin/env bash

# model architecture
export SEQ_LENGTH=8192
export HS=4096
export N_LAYERS=42
export N_AH=32
export FFN_HS=16384
export TP=8
export PP=2

# hyperparameters
export FIM_RATE=0.5
export FIM_SPM_RATE=0
export TRAIN_ITERS=2000
export GBS=$((256*SLURM_NTASKS))

./test_bigcode.sh
