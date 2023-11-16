#!/usr/bin/env bash

# model architecture
export SEQ_LENGTH=8192
export HS=6144
export N_LAYERS=40
export N_AH=48
export FFN_HS=24576
export TP=8
export PP=4

# hyperparameters
export FIM_RATE=0.5
export FIM_SPM_RATE=0
export TRAIN_ITERS=2000
export GBS=$((256*SLURM_NTASKS))

./test_bigcode.sh
