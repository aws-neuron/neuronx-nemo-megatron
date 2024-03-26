#!/usr/bin/env bash

export SEQ_LENGTH=8192
export HS=4096
export TP=8
export PP=1
export N_LAYERS=32
export N_AH=32
export FFN_HS=14336
export GBS=96
export KV_HEADS=8
: ${TRAIN_ITERS:=20000}

export INIT_METHOD_STD=0.02
export LAYERNORM_EPSILON=1e-5
export WARMUP_STEPS=10


./test_mistral.sh