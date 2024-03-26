#!/usr/bin/env bash

export SEQ_LENGTH=4096
export HS=8192
export TP=8
export PP=8
export N_LAYERS=80
export N_AH=64
export FFN_HS=28672
export GBS=512
export KV_HEADS=8
export TRAIN_ITERS=20000

export INIT_METHOD_STD=0.02
export LAYERNORM_EPSILON=1e-5
export WARMUP_STEPS=2000


./test_llama_gqa.sh