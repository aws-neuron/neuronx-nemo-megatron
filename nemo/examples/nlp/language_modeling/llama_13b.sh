#!/usr/bin/env bash
export SEQ_LENGTH=4096
export HS=5120
export TP=8
export PP=4
export N_LAYERS=40
export N_AH=40
export FFN_HS=13824
export GBS=1024
export UBS=1
export TRAIN_ITERS=400000

export INIT_METHOD_STD=0.02
export LAYERNORM_EPSILON=1e-6
export WARMUP_STEPS=2000

# This helps to build the helpers.cpp required (only once)
# cd /usr/local/lib/python3.8/site-packages/nemo/collections/nlp/data/language_modeling/megatron/
# make
# cd /root/scripts/nemo
#

./test_llama.sh
