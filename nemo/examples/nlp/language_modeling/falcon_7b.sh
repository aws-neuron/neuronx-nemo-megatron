#!/usr/bin/env bash

export SEQ_LENGTH=2048
export HS=4544
export TP=8
export PP=1
export N_LAYERS=32
export N_AH=71
export FFN_HS=18176
export GBS=256
: ${TRAIN_ITERS:=20000}

export INIT_METHOD_STD=0.02
export LAYERNORM_EPSILON=1e-5
export WARMUP_STEPS=10

# pushd .
# cd /usr/local/lib/python3.8/site-packages/nemo/collections/nlp/data/language_modeling/megatron/
# make
# popd

./test_falcon.sh
