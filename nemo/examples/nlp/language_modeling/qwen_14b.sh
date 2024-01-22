#!/usr/bin/env bash
export SEQ_LENGTH=4096
export HS=5120
export TP=8
export PP=4
export N_LAYERS=40
export N_AH=40
export FFN_HS=13696
export GBS=512
export UBS=1
export TRAIN_ITERS=50000

export VALIDATE_INTERVAL=250
export SAVE_CHECKPOINT_INTERVAL=1000

export INIT_METHOD_STD=0.02
export LAYERNORM_EPSILON=1e-8
export WARMUP_STEPS=500

export LOAD_CHECKPOINT_FROM='/fsx/qwen-14b-tp8-pp4/tp_rank_07_pp_rank_003/model_optim_rng.ckpt'

# This helps to build the helpers.cpp required (only once)
# cd /usr/local/lib/python3.8/site-packages/nemo/collections/nlp/data/language_modeling/megatron/
# make
# cd /root/scripts/nemo
#

./test_qwen.sh
