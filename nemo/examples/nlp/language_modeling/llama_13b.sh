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

# cd /usr/local/lib/python3.8/site-packages/nemo/collections/nlp/data/language_modeling/megatron/
# make
# cd /root/scripts/nemo

./test_llama_mixed_precision.sh