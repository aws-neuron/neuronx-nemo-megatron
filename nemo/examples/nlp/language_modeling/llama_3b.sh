#!/usr/bin/env bash
export SEQ_LENGTH=2048
export HS=2560
export TP=8
export PP=1
export N_LAYERS=32
export N_AH=32
export FFN_HS=6912
export GBS=32
export UBS=1

cd /usr/local/lib/python3.8/site-packages/nemo/collections/nlp/data/language_modeling/megatron/
make
cd /root/scripts/nemo

/root/scripts/nemo/nemo/examples/nlp/language_modeling/test_llama_mixed_precision.sh