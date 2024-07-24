#!/usr/bin/env bash

export SEQ_LENGTH=2048
export HS=12288
export TP=32
export PP=8
export N_LAYERS=96
export N_AH=96
export ACT_CHKPNT_GRANULARITY=selective
export GBS_MULTIPLE=8
export LEARNING_RATE=5e-5
./test.sh

