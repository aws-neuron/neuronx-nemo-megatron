#!/usr/bin/env bash

source ./train_setup.sh

: ${TRAIN_ITERS:=20000}
: ${INIT_METHOD_STD:=0.021}
: ${LAYERNORM_EPSILON:=1e-5}
: ${WARMUP_STEPS:=10}

: ${SEQ_LENGTH:=8192}
: ${HS:=4096}
: ${TP:=8}
: ${PP:=2}
: ${N_LAYERS:=42}
: ${N_AH:=32}
: ${UBS:=1}
: ${FFN_HS:=16384}
: ${GBS:=256}
: ${FIM_RATE:=0.5}
: ${FIM_SPM_RATE:=0}
: ${TOKENIZER_PATH=$HOME/examples_datasets/starcoder/starcoder_model}
: ${DATASET_PATH=$HOME/examples_datasets/starcoder/codeparrot-clean-mmap_content_document}

echo "SEQ_LEN=$SEQ_LENGTH, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS TRAIN_ITERS=$TRAIN_ITERS"

LOG_PATH=logs/$SLURM_JOB_ID/$NODEID/
mkdir -p $LOG_PATH

$MAYBE_COMPILE torchrun $DISTRIBUTED_ARGS megatron_bigcode_pretraining.py  \
    --config-path=conf \
    --config-name=megatron_bigcode_config \
    trainer.devices=$PROCESSES_PER_NODE \
    trainer.num_nodes=$NTASKS \
    trainer.max_epochs=null \
    trainer.max_steps=$TRAIN_ITERS\
    trainer.val_check_interval=$TRAIN_ITERS \
    trainer.log_every_n_steps=1 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=32 \
    model.megatron_amp_O2=$megatron_amp_O2 \
    model.tokenizer.type=$TOKENIZER_PATH \
    model.micro_batch_size=$UBS \
    model.global_batch_size=$GBS \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.max_position_embeddings=$SEQ_LENGTH \
    model.encoder_seq_length=$SEQ_LENGTH \
    model.hidden_size=$HS \
    model.ffn_hidden_size=$FFN_HS \
    model.num_layers=$N_LAYERS \
    model.num_attention_heads=$N_AH \
    model.init_method_std=$INIT_METHOD_STD \
    model.hidden_dropout=0 \
    model.layernorm_epsilon=$LAYERNORM_EPSILON \
    model.data.data_prefix=[1.0,$DATASET_PATH] \
    model.data.num_workers=1 \
    model.data.seq_length=$SEQ_LENGTH \
    model.optim.name=$OPTIM_NAME \
    model.optim.lr=3.0e-4 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.1 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=$WARMUP_STEPS \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=3.0e-5 \
    model.optim.capturable=True \
    model.sequence_parallel=True  \
    model.activations_checkpoint_granularity=full \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=1 \
    +model.save_xser=True\
    model.data.fim_rate=$FIM_RATE \
    model.data.fim_spm_rate=$FIM_SPM_RATE \
    exp_manager.create_tensorboard_logger=$CREATE_TB_LOGGER \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=False \
    exp_manager.create_checkpoint_callback=$CHECKPOINT_CALLBACK \
    +exp_manager.checkpoint_callback_params.train_time_interval=36000 \
    model.use_cpu_initialization=True   2>&1  | tee  $LOG_PATH/log

# Note: to resume training using a checkpoint, please add the following configuration above, adjusting for your checkpoint path
#    model.use_cpu_initialization=False \
#    +model.load_xser=True \
#    model.resume_from_checkpoint='/efs/checkpoint/megatron_gpt--step\=1085-consumed_samples\=69632.0-last.ckpt' \
