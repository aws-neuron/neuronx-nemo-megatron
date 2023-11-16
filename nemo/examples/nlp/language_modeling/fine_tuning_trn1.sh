#!/usr/bin/env bash
set -o pipefail

ulimit -n 65535

sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

if [ -z "${SLURM_NNODES}" ]
then
    # Single-node, non-SLURM runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export NEMO_EXPM_VERSION=$(date "+%Y-%m-%d_%H-%M-%S")
else
    # SLURM runs, single or multi-node
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
fi

export HYDRA_FULL_ERROR=1
export PROCESSES_PER_NODE=32
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000

export NEURON_RT_EXEC_TIMEOUT=10
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES
export NEURON_RT_DBG_A2A_CC=0
export NEURON_RT_ASYNC_EXEC_MODE=0

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_ENABLE_VERBOSE_NUMERICAL_ERRORS=0
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export MALLOC_ARENA_MAX=128

export XLA_USE_BF16=1
export NEURON_CC_FLAGS="--model-type transformer --enable-mixed-precision-accumulation --enable-experimental-O1 --distribution-strategy=nemo --cache_dir=$NEURON_CACHE_DIR"
export TF_NUM_INTEROP_THREADS=8192

CREATE_TB_LOGGER=True
CHECKPOINT_CALLBACK=True
if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    export TRAIN_ITERS=3
    CREATE_TB_LOGGER=False
    CHECKPOINT_CALLBACK=False
fi

: ${SEQ_LENGTH:=4096}
: ${TP:=8}
: ${PP:=1}
: ${UBS:=1}
: ${GBS:=512}
echo "SEQ_LEN=$SEQ_LENGTH, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS MIN_LR=$MIN_LR"
echo "INIT_METHOD_STD=$INIT_METHOD_STD, HIDDEN_DROPOUT=$HIDDEN_DROPOUT, LAYERNORM_EPSILON=$LAYERNORM_EPSILON, OPTIM_NAME=$OPTIM_NAME, OPTIM_LR=$OPTIM_LR"
echo "OPTIM_WEIGHT_DECAY=$OPTIM_WEIGHT_DECAY, OPTIM_SCHED_NAME=$OPTIM_SCHED_NAME"


LOG_PATH=logs/$SLURM_JOB_ID/$NODEID/
mkdir -p $LOG_PATH



torchrun $DISTRIBUTED_ARGS megatron_gpt_pretraining.py  \
    --config-path=conf \
    --config-name=megatron_llama_config \
    trainer.devices=$PROCESSES_PER_NODE \
    trainer.num_nodes=$NTASKS \
    trainer.max_epochs=null \
    trainer.max_steps=$TRAIN_ITERS\
    trainer.val_check_interval=0.99 \
    trainer.log_every_n_steps=1 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=32 \
    model.tokenizer.type=$PATH_TO_TOKENIZER \
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
    model.init_method_std=0.021 \
    model.hidden_dropout=0 \
    model.layernorm_epsilon=$LAYERNORM_EPSILON \
    model.data.data_prefix=[1.0,/opt/ml/code/tmp/tokenized_data_text_document] \
    model.data.num_workers=1 \
    model.data.seq_length=$SEQ_LENGTH \
    model.data.splits_string=$VALIDATON_SPLIT_RATIO \
    model.optim.name=adamw \
    model.optim.lr=$OPTIM_LR \
    model.optim.betas=[$ADAM_BETA1,$ADAM_BETA2] \
    model.optim.weight_decay=$OPTIM_WEIGHT_DECAY \
    model.optim.sched.name=$OPTIM_SCHED_NAME \
    model.optim.sched.warmup_steps=$OPTIM_WARMUP_STEPS \
    model.optim.sched.constant_steps=$OPTIM_CONSTANT_STEPS \
    model.optim.sched.min_lr=$MIN_LR \
    model.optim.capturable=True \
    model.sequence_parallel=True  \
    model.activations_checkpoint_granularity=full \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=1 \
    +model.save_xser=True \
    exp_manager.create_tensorboard_logger=$CREATE_TB_LOGGER \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=False \
    exp_manager.create_checkpoint_callback=$CHECKPOINT_CALLBACK \
    +exp_manager.checkpoint_callback_params.train_time_interval=36000 \
    exp_manager.checkpoint_callback_params.save_last=True \
    exp_manager.exp_dir=/tmp \
    model.use_cpu_initialization=False \
    +model.load_xser=True \
    model.megatron_amp_O2=$MIXED_PRECISION \
    +model.resume_from_checkpoint=$PROCESSED_CHECKPOINTS_NEMO 2>&1  | tee  $LOG_PATH/log
