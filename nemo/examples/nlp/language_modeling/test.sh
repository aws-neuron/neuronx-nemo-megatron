#!/usr/bin/env bash
set -o pipefail
set -e

ulimit -n 65535

sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

if [ -v SLURM_NNODES ]
then
    # SLURM runs
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
    export EXPLICIT_LOGDIR=null
    : ${SLURM_RESTART_COUNT:=0}
    LOG_PATH=logs/$SLURM_JOB_ID/$SLURM_RESTART_COUNT/$NODEID/
elif [ -v OMPI_COMM_WORLD_RANK ]
then
    # MPI runs on EKS
    export CCOM_SOCKET_IFNAME=eth0
    NODELIST=`/nodelist_helper.py`
    HOSTS=(${NODELIST//\ / })
    NODEID=$OMPI_COMM_WORLD_RANK
    NTASKS=$OMPI_COMM_WORLD_SIZE
    export EXPLICIT_LOGDIR=/shared/nemo_experiments/$POD_UID
    LOG_PATH=$EXPLICIT_LOGDIR/$NODEID/
else
    # Single-node, non-SLURM, non-MPI runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export NEMO_EXPM_VERSION=$(date "+%Y-%m-%d_%H-%M-%S")
    export EXPLICIT_LOGDIR=null
    LOG_PATH=./nemo_experiments/logs
fi
./setup.sh   2>&1  | tee  $LOG_PATH/setup.log

mkdir -p $LOG_PATH

export HYDRA_FULL_ERROR=1
export PROCESSES_PER_NODE=32
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000

export NEURON_RT_EXEC_TIMEOUT=10
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export MALLOC_ARENA_MAX=128
export TF_NUM_INTEROP_THREADS=8192

export NEURON_RT_STOCHASTIC_ROUNDING_EN=1

#training_precision is one of 'bf16SR', 'megatron_amp_O2', 'fp32_OptStates'
#training_precision = "bf16SR", uses BF16 + Stochastic Rounding
#training_precision = "megatron_amp_O2", master weights and optimizer states are stored in fp32, model weights in bf16
#training_precision = "fp32_OptStates", optimizer states are stored in fp32, model weights in bf16
training_precision="bf16SR"
if [[ $training_precision == "bf16SR" ]];then
    echo using BF16 SR
    export XLA_USE_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=nemo --enable-mixed-precision-accumulation"
    OPTIM_NAME=adamw
    megatron_amp_O2=false
elif [[ $training_precision == "megatron_amp_O2" ]]; then
    echo using megatron_amp_O2
    export XLA_DOWNCAST_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=nemo --enable-mixed-precision-accumulation"
    OPTIM_NAME=adamw
    megatron_amp_O2=true
elif [[ $training_precision == "fp32_OptStates" ]]; then
    echo using FP32 Optimizer States
    export XLA_DOWNCAST_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=nemo --enable-mixed-precision-accumulation"
    OPTIM_NAME=adamw_fp32OptState
    megatron_amp_O2=false
else
    echo Incorrect Training Precision Provided
fi 

export NEURON_COMPILE_CACHE_URL="$HOME/neuron_cache" # Place cache on shared storage to reduce redundant compilations


export TRAIN_ITERS=300000
CREATE_TB_LOGGER=True

if [ "$COMPILE" = "1" ]; then
    echo "compiling only run"
    MAYBE_COMPILE="neuron_parallel_compile"
    TRAIN_ITERS=4
    CREATE_TB_LOGGER=False
fi

: ${SEQ_LENGTH:=2048}
: ${HS:=4096}
: ${TP:=8}
: ${PP:=1}
: ${N_LAYERS:=32}
: ${N_AH:=32}
: ${UBS:=1}
: ${ACT_CHKPNT_GRANULARITY:=full}
: ${GBS_MULTIPLE:=32}
GBS=$((NTASKS*GBS_MULTIPLE))

FFN_HS=$(($HS*4))
echo "SEQ_LEN=$SEQ_LENGTH, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH GBS=$GBS UBS=$UBS"


$MAYBE_COMPILE torchrun $DISTRIBUTED_ARGS megatron_gpt_pretraining.py  \
    --config-path=conf \
    --config-name=megatron_gpt_config \
    trainer.devices=$PROCESSES_PER_NODE \
    trainer.num_nodes=$NTASKS \
    trainer.max_epochs=null \
    trainer.max_steps=$TRAIN_ITERS\
    trainer.val_check_interval=$(($TRAIN_ITERS+1)) \
    trainer.log_every_n_steps=1 \
    trainer.limit_val_batches=1 \
    trainer.limit_test_batches=1 \
    trainer.accumulate_grad_batches=1 \
    trainer.precision=32 \
    model.megatron_amp_O2=$megatron_amp_O2 \
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
    model.hidden_dropout=0.1 \
    model.layernorm_epsilon=1e-5 \
    model.tokenizer.vocab_file=$HOME/examples_datasets/gpt2/gpt2-vocab.json \
    model.tokenizer.merge_file=$HOME/examples_datasets/gpt2/gpt2-merges.txt \
    model.data.data_prefix=[1.0,$HOME/examples_datasets/gpt2/my-gpt2_text_document] \
    model.data.num_workers=1 \
    model.data.seq_length=$SEQ_LENGTH \
    model.optim.name=$OPTIM_NAME \
    model.optim.capturable=True \
    model.optim.lr=0.00015 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.01 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=750 \
    model.optim.sched.constant_steps=80000 \
    model.optim.sched.min_lr=1.0e-5 \
    model.sequence_parallel=True  \
    model.activations_checkpoint_granularity=$ACT_CHKPNT_GRANULARITY \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=1 \
    +model.save_xser=False \
    exp_manager.create_tensorboard_logger=$CREATE_TB_LOGGER \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=False \
    exp_manager.create_checkpoint_callback=False \
    exp_manager.explicit_log_dir=$EXPLICIT_LOGDIR \
    +exp_manager.checkpoint_callback_params.train_time_interval=3600 \
    model.use_cpu_initialization=True   2>&1  | tee  $LOG_PATH/log

# Note: to resume training using a checkpoint, please add the following configuration above, adjusting for your checkpoint path
#    +model.load_xser=True \
#    model.resume_from_checkpoint='/efs/checkpoint/megatron_gpt--step\=1085-consumed_samples\=69632.0-last.ckpt' \
