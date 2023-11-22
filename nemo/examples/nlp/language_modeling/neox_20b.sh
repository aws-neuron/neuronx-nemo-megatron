#!/usr/bin/env bash
set -o pipefail

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

export HYDRA_FULL_ERROR=1
export PROCESSES_PER_NODE=32
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000

export NEURON_RT_EXEC_TIMEOUT=10
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export BUCKET_CAP_MB=1024
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=5
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export MALLOC_ARENA_MAX=128

training_precision="bf16SR"
if [[ $training_precision == "bf16SR" ]];then
    echo using BF16 SR
    export XLA_USE_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
    OPTIM_NAME=adamw
    megatron_amp_O2=false
elif [[ $training_precision == "megatron_amp_O2" ]]; then
    echo using megatron_amp_O2
    export XLA_DOWNCAST_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
    OPTIM_NAME=adamw
    megatron_amp_O2=true
elif [[ $training_precision == "fp32_OptStates" ]]; then
    echo using FP32 Optimizer States
    export XLA_DOWNCAST_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
    OPTIM_NAME=adamw_fp32OptState
    megatron_amp_O2=false
else
    echo Incorrect Training Precision Provided
fi 

export TF_NUM_INTEROP_THREADS=1024
export XLA_THREAD_POOL_SIZE=4
export XLA_IO_THREAD_POOL_SIZE=4

: ${SEQ_LENGTH:=2048}
: ${HS:=6144}
: ${TP:=8}
: ${PP:=4}
: ${N_LAYERS:=44}
: ${N_AH:=64}
: ${UBS:=1}
: ${GBS:=256}
export FFN_HS=$(($HS*4))
AL=1
echo "SEQ_LEN=$SEQ_LENGTH, HS=$HS, FFN_HS=$FFN_HS TP=$TP PP=$PP N_LAYERS=$N_LAYERS N_AH=$N_AH AL=$AL GBS=$GBS UBS=$UBS"

LOG_PATH=logs/$SLURM_JOB_ID/$NODEID/
EXP_DIR_OPTION="exp_manager.checkpoint_callback_params.save_last=True"
export TRAIN_ITERS=10000
CREATE_TB_LOGGER=True
LOG_FILE_NAME="run_gpt_neox_20B_seq_len_"$SEQ_LENGTH"_BS_"$UBS"_GBS_"$GBS"_TP_"$TP"_PP_"$PP"_"$(date +"%m-%d-%Y")_$(date +"%H:%M:%S").txt
if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    export TRAIN_ITERS=3
    CREATE_TB_LOGGER=False
    LOG_FILE_NAME="compile_gpt_neox_20B_seq_len_"$SEQ_LENGTH"_BS_"$UBS"_GBS_"$GBS"_TP_"$TP"_PP_"$PP"_"$(date +"%m-%d-%Y")_$(date +"%H:%M:%S").txt
    EXP_DIR_OPTION="exp_manager.checkpoint_callback_params.save_last=False"
fi
mkdir -p $LOG_PATH

torchrun $DISTRIBUTED_ARGS megatron_gpt_pretraining.py \
    --config-path=conf \
    --config-name=megatron_neox_config \
    model.make_vocab_size_divisible_by=$TP \
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
    model.hidden_dropout=0.0 \
    model.layernorm_epsilon=1e-5 \
    model.position_embedding_type=rope \
    model.rotary_percentage=0.25 \
    model.transformer_block_type=gpt_j \
    model.tokenizer.vocab_file=$HOME/examples_datasets/gpt2/gpt2-vocab.json \
    model.tokenizer.merge_file=$HOME/examples_datasets/gpt2/gpt2-merges.txt \
    model.data.data_prefix=[1.0,$HOME/examples_datasets/gpt2/my-gpt2_text_document] \
    model.data.num_workers=1 \
    model.data.seq_length=$SEQ_LENGTH \
    model.data.splits_string=\'980,10,10\' \
    model.optim.name=adamw \
    model.optim.capturable=True \
    model.optim.lr=9.7e-5 \
    model.optim.betas=[0.9,0.95] \
    model.optim.weight_decay=0.01 \
    model.optim.sched.name=CosineAnnealing \
    model.optim.sched.warmup_steps=100 \
    model.optim.sched.constant_steps=0 \
    model.optim.sched.min_lr=1e-5 \
    model.sequence_parallel=True  \
    model.activations_checkpoint_granularity=full \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=$AL \
    +model.save_xser=True \
    exp_manager.create_tensorboard_logger=$CREATE_TB_LOGGER \
    exp_manager.resume_if_exists=False \
    exp_manager.resume_ignore_no_checkpoint=False \
    exp_manager.create_checkpoint_callback=True \
    +exp_manager.checkpoint_callback_params.train_time_interval=48000\
    $EXP_DIR_OPTION \
    model.use_cpu_initialization=True   2>&1  | tee  $LOG_PATH/$LOG_FILE_NAME

# Note: to resume training using a checkpoint, please add the following configuration above, adjusting for your checkpoint path
#    +model.load_xser=True \
#    model.resume_from_checkpoint='/efs/checkpoint/megatron_gpt--step\=1085-consumed_samples\=69632.0-last.ckpt' \
