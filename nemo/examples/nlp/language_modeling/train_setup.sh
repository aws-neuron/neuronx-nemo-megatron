#!/usr/bin/env bash
set -o pipefail
set -e

ulimit -n 65535

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

if [ -v SLURM_NNODES ]
then
    # SLURM runs
    sudo sysctl -w net.ipv4.ip_local_reserved_ports=41000
    if which lctl >/dev/null 2>&1; then
        sudo lctl set_param 'osc.*.max_dirty_mb=64' # Cap max space each connection to FSx reserves so we avoid OODs
    fi
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
    mkdir -p $LOG_PATH
    export NEURON_COMPILE_CACHE_URL="$HOME/neuron_cache" # Place cache on shared storage to reduce redundant compilations
    # Make sure to install latest runtime
    ./setup.sh   2>&1  | tee  $LOG_PATH/setup.log
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
    mkdir -p $LOG_PATH
    export NEURON_COMPILE_CACHE_URL="/shared/neuron_cache" # Place cache on shared storage to reduce redundant compilations
else
    # Single-node, non-SLURM, non-MPI runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export NEMO_EXPM_VERSION=$(date "+%Y-%m-%d_%H-%M-%S")
    export EXPLICIT_LOGDIR=null
    LOG_PATH=./nemo_experiments/logs
    mkdir -p $LOG_PATH
fi

export HYDRA_FULL_ERROR=1
export PROCESSES_PER_NODE=32
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000

export NEURON_RT_EXEC_TIMEOUT=100
export DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export BUCKET_CAP_MB=1024
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=5
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export MALLOC_ARENA_MAX=128
export TF_NUM_INTEROP_THREADS=1024
export XLA_THREAD_POOL_SIZE=4
export XLA_IO_THREAD_POOL_SIZE=4


export NEURON_RT_STOCHASTIC_ROUNDING_EN=1

#training_precision is one of 'bf16SR', 'megatron_amp_O2', 'fp32_OptStates'
#training_precision = "bf16SR", uses BF16 + Stochastic Rounding
#training_precision = "megatron_amp_O2", master weights and optimizer states are stored in fp32, model weights in bf16
#training_precision = "fp32_OptStates", optimizer states are stored in fp32, model weights in bf16
training_precision="bf16SR"
if [[ $training_precision == "bf16SR" ]];then
    echo using BF16 SR
    export XLA_USE_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
    export OPTIM_NAME=adamw
    export megatron_amp_O2=false
elif [[ $training_precision == "megatron_amp_O2" ]]; then
    echo using megatron_amp_O2
    export XLA_DOWNCAST_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
    export OPTIM_NAME=adamw
    export megatron_amp_O2=true
elif [[ $training_precision == "fp32_OptStates" ]]; then
    echo using FP32 Optimizer States
    export XLA_DOWNCAST_BF16=1
    export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
    export OPTIM_NAME=adamw_fp32OptState
    export megatron_amp_O2=false
else
    echo Incorrect Training Precision Provided
fi



export CREATE_TB_LOGGER=True
export CHECKPOINT_CALLBACK=True

if [ "$COMPILE" = "1" ]; then
    echo "compiling only run"
    MAYBE_COMPILE="neuron_parallel_compile"
    export TRAIN_ITERS=4
    CREATE_TB_LOGGER=False
    CHECKPOINT_CALLBACK=False
fi

