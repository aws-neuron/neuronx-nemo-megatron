#!/usr/bin/env bash
set -o pipefail
ulimit -n 65535
sysctl -w net.ipv4.ip_local_reserved_ports=41000

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export CCOM_SOCKET_IFNAME=eth0

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
elif [ -v OMPI_COMM_WORLD_RANK ]
then
    # MPI runs
    NODELIST=`/nodelist_helper.py`
    HOSTS=(${NODELIST//\ / })
    NODEID=$OMPI_COMM_WORLD_RANK
    NTASKS=$OMPI_COMM_WORLD_SIZE
else
    # Single-node, non-SLURM, non-MPI runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
fi

export PROCESSES_PER_NODE=32
export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=41000

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

export MALLOC_ARENA_MAX=128
export XLA_USE_BF16=1
export TF_NUM_INTEROP_THREADS=8192

torchrun $DISTRIBUTED_ARGS allreduce.py 2>&1 | tee /data/$(hostname).log
