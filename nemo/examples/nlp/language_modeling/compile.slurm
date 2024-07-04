#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127

export COMPILE=1
export NEURON_PARALLEL_COMPILE_DUMP_RESULTS=1
srun $1
