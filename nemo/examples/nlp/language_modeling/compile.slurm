#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127

export COMPILE=1
srun $1
