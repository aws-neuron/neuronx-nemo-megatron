#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127
#SBATCH --open-mode=append

srun  --kill-on-bad-exit=1  $1