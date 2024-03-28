#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task 127
#SBATCH --open-mode=append
#SBATCH --requeue

# Tune this as desired, it exists to help catch runaway misconfigurations of
# slurm's MaxBatchRequeue parameter if a software error triggers immediate job
# failures.  https://slurm.schedmd.com/slurm.conf.html#OPT_MaxBatchRequeue
restart_limit=100

if srun --kill-on-bad-exit=1 $1 ; then
  # success case
  :
else
  # The ML framework layers may have detected a hardware failure before the
  # slurm healthcheck did.  In that case srun will fail early and won't trigger
  # requeueing without adding this logic.
  if [[ $SLURM_RESTART_COUNT -lt $restart_limit ]] ; then
    scontrol requeue $SLURM_JOB_ID
  else
    echo "exceeded retry limit of ${restart_limit}"
  fi
fi
