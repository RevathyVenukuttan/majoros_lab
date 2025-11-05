#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J DMSO_MSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/DMSO_MSELoss_6layer_adjusted_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/DMSO_MSELoss_6layer_adjusted_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH -p scavenger-gpu,majoroslab-gpu,igvf-gpu,gpu-common
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
DMSO=(\
DMSO-11 \
DMSO-1 \
DMSO-2 \
DMSO-3 \
DMSO-4 \
DMSO-5 \
DMSO-6 \
DMSO-7 \
DMSO-8 \
DMSO-9 \
DMSO-10 \
)

DMSO=${DMSO[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/A549/configs/A549-MSELoss-6layer-adjusted.config \
    /hpc/group/igvf/A549/full-set/DMSO-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/A549/mse_loss/DMSO-200/6-layer/adjusted/${DMSO}



