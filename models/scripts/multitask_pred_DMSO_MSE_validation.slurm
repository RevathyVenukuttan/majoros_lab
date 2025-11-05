#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J DMSO_MSE_validation
#SBATCH -o /hpc/group/igvf/revathy/models/logs/DMSO_MSELoss_validation_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/DMSO_MSELoss_validation_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-5%5
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
DMSO=( \
DMSO-1 \
DMSO-2 \
DMSO-3 \
DMSO-4 \
DMSO-5)

DMSO=${DMSO[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred-validation.py \
    /hpc/group/igvf/SHARED/A549/A549.config \
    /hpc/group/igvf/A549/full-set/DMSO-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/mse_loss/DMSO-200/validation/${DMSO}



