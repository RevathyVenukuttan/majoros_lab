#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J DMSO_MSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/DMSO_MSELoss_5layer_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/DMSO_MSELoss_5layer_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-30%30
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
DMSO=(\
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
DMSO-11 \
DMSO-12 \
DMSO-13 \
DMSO-14 \
DMSO-15 \
DMSO-16 \
DMSO-17 \
DMSO-18 \
DMSO-19 \
DMSO-20 \
DMSO-21 \
DMSO-22 \
DMSO-23 \
DMSO-24 \
DMSO-25 \
DMSO-26 \
DMSO-27 \
DMSO-28 \
DMSO-29 \
DMSO-30 \
)

DMSO=${DMSO[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/A549/configs/A549-MSELoss-5layer.config \
    /hpc/group/igvf/A549/full-set/DMSO-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/A549/mse_loss/DMSO-200/5-layer/${DMSO}



