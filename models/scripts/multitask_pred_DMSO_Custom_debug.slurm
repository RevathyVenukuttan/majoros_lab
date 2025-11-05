#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J DMSO_Custom
#SBATCH -o /hpc/group/igvf/revathy/models/logs/DMSO_CustomLoss_test_v1.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/DMSO_CustomLoss_test_v1.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
DMSO=( \
DMSO-1 \
)

DMSO=${DMSO[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/custom_loss/A549-CustomLoss.config \
    /hpc/group/igvf/A549/full-set/DMSO-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/custom_loss/DMSO-200/test/${DMSO}



