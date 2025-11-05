#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J DMSO_v3
#SBATCH -o /hpc/group/igvf/revathy/models/logs/DMSO_CustomLoss_test_v3.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/DMSO_CustomLoss_test_v3.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
DMSO=( \
DMSO-3 \
)

DMSO=${DMSO[${SLURM_ARRAY_TASK_ID}]}
python /hpc/home/rv103/igvf/A549/full-set/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/custom_loss/A549-CustomLoss_v3.config \
    /hpc/group/igvf/A549/full-set/DMSO-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/custom_loss/DMSO-200/test/${DMSO}



