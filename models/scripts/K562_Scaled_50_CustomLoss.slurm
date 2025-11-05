#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_0.50
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_Scaled_50_Custom_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_Scaled_50_Custom_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname && nvidia-smi
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-NLL.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-CustomLoss.config \
    /hpc/group/igvf/SHARED/K562/with-lib-sizes/scaled-0.50/custom \
    /hpc/group/igvf/revathy/models/K562/scaled/0.50/custom/${model}