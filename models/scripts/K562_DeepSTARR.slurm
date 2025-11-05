#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_DeepSTARR_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_DeepSTARR_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-DS.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562_DeepSTARR.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/group/igvf/revathy/models/K562/DeepSTARR/${model}
