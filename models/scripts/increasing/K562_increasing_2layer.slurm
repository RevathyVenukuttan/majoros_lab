#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_2layer
#SBATCH -o /hpc/group/igvf/revathy/models/logs/increasing/K562_2layer_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/increasing/K562_2layer_%a.err
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
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/K562/configs/K562-MSELoss-2layer.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/home/rv103/igvf/revathy/models/K562/increasing/2-layer/${model}