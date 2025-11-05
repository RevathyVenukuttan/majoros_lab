#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A h200ea
#SBATCH -p h200ea
#SBATCH -J K562_4layer
#SBATCH -o /hpc/group/igvf/revathy/models/logs/increasing/K562_4layer_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/increasing/K562_4layer_%a.err
#SBATCH --gres=gpu:h200:1
#SBATCH --array=1-10%10
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#
hostname && nvidia-smi
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/K562/configs/K562-MSELoss-4layer.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/home/rv103/igvf/revathy/models/K562/increasing/4-layer/${model}