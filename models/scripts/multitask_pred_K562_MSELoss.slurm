#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_MSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_MSELoss_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_MSELoss_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
K562=(\
K562-1 \
K562-2 \
K562-3 \
K562-4 \
K562-5 \
K562-6 \
K562-7 \
K562-8 \
K562-9 \
K562-10 \
)

K562=${K562[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-MSELoss.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/group/igvf/revathy/models/K562/mse_loss/${K562}