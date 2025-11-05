#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J Dex_MSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/Dex_MSELoss_v2_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/Dex_MSELoss_v2_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
Dex=(Dex-1)

Dex=${Dex[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/custom_loss/A549-MSELoss-v2.config \
    /hpc/group/igvf/A549/full-set/Dex-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/mse_loss/Dex-200/validation/${Dex}