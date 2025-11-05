#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J Dex
#SBATCH -o /hpc/group/igvf/revathy/models/logs/Dex_MSELoss_5layer_new_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/Dex_MSELoss_5layer_new_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-2%2
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#

hostname
nvidia-smi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}

python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/A549/configs/A549-MSELoss-5layer.config \
    /hpc/group/igvf/A549/full-set/Dex-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/A549/mse_loss/Dex-200/5-layer/new/${model}



