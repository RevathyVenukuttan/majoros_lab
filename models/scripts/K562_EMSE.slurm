#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_EMSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_EMSE_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_EMSE_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/home/rv103/igvf/revathy/BlueSTARR/BlueSTARR-EMSE.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-CustomLoss.config \
    /hpc/group/igvf/SHARED/K562/with-lib-sizes/ \
    /hpc/group/igvf/revathy/models/K562/mse_loss/${model}