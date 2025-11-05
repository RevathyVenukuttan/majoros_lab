#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_EMSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_EMSE_test_v1.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_EMSE_test_v1.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python /hpc/home/rv103/igvf/revathy/BlueSTARR/BlueSTARR-EMSE.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-CustomLoss.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/group/igvf/revathy/models/K562/mse_loss/test_v1