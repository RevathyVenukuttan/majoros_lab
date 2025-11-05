#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_Uncertainty
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_Uncertainty_test.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_Uncertainty_test.err
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
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-Uncertainty-v2.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-Uncertainty-test.config \
    /hpc/group/igvf/SHARED/K562/with-lib-sizes \
    /hpc/group/igvf/revathy/models/K562/uncertainty/test