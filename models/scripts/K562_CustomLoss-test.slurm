#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_Custom
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_CustomLoss_test1.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_CustomLoss_test1.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname && nvidia-smi
python /hpc/home/rv103/igvf/revathy/BlueSTARR/BlueSTARR-NLL.py \
    /hpc/home/rv103/igvf/revathy/models/K562/configs/K562-CustomLoss-test.config \
    /hpc/group/igvf/SHARED/K562/with-lib-sizes \
    /hpc/group/igvf/revathy/models/K562/custom_loss/test_v1