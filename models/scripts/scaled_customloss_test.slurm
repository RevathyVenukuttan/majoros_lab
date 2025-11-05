#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_0.15
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_Scaled_15_Custom_test.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_Scaled_15_Custom_test.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname && nvidia-smi
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-NLL.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-CustomLoss-test.config \
    /hpc/group/igvf/SHARED/K562/with-lib-sizes/scaled-0.15/custom \
    /hpc/group/igvf/revathy/models/K562/scaled/0.15/custom/testv1