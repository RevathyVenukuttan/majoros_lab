#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A h200ea
#SBATCH -p h200ea
#SBATCH -J K562_1000bp
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_1000bp_test.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_1000bp_test.err
#SBATCH --nice=100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=72:00:00
#
hostname
nvidia-smi
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-MSELoss-1000bp.config \
    /hpc/group/igvf/revathy/sequences/K562/1000bp \
    /hpc/group/igvf/revathy/models/K562/1000bp/K562-1

