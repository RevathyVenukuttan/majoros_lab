#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A h200ea
#SBATCH -p h200ea
#SBATCH -J K562_1000bp
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_1000bp_v1_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_1000bp_v1_%a.err
#SBATCH --gres=gpu:h200:1
#SBATCH --array=1-10%10
#SBATCH --nice=100
#SBATCH --mem=350G
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#
hostname
nvidia-smi
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-MSELoss-1000bp.config \
    /hpc/group/igvf/revathy/sequences/K562/1000bp \
    /hpc/group/igvf/revathy/models/K562/1000bp/new/${model}

