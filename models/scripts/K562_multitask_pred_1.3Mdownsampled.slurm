#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_1.3M
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_1.3M_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_1.3M_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=11-20%10
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
K562=(K562-10 \
K562-11 \
K562-12 \
K562-13 \
K562-14 \
K562-15 \
K562-16 \
K562-17 \
K562-18 \
K562-19 \
K562-20)

K562=${K562[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-MSELoss-downsampled1.3M.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/group/igvf/revathy/models/K562/downsampled/unbiased/1.3M/new/${K562}



