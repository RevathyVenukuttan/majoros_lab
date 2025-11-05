#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_TE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/K562_TE_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/K562_TE_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --array=1-10%10
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#
hostname
nvidia-smi
model=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/group/igvf/revathy/hyper_git/BlueSTARR-Transformer.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-TE.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/group/igvf/revathy/models/K562/TE/${model}



