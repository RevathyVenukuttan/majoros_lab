#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J biased_pred
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_biased.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_biased.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
module load BigWig
python /hpc/group/igvf/K562/leave-one-out/BlueSTARR/mutator2-parent.py \
 /hpc/group/igvf/revathy/models/K562/downsampled/biased/K562-1 \
 /hpc/group/igvf/K562/mdl_eval/BlueSTARR_vs_DeltaSVM/Kircher_entire_regions/Kircher_region_chunk.txt \
 300 3000 \
 /hpc/group/igvf/revathy/Blueplots/predictions/K562_MSELoss_biased_downsampled.txt
