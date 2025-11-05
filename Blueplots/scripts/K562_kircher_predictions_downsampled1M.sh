#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J prediction_1M
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_1M.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_1M.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#

module load BigWig
python /hpc/group/igvf/K562/leave-one-out/BlueSTARR/mutator2-parent.py \
 /hpc/home/rv103/igvf/revathy/models/K562/downsampled/unbiased/1M/K562-10 \
 /hpc/group/igvf/K562/mdl_eval/BlueSTARR_vs_DeltaSVM/Kircher_entire_regions/Kircher_region_chunk.txt \
 300 3000 \
 /hpc/home/rv103/igvf/revathy/Blueplots/predictions/K562_DMSO_MSELoss_downsampled1M.txt
