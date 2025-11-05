#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J MUTATE
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_customloss.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_customloss.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,biostat-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#

module load BigWig
python /hpc/group/igvf/K562/leave-one-out/BlueSTARR/mutator2-parent.py \
 /hpc/home/rv103/igvf/revathy/models/K562/custom_loss/DMSO/DMSO-3 \
 /hpc/group/igvf/K562/mdl_eval/BlueSTARR_vs_DeltaSVM/Kircher_entire_regions/Kircher_region_chunk.txt \
 300 3000 \
 /hpc/home/rv103/igvf/revathy/Blueplots/predictions/K562_CustomLoss_Kircher_region_pred.txt
