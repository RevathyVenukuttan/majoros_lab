#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_Scaled_15_CustomLoss
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_K562_Scaled_15_MSELoss.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_K562_Scaled_15_MSELoss.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
module load BigWig
python /hpc/group/igvf/revathy/BlueSTARR/predict_variant.py \
--cre /hpc/group/igvf/K562/mdl_eval/BlueSTARR_vs_DeltaSVM/Kircher_entire_regions/Kircher_region_chunk.txt \
--model-stem /hpc/group/igvf/revathy/models/K562/scaled/0.15/mse/K562_0.15-5 \
--two-bit /hpc/group/igvf/A549/extra_GCs/hg38.2bit \
--seq-len 300 \
--output /hpc/group/igvf/revathy/Blueplots/predictions/K562_Scaled_15_MSELoss.tsv
