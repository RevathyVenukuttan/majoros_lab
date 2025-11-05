#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J biased_pred
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_biased_v1.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/kircher_region_pred_biased_v1.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
python /hpc/home/rv103/igvf/revathy/BlueSTARR/predict-variants-v1.py \
--cre /hpc/group/igvf/K562/mdl_eval/BlueSTARR_vs_DeltaSVM/Kircher_entire_regions/Kircher_region_chunk.txt \
--model-stem /hpc/group/igvf/revathy/models/K562/downsampled/biased/K562-1 \
--two-bit /hpc/group/igvf/A549/extra_GCs/hg38.2bit \
--seq-len 300 \
--output /hpc/group/igvf/revathy/Blueplots/predictions/K562_MSELoss_biased_downsampled_v1.tsv

