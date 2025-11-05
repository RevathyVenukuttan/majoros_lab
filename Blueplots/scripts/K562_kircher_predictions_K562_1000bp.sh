#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_1000
#SBATCH -o /hpc/group/igvf/revathy/Blueplots/logs/kircher_region_pred_K562_1000bp.out
#SBATCH -e /hpc/group/igvf/revathy/Blueplots/logs/kircher_region_pred_K562_1000bp.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
module load BigWig
python /hpc/home/rv103/igvf/revathy/BlueSTARR/predict_variant.py \
--cre /hpc/group/igvf/K562/mdl_eval/BlueSTARR_vs_DeltaSVM/Kircher_entire_regions/Kircher_region_chunk.txt \
--model-stem /hpc/home/rv103/igvf/revathy/models/K562/1000bp/new/K562_1000bp-3 \
--two-bit /hpc/group/igvf/A549/extra_GCs/hg38.2bit \
--seq-len 1000 \
--output /hpc/group/igvf/revathy/Blueplots/predictions/K562_1000bp.tsv
