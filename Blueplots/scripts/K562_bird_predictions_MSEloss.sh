#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J bird_pred
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/K562_MSEloss_bird_predictions.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/K562_MSEloss_bird_predictions.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#
hostname
nvidia-smi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python /hpc/home/rv103/igvf/revathy/BlueSTARR/predict_variant_v1.py \
--cre /hpc/group/igvf/revathy/Blueplots/data/Bird_calls_pred_subset_count50.txt \
--input-format bird \
--model-stem /hpc/group/igvf/revathy/models/K562/mse_loss/K562-2 \
--two-bit /hpc/group/igvf/A549/extra_GCs/hg38.2bit \
--seq-len 300 \
--output /hpc/group/igvf/revathy/Blueplots/predictions/K562_MSELoss_bird_predictions_count50.tsv

