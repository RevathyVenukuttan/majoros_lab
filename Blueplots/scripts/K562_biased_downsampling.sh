#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J biased_downsampling
#SBATCH -o /hpc/home/rv103/igvf/revathy/Blueplots/logs/K562_biased_downsampling.out
#SBATCH -e /hpc/home/rv103/igvf/revathy/Blueplots/logs/K562_biased_downsampling.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#

python /hpc/home/rv103/igvf/A549/full-set/BlueSTARR/downsample-biased-sim.py \
    /hpc/home/rv103/igvf/K562/full-set/300bp/data-normalized/all-train.fasta \
    /hpc/home/rv103/igvf/K562/full-set/300bp/data-normalized/all-train-counts.txt.gz \
    100 1700000 /hpc/home/rv103/igvf/revathy/Blueplots/data/biased train-lognormal10 lognormal^10
