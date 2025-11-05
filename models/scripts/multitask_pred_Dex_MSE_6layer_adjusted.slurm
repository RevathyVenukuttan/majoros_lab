#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J Dex_MSE
#SBATCH -o /hpc/group/igvf/revathy/models/logs/Dex_MSELoss_6layer_adjusted_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/Dex_MSELoss_6layer_adjusted_%a.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH -p majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#
DEX=(\
Dex-11 \
Dex-1 \
Dex-2 \
Dex-3 \
Dex-4 \
Dex-5 \
Dex-6 \
Dex-7 \
Dex-8 \
Dex-9 \
Dex-10 \
)

DEX=${DEX[${SLURM_ARRAY_TASK_ID}]}
python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/home/rv103/igvf/revathy/models/A549/configs/A549-MSELoss-6layer-adjusted.config \
    /hpc/group/igvf/A549/full-set/Dex-200/300-bases/data-normalized \
    /hpc/group/igvf/revathy/models/A549/mse_loss/Dex-200/6-layer/adjusted/${DEX}



