#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_1.3M
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=19-20%2
#SBATCH -p gpu-common,scavenger-gpu,majoroslab-gpu,igvf-gpu
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#SBATCH --output=/hpc/group/igvf/revathy/models/logs/K562_1.3M_%x_%a.out
#SBATCH --error=/hpc/group/igvf/revathy/models/logs/K562_1.3M_%x_%a.err

K562="K562-${SLURM_ARRAY_TASK_ID}"
LOGDIR="/hpc/group/igvf/revathy/models/logs"
mv "${LOGDIR}/K562_1.3M_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.out" "${LOGDIR}/K562_1.3M_${SLURM_ARRAY_TASK_ID}.out"
mv "${LOGDIR}/K562_1.3M_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.err" "${LOGDIR}/K562_1.3M_${SLURM_ARRAY_TASK_ID}.err"

python /hpc/group/igvf/revathy/BlueSTARR/BlueSTARR-multitask-pred.py \
    /hpc/group/igvf/revathy/models/K562/configs/K562-MSELoss-downsampled1.3M.config \
    /hpc/group/igvf/K562/full-set/300bp/data-normalized \
    /hpc/group/igvf/revathy/models/K562/downsampled/unbiased/1.3M/new/${K562}
