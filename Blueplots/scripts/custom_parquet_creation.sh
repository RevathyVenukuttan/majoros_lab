#!/bin/sh
#
#SBATCH --get-user-env
#SBATCH -A majoroslab
#SBATCH -J K562_0.15
#SBATCH -o /hpc/group/igvf/revathy/models/logs/custom_parquet_creation_%a.out
#SBATCH -e /hpc/group/igvf/revathy/models/logs/custom_parquet_creation_%a.err
#SBATCH --exclusive
#SBATCH --array=1-10%10
#SBATCH --nice=100
#SBATCH --mem=102400
#SBATCH --cpus-per-task=1
#SBATCH -t 2-00:00:00
#
file=${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}
python /hpc/home/rv103/igvf/revathy/BlueSTARR-viz/starrutil/igvf_allelepred2db.py \
    --verbose --format unprocessed --partition-by "" \
    -i /hpc/home/rv103/igvf/revathy/Blueplots/predictions/Scaled_15/custom/${file}.txt \
    --db /hpc/home/rv103/igvf/revathy/db/scaled_data_15/Kircher-2019_K562-15-Custom-${SLURM_ARRAY_TASK_ID}.parquet