#!/bin/bash
#SBATCH --job-name=rel_ret_hit_miss_nosm
#SBATCH --partition=comp
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/rel_ret_hit_miss_nosm_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/rel_ret_hit_miss_nosm_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_pattern_reliability.py \
  --analysis ret_hit_miss_3way_cleaned \
  --output-dir id_analysis_stats/pattern_reliability_ret_hit_miss_3way_cleaned_nosm \
  --split odd-even \
  --voxel-subsample all \
  --betaseries-root derivatives/nibetaseries_nosm/nibetaseries
