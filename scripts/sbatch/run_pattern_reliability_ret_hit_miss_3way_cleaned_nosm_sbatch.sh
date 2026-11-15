#!/bin/bash
#SBATCH --job-name=rel_ret_hit_miss_nosm
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/rel_ret_hit_miss_nosm_%j.out
#SBATCH --error=logs/rel_ret_hit_miss_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_pattern_reliability.py \
  --analysis ret_hit_miss_3way_cleaned \
  --output-dir id_analysis_stats/pattern_reliability_ret_hit_miss_3way_cleaned_nosm \
  --split odd-even \
  --voxel-subsample all \
  --betaseries-root derivatives/nibetaseries_nosm/nibetaseries
