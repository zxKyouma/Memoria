#!/bin/bash
#SBATCH --job-name=rel_enc_ret_clean
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/rel_enc_ret_clean_%j.out
#SBATCH --error=logs/rel_enc_ret_clean_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_pattern_reliability.py \
  --analysis enc_vs_ret_cleaned \
  --output-dir id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned \
  --split odd-even \
  --voxel-subsample all
