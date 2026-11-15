#!/bin/bash
#SBATCH --job-name=cov_enc_ret
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/cov_enc_ret_%j.out
#SBATCH --error=logs/cov_enc_ret_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/analyze_slope_covariate_reliability.py \
  --slopes-csv id_analysis_stats/enc_vs_ret_cleaned/subject_condition_slopes.csv \
  --reliability-csv id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned/pattern_reliability.csv \
  --output-dir id_analysis_stats/reliability_covariate_enc_vs_ret_cleaned \
  --condition-a Enc \
  --condition-b ret
