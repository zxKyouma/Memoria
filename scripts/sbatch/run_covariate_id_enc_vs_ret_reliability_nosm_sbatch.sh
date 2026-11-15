#!/bin/bash
#SBATCH --job-name=cov_id_enc_ret_rel_nosm
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/cov_id_enc_ret_rel_nosm_%j.out
#SBATCH --error=logs/cov_id_enc_ret_rel_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/stats/analyze_id_covariate_reliability.py \
  --id-csv id_analysis_results_by_condition_enc_vs_ret_cleaned_nosm/group_id_full_results_by_condition.csv \
  --reliability-csv id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned_nosm/pattern_reliability.csv \
  --output-dir id_analysis_stats/reliability_covariate_id_enc_vs_ret_cleaned_nosm \
  --condition-a Enc \
  --condition-b ret
