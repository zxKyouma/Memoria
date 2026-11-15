#!/bin/bash
#SBATCH --job-name=rel_bin_enc_ret_nosm
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/rel_bin_enc_ret_nosm_%j.out
#SBATCH --error=logs/rel_bin_enc_ret_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/stats/reliability_binned_enc_ret.py \
  --slopes id_analysis_stats/enc_vs_ret_cleaned_nosm/subject_condition_slopes.csv \
  --reliability id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned_nosm/pattern_reliability.csv \
  --output-dir id_analysis_stats/reliability_binned_enc_vs_ret_cleaned_nosm
