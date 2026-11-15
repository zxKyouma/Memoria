#!/bin/bash
#SBATCH --job-name=control_roi_enc_ret_nosm
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/control_roi_enc_ret_nosm_%j.out
#SBATCH --error=logs/control_roi_enc_ret_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/stats/control_roi_condition_lmm.py \
  --input id_analysis_results_by_condition_enc_vs_ret_cleaned_controls_nosm/group_id_full_results_by_condition.csv \
  --output-dir id_analysis_stats/control_roi_condition_lmm_enc_vs_ret_cleaned_controls_nosm \
  --condition-ref Enc \
  --condition-test ret \
  --conditions Enc ret \
  --roi-ref hippocampus \
  --contrast-rois A1 M1 S1 MT V1
