#!/bin/bash
#SBATCH --job-name=control_roi_persistent_nosm
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/control_roi_persistent_nosm_%j.out
#SBATCH --error=logs/control_roi_persistent_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/stats/control_roi_condition_lmm.py \
  --input id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_controls_nosm/group_id_full_results_by_condition.csv \
  --output-dir id_analysis_stats/control_roi_condition_lmm_ret_hit_vs_persistent_controls_nosm \
  --condition-ref ret_hit \
  --condition-test ret_miss_recog_correct \
  --conditions ret_hit ret_miss_recog_correct \
  --roi-ref hippocampus \
  --contrast-rois A1 M1 S1 MT V1
