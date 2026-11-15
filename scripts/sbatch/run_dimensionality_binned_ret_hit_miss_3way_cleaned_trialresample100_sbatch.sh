#!/bin/bash
#SBATCH --job-name=id_binned_ret_hit_miss_3way_tr100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/id_binned_ret_hit_miss_3way_tr100_%j.out
#SBATCH --error=logs/id_binned_ret_hit_miss_3way_tr100_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis ret_hit_miss_3way_cleaned \
  --trial-resamples 100 \
  --output-dir ./id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_trialresample100
