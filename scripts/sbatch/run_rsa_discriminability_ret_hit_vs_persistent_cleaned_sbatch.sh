#!/bin/bash
#SBATCH --job-name=rsa_disc_persist
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/rsa_disc_persist_%j.out
#SBATCH --error=logs/rsa_disc_persist_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_rsa_discriminability.py \
  --analysis ret_hit_miss_3way_cleaned \
  --condition-a ret_hit \
  --condition-b ret_miss_recog_correct \
  --output-dir id_analysis_stats/discriminability_ret_hit_miss_3way_cleaned \
  --voxel-subsample all \
  --id-results id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned/group_id_full_results_by_condition.csv
