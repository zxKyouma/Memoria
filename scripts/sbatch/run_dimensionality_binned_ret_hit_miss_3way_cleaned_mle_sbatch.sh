#!/bin/bash
#SBATCH --job-name=id_3way_mle
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/id_3way_mle_%j.out
#SBATCH --error=logs/id_3way_mle_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis ret_hit_miss_3way_cleaned \
  --output-dir id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_mle \
  --estimator mle \
  --mle-k 10
