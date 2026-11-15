#!/bin/bash
#SBATCH --job-name=tsnr_nosm
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/tsnr_nosm_%j.out
#SBATCH --error=logs/tsnr_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_tsnr_roi.py \
  --id-results id_analysis_results_by_condition_enc_vs_ret_cleaned/group_id_full_results_by_condition.csv \
  --output-dir id_analysis_stats/tsnr_nosm
