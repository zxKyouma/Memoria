#!/bin/bash
#SBATCH --job-name=retRating_id_pilot
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/retRating_id_pilot_%j.out
#SBATCH --error=logs/retRating_id_pilot_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/stats/retRating_time_resolved_id_pilot.py \
  --output-dir id_analysis_stats/retRating_time_resolved_pilot

python code/fmri/analysis/stats/retRating_time_resolved_activation_pilot.py \
  --output-dir id_analysis_stats/retRating_time_resolved_pilot
