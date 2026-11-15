#!/bin/bash
#SBATCH --job-name=mean_beta_persistent_nosm
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/mean_beta_persistent_nosm_%j.out
#SBATCH --error=logs/mean_beta_persistent_nosm_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/stats/mean_beta_activation_lmm.py \
  --analysis persistent_trace \
  --output-dir id_analysis_stats/mean_beta_activation_nosm \
  --betaseries-root derivatives/nibetaseries_nosm/nibetaseries
