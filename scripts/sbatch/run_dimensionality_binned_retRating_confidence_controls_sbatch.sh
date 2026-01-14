#!/bin/bash
# Compute retRating confidence IDs including control ROIs (smoothed betaseries).
# Output goes to a separate directory so existing results remain unchanged.
#SBATCH --job-name=id_binned_retRating_conf_controls
#SBATCH --partition=comp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_binned_retRating_conf_controls_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_binned_retRating_conf_controls_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis retRating_confidence \
  --output-dir id_analysis_results_by_condition_ses-2_retRating_confidence_controls \
  --roi-list V1 hV4 LOIT phc hippocampus A1 M1 S1 MT
