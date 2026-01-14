#!/bin/bash
#SBATCH --job-name=id_binned_ret_hit_miss_3way_tr100
#SBATCH --partition=comp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_binned_ret_hit_miss_3way_tr100_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_binned_ret_hit_miss_3way_tr100_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis ret_hit_miss_3way_cleaned \
  --trial-resamples 100 \
  --output-dir ./id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_trialresample100
