#!/bin/bash
#SBATCH --job-name=rsa_disc_persist
#SBATCH --partition=comp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/rsa_disc_persist_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/rsa_disc_persist_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_rsa_discriminability.py \
  --analysis ret_hit_miss_3way_cleaned \
  --condition-a ret_hit \
  --condition-b ret_miss_recog_correct \
  --output-dir id_analysis_stats/discriminability_ret_hit_miss_3way_cleaned \
  --voxel-subsample all \
  --id-results id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned/group_id_full_results_by_condition.csv
