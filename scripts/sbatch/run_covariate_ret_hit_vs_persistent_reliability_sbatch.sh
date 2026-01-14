#!/bin/bash
#SBATCH --job-name=cov_hit_persist
#SBATCH --partition=comp
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/cov_hit_persist_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/cov_hit_persist_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/analyze_slope_covariate_reliability.py \
  --slopes-csv id_analysis_stats/ret_hit_miss_3way_cleaned/subject_condition_slopes.csv \
  --reliability-csv id_analysis_stats/pattern_reliability_ret_hit_miss_3way_cleaned/pattern_reliability.csv \
  --output-dir id_analysis_stats/reliability_covariate_ret_hit_vs_persistent_cleaned \
  --condition-a ret_hit \
  --condition-b ret_miss_recog_correct
