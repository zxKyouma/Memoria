#!/bin/bash
#SBATCH --job-name=cov_enc_ret_rel_nosm
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/cov_enc_ret_rel_nosm_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/cov_enc_ret_rel_nosm_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/analyze_slope_covariate_reliability.py \
  --slopes-csv id_analysis_stats/enc_vs_ret_cleaned_nosm/subject_condition_slopes.csv \
  --reliability-csv id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned_nosm/pattern_reliability.csv \
  --output-dir id_analysis_stats/reliability_covariate_enc_vs_ret_cleaned_nosm \
  --condition-a Enc \
  --condition-b ret
