#!/bin/bash
#SBATCH --job-name=rel_id_summary
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/rel_id_summary_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/rel_id_summary_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/analyze_reliability_vs_id.py \
  --reliability-csv id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned/pattern_reliability.csv \
  --id-csv id_analysis_results_by_condition_enc_vs_ret_cleaned/group_id_full_results_by_condition.csv \
  --output-dir id_analysis_stats/reliability_vs_id_enc_vs_ret_cleaned \
  --condition-a Enc \
  --condition-b ret \
  --threshold 0.02
