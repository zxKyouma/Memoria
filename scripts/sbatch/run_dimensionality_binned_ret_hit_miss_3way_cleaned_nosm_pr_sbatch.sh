#!/bin/bash
#SBATCH --job-name=id_binned_hit_miss_pr_nosm
#SBATCH --partition=comp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_binned_hit_miss_pr_nosm_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_binned_hit_miss_pr_nosm_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis ret_hit_miss_3way_cleaned \
  --estimator pr \
  --output-dir id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_nosm_pr \
  --betaseries-root derivatives/nibetaseries_nosm/nibetaseries
