#!/bin/bash
#SBATCH --job-name=mean_beta_persistent_nosm
#SBATCH --partition=comp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/mean_beta_persistent_nosm_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/mean_beta_persistent_nosm_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/stats/mean_beta_activation_lmm.py \
  --analysis persistent_trace \
  --output-dir id_analysis_stats/mean_beta_activation_nosm \
  --betaseries-root derivatives/nibetaseries_nosm/nibetaseries
