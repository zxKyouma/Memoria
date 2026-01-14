#!/bin/bash
#SBATCH --job-name=retRating_id_pilot
#SBATCH --partition=comp
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/retRating_id_pilot_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/retRating_id_pilot_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/stats/retRating_time_resolved_id_pilot.py \
  --output-dir id_analysis_stats/retRating_time_resolved_pilot

python code/fmri/analysis/stats/retRating_time_resolved_activation_pilot.py \
  --output-dir id_analysis_stats/retRating_time_resolved_pilot
