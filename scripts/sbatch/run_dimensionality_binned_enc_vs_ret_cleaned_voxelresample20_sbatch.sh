#!/bin/bash
#SBATCH --job-name=id_enc_ret_vox20
#SBATCH --partition=comp
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_enc_ret_vox20_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_enc_ret_vox20_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis enc_vs_ret_cleaned \
  --output-dir id_analysis_results_by_condition_enc_vs_ret_cleaned_voxelresample20 \
  --voxel-resamples 20
