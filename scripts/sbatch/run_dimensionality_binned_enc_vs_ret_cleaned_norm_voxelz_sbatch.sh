#!/bin/bash
#SBATCH --job-name=id_binned_enc_ret_norm_voxelz
#SBATCH --partition=comp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_binned_enc_ret_norm_voxelz_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_binned_enc_ret_norm_voxelz_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

python code/fmri/analysis/compute_dimensionality_binned.py \
    --analysis enc_vs_ret_cleaned \
    --normalization voxel_z \
    --output-dir ./id_analysis_results_by_condition_enc_vs_ret_cleaned_norm-voxelz
