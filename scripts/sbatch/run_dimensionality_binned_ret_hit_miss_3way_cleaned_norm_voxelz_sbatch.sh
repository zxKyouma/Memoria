#!/bin/bash
#SBATCH --job-name=id_binned_ret_3way_norm_voxelz
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/id_binned_ret_3way_norm_voxelz_%j.out
#SBATCH --error=logs/id_binned_ret_3way_norm_voxelz_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_dimensionality_binned.py \
    --analysis ret_hit_miss_3way_cleaned \
    --normalization voxel_z \
    --output-dir ./id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_norm-voxelz
