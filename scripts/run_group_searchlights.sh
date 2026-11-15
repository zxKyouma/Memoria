#!/bin/bash
#SBATCH --job-name=id_group_maps
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=logs/id_group_maps_%j.out
#SBATCH --error=logs/id_group_maps_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

# 1. Persistent Trace r=5
python code/fmri/analysis/searchlight_id_group.py \
  --input-pattern "outputs/searchlight_id_maps_persistent_r5_nosm/*_id-diff.nii.gz" \
  --output-dir outputs/searchlight_id_maps_persistent_r5_nosm/group

# 2. Metacog Contrast
python code/fmri/analysis/searchlight_id_group.py \
  --input-pattern "outputs/searchlight_id_maps_nosm/*_id-diff.nii.gz" \
  --output-dir outputs/searchlight_id_maps_nosm/group
