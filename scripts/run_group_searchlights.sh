#!/bin/bash
#SBATCH --job-name=id_group_maps
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_group_maps_%j.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_group_maps_%j.err

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

# 1. Persistent Trace r=5
python code/fmri/analysis/searchlight_id_group.py \
  --input-pattern "outputs/searchlight_id_maps_persistent_r5_nosm/*_id-diff.nii.gz" \
  --output-dir outputs/searchlight_id_maps_persistent_r5_nosm/group

# 2. Metacog Contrast
python code/fmri/analysis/searchlight_id_group.py \
  --input-pattern "outputs/searchlight_id_maps_nosm/*_id-diff.nii.gz" \
  --output-dir outputs/searchlight_id_maps_nosm/group
