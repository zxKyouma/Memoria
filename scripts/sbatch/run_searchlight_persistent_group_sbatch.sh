#!/bin/bash
#SBATCH --job-name=id_searchlight_persist_group
#SBATCH --partition=comp
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_searchlight_persist_group_%A.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_searchlight_persist_group_%A.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

export MKL_THREADING_LAYER=SEQ
export OMP_NUM_THREADS=1

for r in 3 5 7; do
  python code/fmri/analysis/searchlight_id_group.py \
    --input-pattern "outputs/searchlight_id_maps_persistent_r${r}/sub-*_desc-ret_miss_recog_correct_minus_ret_hit_id-diff.nii.gz" \
    --output-dir "outputs/searchlight_id_maps_persistent_r${r}/group" \
    --permutations 1000
done
