#!/bin/bash
#SBATCH --job-name=id_searchlight_persist
#SBATCH --partition=comp
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_searchlight_persist_%A_%a.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_searchlight_persist_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-19

cd /home/zlow/by84/Brandon/memoria

module load miniforge3/24.3.0-0
conda activate nnunet

export MKL_THREADING_LAYER=SEQ
export OMP_NUM_THREADS=1

radius="${RADIUS:-3}"

subjects=(
  sub-60100 sub-60989 sub-61350 sub-62015 sub-62038 sub-62150 sub-62426 sub-62428
  sub-63062 sub-63896 sub-65037 sub-65304 sub-65399 sub-65672 sub-66287 sub-66325
  sub-66376 sub-67367 sub-67475 sub-68576
)

subject="${subjects[$SLURM_ARRAY_TASK_ID]}"
subject_id="${subject#sub-}"

python code/fmri/analysis/searchlight_id_persistent_trace.py \
  "${subject_id}" \
  --session 2 \
  --radius "${radius}" \
  --min-trials 4 \
  --min-voxels 20 \
  --estimator twonn \
  --mask-rois V1 hV4 LOIT FUS FG1 FG2 FG3 FG4 phc hippocampus \
  --output-dir "outputs/searchlight_id_maps_persistent_r${radius}_nosm"
