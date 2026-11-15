#!/bin/bash
#SBATCH --job-name=id_sl_enc_ret
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-19
#SBATCH --output=logs/id_sl_enc_ret_%A_%a.out
#SBATCH --error=logs/id_sl_enc_ret_%A_%a.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

export MKL_THREADING_LAYER=SEQ
export OMP_NUM_THREADS=1

subjects=(
  60100 60989 61350 62015 62038 62150 62426 62428
  63062 63896 65037 65304 65399 65672 66287 66325
  66376 67367 67475 68576
)

subject_id="${subjects[$SLURM_ARRAY_TASK_ID]}"

python code/fmri/analysis/searchlight_id_single_subject.py \
  "${subject_id}" \
  --session-a 2 --condition-a ret \
  --session-b 1 --condition-b Enc \
  --radius 3 \
  --min-trials 10 \
  --min-voxels 20 \
  --estimator twonn \
  --output-dir "outputs/searchlight_id_maps_enc_vs_ret_nosm"
