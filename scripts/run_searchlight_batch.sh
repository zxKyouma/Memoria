#!/bin/bash
#SBATCH --job-name=searchlight
#SBATCH --partition=comp
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/searchlight_%A_%a.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/searchlight_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-18

module load miniforge3/24.3.0-0
conda activate nnunet

subjects=(
  sub-60989 sub-61350 sub-62015 sub-62038 sub-62150 sub-62426
  sub-62428 sub-63062 sub-63896 sub-65037 sub-65304 sub-65399
  sub-65672 sub-66287 sub-66325 sub-66376 sub-67367 sub-67475 sub-68576
)

subject="${subjects[$SLURM_ARRAY_TASK_ID]}"
subject_id="${subject#sub-}"

PYTHON_BIN="$(which python)"
SCRIPT=/home/zlow/by84/Brandon/memoria/derivatives/bvae_project/run_searchlight_singlesubject.py

echo "[$SLURM_JOB_ID][$SLURM_ARRAY_TASK_ID] Using ${PYTHON_BIN} for ${subject}"

"${PYTHON_BIN}" "${SCRIPT}" "${subject_id}" --session 1 --task Recall
