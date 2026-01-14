#!/bin/bash
#SBATCH --job-name=id_retRating_sure_guess
#SBATCH --partition=comp
#SBATCH --output=/home/zlow/by84/Brandon/memoria/logs/id_retRating_sure_guess_%A.out
#SBATCH --error=/home/zlow/by84/Brandon/memoria/logs/id_retRating_sure_guess_%A.err
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

python code/fmri/analysis/compute_dimensionality_binned.py \
  --analysis retRating_sure_guess_matched \
  --min-trials 4 \
  --output-dir id_analysis_results_by_condition_ses-2_retRating_sure_guess_matched \
  --roi-list V1 hV4 LOIT phc hippocampus A1 M1
