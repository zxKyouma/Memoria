#!/bin/bash
#SBATCH --job-name=tsnr_roi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/tsnr_roi_%j.out
#SBATCH --error=logs/tsnr_roi_%j.err

cd "${MEMORIA_ROOT:-$PWD}"
# Activate your Python environment here (see requirements.txt)

python code/fmri/analysis/compute_tsnr_roi.py
