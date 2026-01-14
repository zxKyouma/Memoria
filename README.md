# Project Memoria

## Contents
This repository contains code and data files required to reproduce the results of the study  "Generative expansion of representational geometry during memory retrieval". 

This project is a secondary analysis of the 7T fMRI OpenNeuro dataset ds006266 by Tom Willems, Konstantinos Zervas, Finn Rabe, Andrea Federspiel, and Katharina Henke. The fMRI data are available on OpenNeuro:
- [whole-brain fMRI group](https://doi.org/10.18112/openneuro.ds006266.v1.0.0)

___

## fMRI
### Core pipelines
Core analysis scripts live in `code/fmri/analysis/`, and preprocessing scripts live in `code/fmri/preprocessing/`. Slurm wrappers are in `scripts/sbatch/`.

### Quick start (core results, Slurm)
- `scripts/sbatch/run_dimensionality_binned_enc_vs_ret_cleaned_nosm_sbatch.sh`
- `scripts/sbatch/run_dimensionality_binned_ret_hit_miss_3way_cleaned_controls_nosm_sbatch.sh`
- `scripts/sbatch/run_dimensionality_binned_retRating_confidence_controls_sbatch.sh`
- `scripts/sbatch/run_mean_beta_activation_lmm_enc_vs_ret_cleaned_nosm_sbatch.sh`
- `scripts/sbatch/run_pattern_reliability_enc_vs_ret_cleaned_nosm_sbatch.sh`
- `scripts/sbatch/run_tsnr_nosm_sbatch.sh`

### Requirements (minimal)
- Python 3.8+ with numpy, scipy, pandas, nibabel, nilearn, scikit-dimension (skdim), statsmodels
- Slurm for sbatch-based runs

### Reproduce results (by section)
- 3.1 enc↔ret expansion: `scripts/sbatch/run_dimensionality_binned_enc_vs_ret_cleaned_nosm_sbatch.sh`
- 3.2 controls (reliability/mean‑beta/tSNR): `scripts/sbatch/run_pattern_reliability_enc_vs_ret_cleaned_nosm_sbatch.sh`, `scripts/sbatch/run_mean_beta_activation_lmm_enc_vs_ret_cleaned_nosm_sbatch.sh`, `scripts/sbatch/run_tsnr_nosm_sbatch.sh`
- 3.3 hit/persistent/lost: `scripts/sbatch/run_dimensionality_binned_ret_hit_miss_3way_cleaned_controls_nosm_sbatch.sh`
- 3.3 confidence fingerprint: `scripts/sbatch/run_dimensionality_binned_retRating_confidence_controls_sbatch.sh` + `code/fmri/analysis/stats/confidence_fingerprint_analysis.py`
- 3.4 model comparison: `code/fmri/analysis/stats/compare_model_brain_dimensionality.py`
- 3.5 confidence generalization: same as 3.3 fingerprint, across ROIs (use `--roi` in `confidence_fingerprint_analysis.py`)
