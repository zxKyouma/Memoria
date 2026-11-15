# Memoria

Analysis code for the study "Generative expansion of representational geometry during
memory retrieval".

The project asks how the intrinsic dimensionality of fMRI activity patterns differs
between memory encoding and retrieval along the visual-to-hippocampal hierarchy
(V1, hV4, LOIT, parahippocampal cortex, hippocampus), and how that geometry relates to
recall success and confidence.

## Data

This is a secondary analysis of the 7T fMRI OpenNeuro dataset ds006266 by Tom Willems,
Konstantinos Zervas, Finn Rabe, Andrea Federspiel and Katharina Henke:

- [whole-brain fMRI group](https://doi.org/10.18112/openneuro.ds006266.v1.0.0)

No imaging data are stored in this repository. Download the dataset from OpenNeuro and
point the scripts at your local BIDS directory.

## Repository layout

```
code/fmri/preprocessing/    fMRIPrep, NiBetaSeries single-trial betas, ROI masks, event fixes
code/fmri/analysis/         intrinsic dimensionality, RSA, pattern reliability, searchlights
code/fmri/analysis/stats/   mixed models, slope contrasts, Bayes factors, behaviour links, figures
scripts/                    searchlight batch launchers
scripts/sbatch/             SLURM wrappers, one per analysis and control variant
requirements.txt            Python dependencies
```

Main analysis entry points in `code/fmri/analysis/`:

| Script | Purpose |
|---|---|
| `compute_dimensionality.py` | Voxel-controlled intrinsic dimensionality per ROI, single subject |
| `compute_dimensionality_binned.py` | Dimensionality for encoding vs retrieval, hit vs miss, and confidence bins |
| `compute_dimensionality_group.py` | Group-level dimensionality summary |
| `compute_pattern_reliability.py` | Split-half pattern reliability used as a covariate |
| `compute_rsa.py`, `compute_rsa_discriminability.py` | Representational similarity and discriminability |
| `searchlight_id_single_subject.py`, `searchlight_id_group.py` | Whole-brain dimensionality searchlights |
| `compute_tsnr_roi.py` | Temporal SNR per ROI (signal-quality control) |

Control analyses (alternative estimators, normalisation schemes, voxel and trial
resampling, smoothed vs unsmoothed betas) each have a wrapper in `scripts/sbatch/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Preprocessing additionally needs fMRIPrep and NiBetaSeries (run through Singularity,
see `code/fmri/preprocessing/build_singularity_container.sh`).

## Running

1. Preprocess with the scripts in `code/fmri/preprocessing/` (fMRIPrep, then
   NiBetaSeries, then ROI mask creation and resampling).
2. Compute ROI dimensionality with `compute_dimensionality_binned.py`.
3. Run the statistics in `code/fmri/analysis/stats/`.

The scripts expect to run from the dataset root, with this repository's `code/` and
`scripts/` folders placed inside it. Set `MEMORIA_ROOT` if you run from elsewhere:

```bash
export MEMORIA_ROOT=/path/to/ds006266
mkdir -p logs
sbatch scripts/sbatch/run_dimensionality_binned_enc_vs_ret_cleaned_nosm_sbatch.sh
```

The SLURM wrappers carry resource requests only. Add your own partition and environment
activation lines. Each wrapper can also be run directly with `bash`.

## Credits and licence

Preprocessing scripts in `code/fmri/preprocessing/` are adapted from the original
dataset authors' code (Mirko Bristle and Tom Willems, University of Bern).

Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). See
`code/LICENSE.txt`.

## Citation

If you use this code, please cite the preprint "Generative expansion of representational
geometry during memory retrieval" and the original dataset (ds006266).
