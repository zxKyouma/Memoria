#!/usr/bin/env python3
"""
Group-level summary for ID searchlight maps (contrast maps).
Computes mean and t-statistic maps; optional sign-flip permutation p-map.
"""
import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.stats import ttest_1samp


def load_maps(pattern: str) -> List[nib.Nifti1Image]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No maps found with pattern: {pattern}")
    imgs = [nib.load(path) for path in paths]
    ref = imgs[0]
    aligned = []
    for img in imgs:
        if img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
            img = resample_from_to(img, ref, order=1)
        aligned.append(img)
    return aligned


def sign_flip_pvals(data: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_subj, n_vox = data.shape
    observed = data.mean(axis=0)
    more_extreme = np.zeros(n_vox, dtype=int)
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_subj)
        perm_mean = (data * signs[:, None]).mean(axis=0)
        more_extreme += (np.abs(perm_mean) >= np.abs(observed))
    return (more_extreme + 1) / (n_perm + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-level ID searchlight summary.")
    parser.add_argument(
        "--input-pattern",
        default="outputs/searchlight_id_maps/sub-*_ses-2_desc-retRatingFeedback_minus_ret_id-diff.nii.gz",
        help="Glob pattern for subject contrast maps.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/searchlight_id_maps/group",
        help="Output directory for group maps.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Number of sign-flip permutations (0 = skip).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutations.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imgs = load_maps(args.input_pattern)
    data_4d = np.stack([img.get_fdata() for img in imgs], axis=0)
    n_subj = data_4d.shape[0]

    mean_map = data_4d.mean(axis=0)
    t_stat, p_vals = ttest_1samp(data_4d, popmean=0.0, axis=0, nan_policy="omit")
    t_stat = np.nan_to_num(t_stat, nan=0.0, posinf=0.0, neginf=0.0)
    p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)

    ref = imgs[0]
    nib.Nifti1Image(mean_map, ref.affine, ref.header).to_filename(output_dir / "group_mean_id_diff.nii.gz")
    nib.Nifti1Image(t_stat, ref.affine, ref.header).to_filename(output_dir / "group_t_id_diff.nii.gz")
    nib.Nifti1Image(p_vals, ref.affine, ref.header).to_filename(output_dir / "group_p_id_diff.nii.gz")

    print(f"Saved group mean/t/p maps for n={n_subj} to {output_dir}")

    if args.permutations > 0:
        data_2d = data_4d.reshape(n_subj, -1)
        p_perm = sign_flip_pvals(data_2d, args.permutations, args.seed)
        p_perm_map = p_perm.reshape(mean_map.shape)
        nib.Nifti1Image(p_perm_map, ref.affine, ref.header).to_filename(output_dir / "group_pperm_id_diff.nii.gz")
        print(f"Saved sign-flip p-map with {args.permutations} permutations.")


if __name__ == "__main__":
    main()
