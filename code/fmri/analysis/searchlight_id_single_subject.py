#!/usr/bin/env python3
"""
Compute an ID searchlight map for a single subject and condition contrast.

Outputs per-condition ID maps and a contrast map (condition_a - condition_b).
"""
import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import nibabel as nib
from nilearn import image as nimg
from nilearn.maskers import NiftiMasker
from scipy.spatial import cKDTree
import skdim


BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
TASK_NAME = "forgetexp"


def load_single_volume_img(path: str):
    img = nimg.load_img(path)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        return nimg.index_img(img, 0)
    return img


def find_brain_mask(subject_id: str, session_label: str) -> str:
    func_dir = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref" / f"sub-{subject_id}" / f"ses-{session_label}" / "func"
    pattern = str(func_dir / f"sub-{subject_id}_ses-{session_label}*_space-T1w_desc-brain_mask.nii.gz")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No brain mask found with pattern: {pattern}")
    return candidates[0]


def find_betas(subject_id: str, session_label: str, condition: str) -> List[str]:
    func_dir = Path(BIDS_DIR) / "derivatives" / "nibetaseries_nosm" / "nibetaseries" / f"sub-{subject_id}" / f"ses-{session_label}" / "func"
    pattern = str(
        func_dir
        / f"sub-{subject_id}_ses-{session_label}_task-{TASK_NAME}_acq-*_space-T1w_desc-{condition}*_betaseries.nii.gz"
    )
    betas = sorted(glob.glob(pattern))
    beta_pattern = re.compile(rf"_desc-{re.escape(condition)}(\d+)_betaseries\.nii\.gz$")
    betas = [path for path in betas if beta_pattern.search(os.path.basename(path))]
    if not betas:
        raise FileNotFoundError(f"No beta series found for {condition} with pattern: {pattern}")
    return betas


def estimate_id(data: np.ndarray, estimator: str, mle_k: int) -> float:
    if estimator == "twonn":
        return float(skdim.id.TwoNN().fit(data).dimension_)
    if estimator == "mle":
        estimator_obj = skdim.id.MLE()
        estimator_obj.fit(data, n_neighbors=mle_k)
        return float(estimator_obj.dimension_)
    raise ValueError(f"Unknown estimator '{estimator}'.")


def compute_id_map(
    beta_paths: List[str],
    mask_img: nib.Nifti1Image,
    radius: float,
    min_trials: int,
    min_voxels: int,
    estimator: str,
    mle_k: int,
    max_centers: Optional[int],
) -> np.ndarray:
    beta_imgs = [load_single_volume_img(path) for path in beta_paths]
    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    voxel_data = masker.fit_transform(beta_imgs)
    n_trials, n_voxels = voxel_data.shape
    if n_trials < min_trials:
        raise ValueError(f"Too few trials: {n_trials} (<{min_trials})")

    mask_data = mask_img.get_fdata() > 0
    coords = np.column_stack(np.where(mask_data))
    if coords.shape[0] != n_voxels:
        raise ValueError(
            f"Mask voxel mismatch: mask has {coords.shape[0]} voxels, data has {n_voxels}."
        )

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)

    if max_centers is not None and max_centers < len(neighbors):
        center_indices = np.linspace(0, len(neighbors) - 1, max_centers, dtype=int)
    else:
        center_indices = range(len(neighbors))

    id_values = np.full(len(neighbors), np.nan, dtype=float)
    for center_idx in center_indices:
        sphere_idx = neighbors[center_idx]
        if len(sphere_idx) < min_voxels:
            continue
        sphere_data = voxel_data[:, sphere_idx]
        try:
            id_values[center_idx] = estimate_id(sphere_data, estimator, mle_k)
        except Exception:
            continue

    id_map = np.zeros(mask_data.shape, dtype=float)
    id_map[mask_data] = id_values
    return id_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ID searchlight maps for a subject.")
    parser.add_argument("subject_id", type=str, help="Subject ID (e.g., 60100).")
    parser.add_argument("--session", default=None, help="Session label (default: 2). Use if A and B are in same session.")
    parser.add_argument("--session-a", default=None, help="Session label for Condition A.")
    parser.add_argument("--session-b", default=None, help="Session label for Condition B.")
    parser.add_argument("--condition-a", default="retRatingFeedback", help="Condition A.")
    parser.add_argument("--condition-b", default="ret", help="Condition B.")
    parser.add_argument("--radius", type=float, default=3.0, help="Searchlight radius in voxels.")
    parser.add_argument("--min-trials", type=int, default=10, help="Minimum trials required.")
    parser.add_argument("--min-voxels", type=int, default=20, help="Minimum voxels per sphere.")
    parser.add_argument("--estimator", choices=["twonn", "mle"], default="twonn", help="ID estimator.")
    parser.add_argument("--mle-k", type=int, default=10, help="MLE neighbor count.")
    parser.add_argument("--max-centers", type=int, default=None, help="Optional cap on centers (debug).")
    parser.add_argument(
        "--output-dir",
        default=str(Path(BIDS_DIR) / "outputs" / "searchlight_id_maps"),
        help="Output directory for ID maps.",
    )
    args = parser.parse_args()

    # Determine sessions
    sess_a = args.session_a or args.session or "2"
    sess_b = args.session_b or args.session or "2"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use session A's mask as reference
    mask_path = find_brain_mask(args.subject_id, sess_a)
    mask_img = nib.load(mask_path)

    betas_a = find_betas(args.subject_id, sess_a, args.condition_a)
    betas_b = find_betas(args.subject_id, sess_b, args.condition_b)

    print(f"Subject {args.subject_id}: {args.condition_a} (ses-{sess_a}) n={len(betas_a)}, {args.condition_b} (ses-{sess_b}) n={len(betas_b)}")
    print(f"Mask: {mask_path}")

    id_map_a = compute_id_map(
        betas_a,
        mask_img,
        args.radius,
        args.min_trials,
        args.min_voxels,
        args.estimator,
        args.mle_k,
        args.max_centers,
    )
    id_map_b = compute_id_map(
        betas_b,
        mask_img,
        args.radius,
        args.min_trials,
        args.min_voxels,
        args.estimator,
        args.mle_k,
        args.max_centers,
    )

    id_img_a = nimg.new_img_like(mask_img, id_map_a)
    id_img_b = nimg.new_img_like(mask_img, id_map_b)
    diff_img = nimg.new_img_like(mask_img, id_map_a - id_map_b)

    base = f"sub-{args.subject_id}_ses-{args.session}"
    id_img_a.to_filename(out_dir / f"{base}_desc-{args.condition_a}_id-map.nii.gz")
    id_img_b.to_filename(out_dir / f"{base}_desc-{args.condition_b}_id-map.nii.gz")
    diff_img.to_filename(out_dir / f"{base}_desc-{args.condition_a}_minus_{args.condition_b}_id-diff.nii.gz")

    print(f"Saved ID maps to {out_dir}")


if __name__ == "__main__":
    main()
