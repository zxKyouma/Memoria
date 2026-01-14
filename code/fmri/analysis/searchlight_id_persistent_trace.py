#!/usr/bin/env python3
"""
Compute ID searchlight maps for persistent-trace vs hit contrast (session 2).

Conditions:
  - ret_hit: ret accuracy correct
  - ret_miss_recog_correct: ret accuracy incorrect + recog accuracy correct
Count-matches trials between conditions per subject.
"""
import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from nilearn import image as nimg
from nilearn.maskers import NiftiMasker
import pandas as pd
from scipy.spatial import cKDTree
import skdim


BIDS_DIR = "/home/zlow/by84/Brandon/memoria"
TASK_NAME = "forgetexp"
RANDOM_SEED = 42

TRIAL_TYPE_RE = re.compile(r"^([A-Za-z]+)(\d+)(?:_([^_]+))?(?:_([^_]+))?$")
DESC_RE = re.compile(r"_desc-([A-Za-z]+)(\d+)_betaseries\.nii\.gz$")


def load_single_volume_img(path: str):
    img = nimg.load_img(path)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        return nimg.index_img(img, 0)
    return img


def parse_trial_type(trial_type: str) -> Tuple[str, str, str, str]:
    match = TRIAL_TYPE_RE.match(trial_type)
    if not match:
        return "", "", "", ""
    stage_raw, item_id, confidence, accuracy = match.groups()
    return stage_raw.lower(), item_id, (confidence or "").lower(), (accuracy or "").lower()


def load_events(subject_id: str, session_label: str) -> pd.DataFrame:
    events_path = Path(BIDS_DIR) / f"sub-{subject_id}" / f"ses-{session_label}" / "func" / (
        f"sub-{subject_id}_ses-{session_label}_task-{TASK_NAME}_events.tsv"
    )
    if not events_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(events_path, sep="\t")
    parsed = df["trial_type"].fillna("").astype(str).apply(parse_trial_type)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["stage", "item_id", "confidence", "accuracy"])
    return parsed_df


def build_recog_accuracy_map(df: pd.DataFrame) -> Dict[str, str]:
    recog_df = df[df["stage"] == "recog"].copy()
    recog_df = recog_df[(recog_df["confidence"] != "noresp") & (recog_df["accuracy"] != "noresp")]
    recog_map: Dict[str, str] = {}
    for _, row in recog_df.iterrows():
        item_id = row["item_id"]
        if item_id in recog_map:
            continue
        if row["accuracy"]:
            recog_map[item_id] = row["accuracy"]
    return recog_map


def filter_items(
    df: pd.DataFrame,
    stage: str,
    accuracy: Optional[str] = None,
    recog_accuracy: Optional[str] = None,
    recog_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    subset = df[df["stage"] == stage].copy()
    subset = subset[(subset["confidence"] != "noresp") & (subset["accuracy"] != "noresp")]
    if accuracy is not None:
        subset = subset[subset["accuracy"] == accuracy]
    items = sorted(set(subset["item_id"].tolist()))
    if recog_accuracy is not None:
        if recog_map is None:
            return []
        items = [item_id for item_id in items if recog_map.get(item_id) == recog_accuracy]
    return items


def build_beta_index(subject_id: str, session_label: str) -> Dict[Tuple[str, str], Path]:
    func_dir = Path(BIDS_DIR) / "derivatives" / "nibetaseries_nosm" / "nibetaseries" / f"sub-{subject_id}" / f"ses-{session_label}" / "func"
    beta_index: Dict[Tuple[str, str], Path] = {}
    for path in func_dir.glob("*.nii.gz"):
        match = DESC_RE.search(path.name)
        if not match:
            continue
        stage, item_id = match.groups()
        beta_index[(stage.lower(), item_id)] = path
    return beta_index


def get_beta_paths(stage: str, item_ids: List[str], beta_index: Dict[Tuple[str, str], Path]) -> List[str]:
    paths = []
    for item_id in item_ids:
        path = beta_index.get((stage, item_id))
        if path is not None:
            paths.append(str(path))
    return paths


def find_brain_mask(subject_id: str, session_label: str) -> str:
    func_dir = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref" / f"sub-{subject_id}" / f"ses-{session_label}" / "func"
    pattern = str(func_dir / f"sub-{subject_id}_ses-{session_label}*_space-T1w_desc-brain_mask.nii.gz")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No brain mask found with pattern: {pattern}")
    return candidates[0]


def load_union_mask(
    subject_id: str,
    session_label: str,
    roi_names: List[str],
    ref_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    masks = []
    for roi_name in roi_names:
        roi_path = Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
            f"sub-{subject_id}_ses-{session_label}_space-T1w_desc-{roi_name}_resampled-betaseries_mask.nii.gz"
        )
        if not roi_path.exists():
            continue
        roi_img = nib.load(str(roi_path))
        if roi_img.shape != ref_img.shape or not np.allclose(roi_img.affine, ref_img.affine):
            roi_img = resample_from_to(roi_img, ref_img, order=0)
        masks.append(roi_img.get_fdata() > 0)
    if not masks:
        return ref_img
    union = np.logical_or.reduce(masks)
    return nib.Nifti1Image(union.astype(np.uint8), ref_img.affine, ref_img.header)


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
    parser = argparse.ArgumentParser(description="Compute persistent-trace vs hit ID searchlight maps.")
    parser.add_argument("subject_id", type=str, help="Subject ID (e.g., 60100).")
    parser.add_argument("--session", default="2", help="Session label (default: 2).")
    parser.add_argument("--radius", type=float, default=3.0, help="Searchlight radius in voxels.")
    parser.add_argument("--min-trials", type=int, default=4, help="Minimum trials per bin.")
    parser.add_argument("--min-voxels", type=int, default=20, help="Minimum voxels per sphere.")
    parser.add_argument("--estimator", choices=["twonn", "mle"], default="twonn", help="ID estimator.")
    parser.add_argument("--mle-k", type=int, default=10, help="MLE neighbor count.")
    parser.add_argument("--max-centers", type=int, default=None, help="Optional cap on centers (debug).")
    parser.add_argument(
        "--mask-rois",
        nargs="*",
        default=None,
        help="ROI names to union as mask (e.g., V1 hV4 LOIT FUS FG1 FG2 FG3 FG4 phc hippocampus).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(BIDS_DIR) / "outputs" / "searchlight_id_maps_persistent"),
        help="Output directory for ID maps.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED + int(args.subject_id))

    events_df = load_events(args.subject_id, args.session)
    if events_df.empty:
        raise FileNotFoundError(f"Missing events for sub-{args.subject_id} ses-{args.session}.")

    recog_map = build_recog_accuracy_map(events_df)
    hit_items = filter_items(events_df, stage="ret", accuracy="correct")
    persistent_items = filter_items(
        events_df,
        stage="ret",
        accuracy="incorrect",
        recog_accuracy="correct",
        recog_map=recog_map,
    )

    beta_index = build_beta_index(args.subject_id, args.session)
    hit_items = [item for item in hit_items if ("ret", item) in beta_index]
    persistent_items = [item for item in persistent_items if ("ret", item) in beta_index]

    target = min(len(hit_items), len(persistent_items))
    if target < args.min_trials:
        raise ValueError(
            f"Too few trials for count-match: hit={len(hit_items)}, persistent={len(persistent_items)} (<{args.min_trials})."
        )

    if len(hit_items) > target:
        hit_items = sorted(rng.choice(hit_items, size=target, replace=False))
    if len(persistent_items) > target:
        persistent_items = sorted(rng.choice(persistent_items, size=target, replace=False))

    betas_hit = get_beta_paths("ret", hit_items, beta_index)
    betas_persistent = get_beta_paths("ret", persistent_items, beta_index)

    print(
        f"Subject {args.subject_id}: ret_hit n={len(betas_hit)}, ret_miss_recog_correct n={len(betas_persistent)}"
    )

    mask_path = find_brain_mask(args.subject_id, args.session)
    ref_mask = nib.load(mask_path)
    if args.mask_rois:
        mask_img = load_union_mask(args.subject_id, args.session, args.mask_rois, ref_mask)
    else:
        mask_img = ref_mask

    id_map_hit = compute_id_map(
        betas_hit,
        mask_img,
        args.radius,
        args.min_trials,
        args.min_voxels,
        args.estimator,
        args.mle_k,
        args.max_centers,
    )
    id_map_persist = compute_id_map(
        betas_persistent,
        mask_img,
        args.radius,
        args.min_trials,
        args.min_voxels,
        args.estimator,
        args.mle_k,
        args.max_centers,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"sub-{args.subject_id}_ses-{args.session}"
    id_img_hit = nimg.new_img_like(mask_img, id_map_hit)
    id_img_persist = nimg.new_img_like(mask_img, id_map_persist)
    diff_img = nimg.new_img_like(mask_img, id_map_persist - id_map_hit)

    id_img_hit.to_filename(out_dir / f"{base}_desc-ret_hit_id-map.nii.gz")
    id_img_persist.to_filename(out_dir / f"{base}_desc-ret_miss_recog_correct_id-map.nii.gz")
    diff_img.to_filename(out_dir / f"{base}_desc-ret_miss_recog_correct_minus_ret_hit_id-diff.nii.gz")

    print(f"Saved ID maps to {out_dir}")


if __name__ == "__main__":
    main()
