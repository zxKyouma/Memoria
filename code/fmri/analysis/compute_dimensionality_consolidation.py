#!/usr/bin/env python3
"""
Compute intrinsic dimensionality for encoding vs consolidation time series.

Uses preprocessed BOLD time series in T1w space (fMRIprepNoSbref) and ROI masks.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker
import skdim
from tqdm import tqdm


# --- USER CONFIGURATION ---
BIDS_DIR = "/home/zlow/by84/Brandon/memoria"
SESSION_ID = "1"
ENC_TASK = "forgetexp"
CONS_TASK = "consolidation"
ROI_NAMES_IN_ORDER = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
RANDOM_SEED = 42
MIN_TRS = 20
OUTPUT_DIR = "./id_analysis_results_by_condition_enc_vs_consolidation"
# --- END OF CONFIGURATION ---


def find_subjects() -> List[str]:
    base = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref"
    return sorted(p.name.split("-")[1] for p in base.glob("sub-*") if p.is_dir())


def find_bold(subject_id: str, task: str) -> Path:
    func_dir = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref" / f"sub-{subject_id}" / f"ses-{SESSION_ID}" / "func"
    pattern = f"sub-{subject_id}_ses-{SESSION_ID}_task-{task}_acq-*_space-T1w_desc-preproc_bold.nii.gz"
    matches = sorted(func_dir.glob(pattern))
    if not matches:
        return Path()
    return matches[0]


def roi_mask_path(subject_id: str, roi_name: str) -> Path:
    return Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{SESSION_ID}_space-T1w_desc-{roi_name}_resampled-betaseries_mask.nii.gz"
    )


def compute_roi_voxel_counts(sample_img, subject_id: str) -> Dict[str, int]:
    roi_counts: Dict[str, int] = {}
    for roi_name in ROI_NAMES_IN_ORDER:
        mask_path = roi_mask_path(subject_id, roi_name)
        if not mask_path.exists():
            continue
        try:
            masker = NiftiMasker(mask_img=str(mask_path))
            n_voxels = masker.fit_transform(sample_img).shape[1]
            if n_voxels > 0:
                roi_counts[roi_name] = n_voxels
        except Exception:
            continue
    return roi_counts


def compute_id_for_condition(
    img,
    subject_id: str,
    condition: str,
    min_voxel_count: int,
    n_trs: int,
) -> List[Dict[str, object]]:
    np.random.seed(RANDOM_SEED)
    results: List[Dict[str, object]] = []
    for roi_name in ROI_NAMES_IN_ORDER:
        mask_path = roi_mask_path(subject_id, roi_name)
        if not mask_path.exists():
            continue
        masker = NiftiMasker(mask_img=str(mask_path), standardize=False)
        try:
            voxel_data = masker.fit_transform(img)
            if voxel_data.shape[0] != n_trs:
                voxel_data = voxel_data[:n_trs]
            n_samples, n_features = voxel_data.shape
            voxel_indices = np.arange(n_features)
            subsampled_indices = np.random.choice(voxel_indices, size=min_voxel_count, replace=False)
            subsampled = voxel_data[:, subsampled_indices]
            id_estimator = skdim.id.TwoNN().fit(subsampled)
            intrinsic_dim = id_estimator.dimension_
            results.append(
                {
                    "Subject": subject_id,
                    "Condition": condition,
                    "ROI": roi_name,
                    "Intrinsic_Dim": intrinsic_dim,
                    "Num_TRs": n_samples,
                    "Original_Voxels": n_features,
                    "Used_Voxels": min_voxel_count,
                }
            )
        except Exception as exc:
            print(f"  - ERROR sub-{subject_id} {condition} ROI-{roi_name}: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ID for encoding vs consolidation.")
    parser.add_argument("--min-trs", type=int, default=MIN_TRS, help="Minimum TRs required per condition.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects()
    all_results: List[Dict[str, object]] = []
    excluded: Dict[str, List[str]] = {}

    for subject_id in tqdm(subjects, desc="Processing Subjects"):
        enc_path = find_bold(subject_id, ENC_TASK)
        cons_path = find_bold(subject_id, CONS_TASK)
        if not enc_path.exists():
            excluded.setdefault(subject_id, []).append("missing enc bold")
            continue
        if not cons_path.exists():
            excluded.setdefault(subject_id, []).append("missing consolidation bold")
            continue

        enc_img = load_img(str(enc_path))
        cons_img = load_img(str(cons_path))

        enc_trs = enc_img.shape[-1]
        cons_trs = cons_img.shape[-1]
        target_trs = min(enc_trs, cons_trs)
        if target_trs < args.min_trs:
            excluded.setdefault(subject_id, []).append(f"insufficient TRs ({target_trs})")
            continue

        roi_counts = compute_roi_voxel_counts(enc_img, subject_id)
        if not roi_counts:
            excluded.setdefault(subject_id, []).append("missing ROI masks")
            continue
        min_voxel_count = min(roi_counts.values())

        results = compute_id_for_condition(enc_img, subject_id, "Enc", min_voxel_count, target_trs)
        all_results.extend(results)
        results = compute_id_for_condition(cons_img, subject_id, "Consolidation", min_voxel_count, target_trs)
        all_results.extend(results)

    if not all_results:
        print("No results generated. Check exclusions.")
        return

    results_df = pd.DataFrame(all_results)
    results_df["ROI"] = pd.Categorical(results_df["ROI"], categories=ROI_NAMES_IN_ORDER, ordered=True)
    results_df = results_df.sort_values(["Subject", "Condition", "ROI"])

    full_path = out_dir / "group_id_full_results_by_condition.csv"
    results_df.to_csv(full_path, index=False)
    print(f"Full results saved to: {full_path}")

    group_stats = (
        results_df.groupby(["Condition", "ROI"])["Intrinsic_Dim"]
        .agg(["mean", "std", "sem", "count"])
        .reset_index()
    )
    summary_path = out_dir / "group_id_summary_stats_by_condition.csv"
    group_stats.to_csv(summary_path, index=False)
    print(f"Summary stats saved to: {summary_path}")

    if excluded:
        print("\n--- Exclusions ---")
        for sub_id, reasons in excluded.items():
            print(f"sub-{sub_id}: {', '.join(reasons)}")


if __name__ == "__main__":
    main()
