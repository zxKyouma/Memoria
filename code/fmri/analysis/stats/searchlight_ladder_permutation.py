#!/usr/bin/env python3
"""
Permutation test for the scale-dependence ladder (persistent trace vs hit).

Computes per-subject hippocampus mean ΔID for each searchlight radius plus ROI-level ΔID,
then tests whether the mean ladder slope is greater than expected under sign-flip permutations.
"""
import os
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def load_roi_mask(subject_id: str, session: str, roi: str, ref_img: nib.Nifti1Image) -> np.ndarray:
    roi_path = Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{session}_space-T1w_desc-{roi}_resampled-betaseries_mask.nii.gz"
    )
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI mask: {roi_path}")
    roi_img = nib.load(str(roi_path))
    if roi_img.shape != ref_img.shape or not np.allclose(roi_img.affine, ref_img.affine):
        roi_img = resample_from_to(roi_img, ref_img, order=0)
    return roi_img.get_fdata() > 0


def load_searchlight_values(
    radius_dir: Path,
    session: str,
    roi: str,
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    maps = sorted(radius_dir.glob("sub-*_desc-ret_miss_recog_correct_minus_ret_hit_id-diff.nii.gz"))
    for path in maps:
        subject = path.name.split("_")[0].replace("sub-", "")
        img = nib.load(str(path))
        roi_mask = load_roi_mask(subject, session, roi, img)
        data = img.get_fdata()
        roi_vals = data[roi_mask]
        roi_vals = roi_vals[np.isfinite(roi_vals)]
        if roi_vals.size == 0:
            continue
        values[subject] = float(np.mean(roi_vals))
    return values


def load_roi_diffs(results_csv: Path, roi: str) -> Dict[str, float]:
    by_subject: Dict[str, Dict[str, float]] = {}
    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["ROI"] != roi:
                continue
            subj = row["Subject"]
            by_subject.setdefault(subj, {})[row["Condition"]] = float(row["Intrinsic_Dim"])
    diffs = {}
    for subj, conds in by_subject.items():
        if "ret_hit" in conds and "ret_miss_recog_correct" in conds:
            diffs[subj] = conds["ret_miss_recog_correct"] - conds["ret_hit"]
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation test for searchlight ladder slope.")
    parser.add_argument("--radius-dirs", nargs="+", required=True, help="Searchlight output dirs per radius.")
    parser.add_argument("--radii", nargs="+", required=True, help="Radius labels matching --radius-dirs.")
    parser.add_argument("--roi", default="hippocampus", help="ROI name.")
    parser.add_argument("--session", default="2", help="Session label for ROI masks.")
    parser.add_argument(
        "--roi-results",
        default="id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned/group_id_full_results_by_condition.csv",
        help="ROI-level results CSV for persistent trace vs hit.",
    )
    parser.add_argument("--permutations", type=int, default=1000, help="Number of sign-flip permutations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/searchlight_radius_ladder",
        help="Output directory for permutation summary.",
    )
    args = parser.parse_args()

    if len(args.radius_dirs) != len(args.radii):
        raise ValueError("--radius-dirs and --radii must have the same length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-subject values for each radius
    per_subject: Dict[str, Dict[str, float]] = {}
    for radius_label, radius_dir in zip(args.radii, args.radius_dirs):
        values = load_searchlight_values(Path(radius_dir), args.session, args.roi)
        for subj, val in values.items():
            per_subject.setdefault(subj, {})[radius_label] = val

    # Add ROI-level diffs
    roi_diffs = load_roi_diffs(Path(args.roi_results), args.roi)
    for subj, val in roi_diffs.items():
        per_subject.setdefault(subj, {})["ROI"] = val

    scales = list(args.radii) + ["ROI"]
    subjects = sorted(
        subj for subj, vals in per_subject.items()
        if all(scale in vals for scale in scales)
    )
    if not subjects:
        raise ValueError("No subjects have complete ladder values for all scales.")

    data = np.array([[per_subject[subj][scale] for scale in scales] for subj in subjects], dtype=float)
    x = np.arange(len(scales), dtype=float)

    # Subject-level slopes
    slopes = []
    for row in data:
        if np.allclose(row, row[0]):
            slopes.append(0.0)
        else:
            slope, _ = np.polyfit(x, row, deg=1)
            slopes.append(float(slope))
    slopes = np.array(slopes, dtype=float)

    observed = float(slopes.mean())

    rng = np.random.default_rng(args.seed)
    more_extreme = 0
    for _ in range(args.permutations):
        signs = rng.choice([-1.0, 1.0], size=len(slopes))
        perm_mean = float((slopes * signs).mean())
        if abs(perm_mean) >= abs(observed):
            more_extreme += 1
    p_value = (more_extreme + 1) / (args.permutations + 1)

    summary_path = output_dir / "ladder_permutation_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["N", "Scales", "MeanSlope", "P_perm"])
        writer.writeheader()
        writer.writerow(
            {
                "N": len(slopes),
                "Scales": ",".join(scales),
                "MeanSlope": observed,
                "P_perm": p_value,
            }
        )

    subject_path = output_dir / "ladder_subject_values.csv"
    with subject_path.open("w", newline="") as handle:
        fieldnames = ["Subject"] + scales + ["Slope"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for subj, row, slope in zip(subjects, data, slopes):
            record = {"Subject": subj, "Slope": slope}
            for scale, val in zip(scales, row):
                record[scale] = val
            writer.writerow(record)

    print(f"Saved permutation summary to {summary_path}")
    print(f"Saved subject ladder values to {subject_path}")


if __name__ == "__main__":
    main()
