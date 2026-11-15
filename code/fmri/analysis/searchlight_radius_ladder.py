#!/usr/bin/env python3
"""
Summarise persistent-trace searchlight results across radii and plot a ladder plot.
"""
import os
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib.pyplot as plt


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


def summarize_radius(radius_dir: Path, session: str, roi: str) -> Tuple[int, float, float, float]:
    maps = sorted(radius_dir.glob("sub-*_desc-ret_miss_recog_correct_minus_ret_hit_id-diff.nii.gz"))
    values = []
    for path in maps:
        subject = path.name.split("_")[0].replace("sub-", "")
        img = nib.load(str(path))
        roi_mask = load_roi_mask(subject, session, roi, img)
        data = img.get_fdata()
        roi_vals = data[roi_mask]
        roi_vals = roi_vals[np.isfinite(roi_vals)]
        if roi_vals.size == 0:
            continue
        values.append(float(np.mean(roi_vals)))
    if not values:
        return 0, float("nan"), float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    mean, low, high = mean_ci(arr)
    return len(arr), mean, low, high


def summarize_roi_delta(results_csv: Path, roi: str) -> Tuple[int, float, float, float]:
    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        by_subject = {}
        for row in reader:
            if row["ROI"] != roi:
                continue
            subj = row["Subject"]
            by_subject.setdefault(subj, {})[row["Condition"]] = float(row["Intrinsic_Dim"])
    diffs = []
    for subj, conds in by_subject.items():
        if "ret_hit" in conds and "ret_miss_recog_correct" in conds:
            diffs.append(conds["ret_miss_recog_correct"] - conds["ret_hit"])
    if not diffs:
        return 0, float("nan"), float("nan"), float("nan")
    arr = np.array(diffs, dtype=float)
    mean, low, high = mean_ci(arr)
    return len(arr), mean, low, high


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ladder plot for searchlight radii.")
    parser.add_argument("--radius-dirs", nargs="+", required=True, help="Searchlight output dirs per radius.")
    parser.add_argument("--radii", nargs="+", required=True, help="Radius labels matching --radius-dirs.")
    parser.add_argument("--roi", default="hippocampus", help="ROI name for summarising searchlight maps.")
    parser.add_argument("--session", default="2", help="Session label for ROI masks.")
    parser.add_argument(
        "--roi-results",
        default="id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned/group_id_full_results_by_condition.csv",
        help="ROI-level results CSV for persistent trace vs hit.",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/searchlight_radius_ladder",
        help="Output directory for CSV and plot.",
    )
    args = parser.parse_args()

    if len(args.radius_dirs) != len(args.radii):
        raise ValueError("--radius-dirs and --radii must have the same length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for radius_label, radius_dir in zip(args.radii, args.radius_dirs):
        n, mean, low, high = summarize_radius(Path(radius_dir), args.session, args.roi)
        summary_rows.append(
            {"Radius": radius_label, "N": n, "MeanDiff": mean, "CI_low": low, "CI_high": high}
        )

    roi_n, roi_mean, roi_low, roi_high = summarize_roi_delta(Path(args.roi_results), args.roi)
    summary_rows.append(
        {"Radius": "ROI", "N": roi_n, "MeanDiff": roi_mean, "CI_low": roi_low, "CI_high": roi_high}
    )

    csv_path = output_dir / "searchlight_radius_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Radius", "N", "MeanDiff", "CI_low", "CI_high"])
        writer.writeheader()
        writer.writerows(summary_rows)

    labels = [row["Radius"] for row in summary_rows]
    means = np.array([row["MeanDiff"] for row in summary_rows], dtype=float)
    ci_low = np.array([row["CI_low"] for row in summary_rows], dtype=float)
    ci_high = np.array([row["CI_high"] for row in summary_rows], dtype=float)
    yerr = np.vstack([means - ci_low, ci_high - means])

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=yerr, fmt="o-", capsize=4, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ ID (Persistent Trace − Hit)")
    ax.set_xlabel("Searchlight Radius")
    ax.set_title(f"Scale-Dependence of Persistent Trace (ROI: {args.roi})")
    fig.tight_layout()
    plot_path = output_dir / "searchlight_radius_ladder.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved summary to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
