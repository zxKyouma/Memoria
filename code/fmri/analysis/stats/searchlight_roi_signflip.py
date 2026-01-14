#!/usr/bin/env python3
"""
ROI-level sign-flip tests on persistent-trace searchlight maps.

For each radius and ROI, compute per-subject mean ΔID within the ROI mask,
then run a sign-flip permutation test on the subject means.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


BIDS_DIR = Path("/home/zlow/by84/Brandon/memoria")


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def load_roi_mask(subject_id: str, session: str, roi: str, ref_img: nib.Nifti1Image) -> np.ndarray:
    roi_path = BIDS_DIR / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{session}_space-T1w_desc-{roi}_resampled-betaseries_mask.nii.gz"
    )
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI mask: {roi_path}")
    roi_img = nib.load(str(roi_path))
    if roi_img.shape != ref_img.shape or not np.allclose(roi_img.affine, ref_img.affine):
        roi_img = resample_from_to(roi_img, ref_img, order=0)
    return roi_img.get_fdata() > 0


def sign_flip_pvalue(values: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    more_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(values))
        perm_mean = float((values * signs).mean())
        if abs(perm_mean) >= abs(observed):
            more_extreme += 1
    return (more_extreme + 1) / (n_perm + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI sign-flip tests for persistent-trace searchlight maps.")
    parser.add_argument(
        "--radius-dirs",
        nargs="+",
        required=True,
        help="Searchlight output dirs per radius.",
    )
    parser.add_argument(
        "--radii",
        nargs="+",
        required=True,
        help="Radius labels matching --radius-dirs (e.g., r3 r5 r7).",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        default=["hippocampus", "phc", "LOIT", "V1", "A1", "M1", "S1", "MT"],
        help="ROI names.",
    )
    parser.add_argument("--session", default="2", help="Session label for ROI masks.")
    parser.add_argument("--permutations", type=int, default=5000, help="Sign-flip permutations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/searchlight_radius_group",
        help="Output directory for ROI summaries.",
    )
    parser.add_argument(
        "--pattern",
        default="sub-*_ses-2_desc-ret_miss_recog_correct_minus_ret_hit_id-diff.nii.gz",
        help="Map filename pattern inside each radius directory.",
    )
    args = parser.parse_args()

    if len(args.radius_dirs) != len(args.radii):
        raise ValueError("--radius-dirs and --radii must have the same length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    subject_rows = []

    for radius_label, radius_dir in zip(args.radii, args.radius_dirs):
        radius_dir = Path(radius_dir)
        maps = sorted(radius_dir.glob(args.pattern))
        if not maps:
            raise FileNotFoundError(f"No maps found in {radius_dir} with pattern {args.pattern}")

        values_by_roi: Dict[str, Dict[str, float]] = {roi: {} for roi in args.rois}

        for path in maps:
            subject = path.name.split("_")[0].replace("sub-", "")
            img = nib.load(str(path))
            data = img.get_fdata()
            finite_mask = np.isfinite(data)

            for roi in args.rois:
                try:
                    roi_mask = load_roi_mask(subject, args.session, roi, img)
                except FileNotFoundError:
                    continue
                roi_mask = roi_mask & finite_mask
                if not roi_mask.any():
                    continue
                values_by_roi[roi][subject] = float(np.nanmean(data[roi_mask]))

        for roi, subj_map in values_by_roi.items():
            subjects = sorted(subj_map.keys())
            if not subjects:
                continue
            values = np.array([subj_map[s] for s in subjects], dtype=float)
            mean, ci_low, ci_high = mean_ci(values)
            p_perm = sign_flip_pvalue(values, args.permutations, args.seed)
            pos_n = int((values > 0).sum())
            neg_n = int((values < 0).sum())
            summary_rows.append(
                {
                    "radius": radius_label,
                    "roi": roi,
                    "N": len(values),
                    "mean_diff": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "pos_n": pos_n,
                    "neg_n": neg_n,
                    "p_perm": p_perm,
                }
            )
            for subject, val in subj_map.items():
                subject_rows.append(
                    {
                        "radius": radius_label,
                        "roi": roi,
                        "Subject": subject,
                        "mean_diff": val,
                    }
                )

    summary_path = output_dir / "persistent_trace_group_roi_signflip_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["radius", "roi", "N", "mean_diff", "ci_low", "ci_high", "pos_n", "neg_n", "p_perm"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    subject_path = output_dir / "persistent_trace_group_roi_signflip_subjects.csv"
    with subject_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["radius", "roi", "Subject", "mean_diff"],
        )
        writer.writeheader()
        writer.writerows(subject_rows)

    print(f"Saved ROI sign-flip summary to {summary_path}")
    print(f"Saved ROI subject means to {subject_path}")


if __name__ == "__main__":
    main()
