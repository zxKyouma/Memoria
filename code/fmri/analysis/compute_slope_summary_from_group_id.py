#!/usr/bin/env python3
"""
Compute ventral-stream dimensionality slopes from group-level ID results.

This script expects the voxel-controlled intrinsic dimensionality results
produced by `compute_dimensionality_group.py`, and derives:
  1) Per-subject, per-condition slopes across ROIs.
  2) Condition-wise group summaries (mean, SEM, 95% CI, N).

Outputs (relative to project root):
  - id_analysis_stats/subject_condition_slopes.csv
  - id_analysis_stats/slope_summary.csv
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ.get("MEMORIA_ROOT", "."))

# Default input from compute_dimensionality_group.py
GROUP_ID_CSV = PROJECT_ROOT / "id_analysis_results_by_condition_ses-2_forgetexp" / "group_id_full_results_by_condition.csv"

# Default output directory for slope summaries
OUT_DIR = PROJECT_ROOT / "id_analysis_stats"
OUT_DIR.mkdir(exist_ok=True)

# Ventral-stream ROI order used in the ID analysis
ROI_ORDER: List[str] = ["V1", "hV4", "LOIT", "phc", "hippocampus"]


def compute_slopes_per_subject(
    df: pd.DataFrame,
    roi_order: List[str],
    log10: bool = False,
    log_eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Given the full group ID results, compute per-subject, per-condition
    slopes across the ordered ROIs.
    """
    required_cols = {"Subject", "Condition", "ROI", "Intrinsic_Dim"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    # Ensure ROI and Condition are ordered consistently
    df = df.copy()
    df["ROI"] = pd.Categorical(df["ROI"], categories=roi_order, ordered=True)
    df = df.sort_values(["Subject", "Condition", "ROI"])

    records = []

    for (subject, condition), sub_df in df.groupby(["Subject", "Condition"]):
        # Drop any rows with ROIs outside ROI_ORDER or NaN IDs
        sub_df = sub_df.dropna(subset=["Intrinsic_Dim"])
        sub_df = sub_df[sub_df["ROI"].isin(roi_order)]

        if sub_df.empty:
            continue

        # Sort by ROI order and map to x positions
        sub_df = sub_df.sort_values("ROI")
        y = sub_df["Intrinsic_Dim"].to_numpy()
        if log10:
            y = np.log10(np.clip(y, log_eps, None))
        x = np.arange(len(y), dtype=float)

        # Require at least 3 ROIs to fit a meaningful slope
        if len(y) < 3:
            continue

        # Linear fit: y = slope * x + intercept
        slope, intercept = np.polyfit(x, y, deg=1)
        mean_dim = float(np.mean(y))

        records.append(
            {
                "Subject": subject,
                "Condition": condition,
                "Slope": float(slope),
                "Intercept": float(intercept),
                "MeanDim": mean_dim,
                "NumROIs": int(len(y)),
            }
        )

    if not records:
        raise RuntimeError("No subject-level slopes could be computed from the input data.")

    slopes_df = pd.DataFrame(records)

    # Try to give Subject a clean int type if possible
    try:
        slopes_df["Subject"] = slopes_df["Subject"].astype(int)
    except Exception:
        pass

    return slopes_df


def summarize_slopes(slopes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute condition-wise mean slope, SEM, 95% CI, and N.
    """
    summaries = []

    for condition, cond_df in slopes_df.groupby("Condition"):
        n = len(cond_df)
        if n == 0:
            continue

        mean = cond_df["Slope"].mean()
        sem = cond_df["Slope"].sem(ddof=1)

        # Simple normal-approx 95% CI (consistent with previous summary file)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem

        summaries.append(
            {
                "Condition": condition,
                "MeanSlope": float(mean),
                "SEM": float(sem),
                "CI_low": float(ci_low),
                "CI_high": float(ci_high),
                "N": int(n),
            }
        )

    if not summaries:
        raise RuntimeError("No condition-wise summaries could be computed from slopes dataframe.")

    summary_df = pd.DataFrame(summaries)
    # Sort by condition name for readability
    summary_df = summary_df.sort_values("Condition")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ventral-stream dimensionality slopes from group-level ID results."
    )
    parser.add_argument(
        "--input",
        default=str(GROUP_ID_CSV),
        help="Path to group_id_full_results_by_condition.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Output directory for slope summaries",
    )
    parser.add_argument(
        "--roi-order",
        nargs="*",
        default=ROI_ORDER,
        help="Ordered list of ROI names used to compute slopes",
    )
    parser.add_argument(
        "--log10",
        action="store_true",
        help="Apply log10 to Intrinsic_Dim before slope fitting.",
    )
    parser.add_argument(
        "--log-eps",
        type=float,
        default=1e-8,
        help="Epsilon for log transform to avoid log(0).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Group ID CSV not found at {input_path}. "
            "Run compute_dimensionality_group.py first."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_slopes_csv = out_dir / "subject_condition_slopes.csv"
    summary_csv = out_dir / "slope_summary.csv"

    print(f"Loading group ID results from: {input_path}")
    group_df = pd.read_csv(input_path)

    print("Computing subject-level slopes...")
    slopes_df = compute_slopes_per_subject(group_df, args.roi_order, log10=args.log10, log_eps=args.log_eps)
    print(f"Computed slopes for {len(slopes_df)} subject-condition pairs.")

    print(f"Saving subject slopes to: {subject_slopes_csv}")
    slopes_df.to_csv(subject_slopes_csv, index=False)

    print("Computing condition-wise summaries...")
    summary_df = summarize_slopes(slopes_df)
    print(f"Saving slope summary to: {summary_csv}")
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- Subject-level slopes (head) ---")
    print(slopes_df.head())

    print("\n--- Condition-wise slope summary ---")
    print(summary_df)


if __name__ == "__main__":
    main()
