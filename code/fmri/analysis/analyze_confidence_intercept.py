#!/usr/bin/env python3
"""
Assess absolute ID differences across confidence bins for selected ROIs.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_ROIS = ["LOIT", "phc"]
DEFAULT_CONTRASTS = [("retRating_sure", "retRating_guess")]


def summarize_ids(df: pd.DataFrame, rois: List[str]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for roi in rois:
        roi_df = df[df["ROI"] == roi]
        for condition, cond_df in roi_df.groupby("Condition"):
            n = len(cond_df)
            if n == 0:
                continue
            mean = cond_df["Intrinsic_Dim"].mean()
            sem = cond_df["Intrinsic_Dim"].sem(ddof=1)
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem
            records.append(
                {
                    "ROI": roi,
                    "Condition": condition,
                    "MeanID": float(mean),
                    "SEM": float(sem),
                    "CI_low": float(ci_low),
                    "CI_high": float(ci_high),
                    "N": int(n),
                }
            )
    return pd.DataFrame(records)


def compute_contrast(df: pd.DataFrame, roi: str, cond_a: str, cond_b: str):
    sub_a = df[(df["ROI"] == roi) & (df["Condition"] == cond_a)]
    sub_b = df[(df["ROI"] == roi) & (df["Condition"] == cond_b)]
    map_a = {row["Subject"]: row["Intrinsic_Dim"] for _, row in sub_a.iterrows()}
    map_b = {row["Subject"]: row["Intrinsic_Dim"] for _, row in sub_b.iterrows()}
    common = sorted(set(map_a) & set(map_b))
    if len(common) < 2:
        return None
    diffs = np.array([map_a[sub] - map_b[sub] for sub in common], dtype=float)
    mean = diffs.mean()
    sem = diffs.std(ddof=1) / np.sqrt(len(diffs))
    ci_low = mean - 1.96 * sem
    ci_high = mean + 1.96 * sem
    return {
        "ROI": roi,
        "Contrast": f"{cond_a} - {cond_b}",
        "MeanDiff": float(mean),
        "SEM": float(sem),
        "CI_low": float(ci_low),
        "CI_high": float(ci_high),
        "N": int(len(diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize confidence-bin absolute ID differences.")
    parser.add_argument("--input", required=True, help="Path to group_id_full_results_by_condition.csv")
    parser.add_argument(
        "--rois",
        nargs="*",
        default=DEFAULT_ROIS,
        help="ROIs to analyze (default: LOIT phc)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for summaries",
    )
    parser.add_argument(
        "--contrast",
        nargs="*",
        default=[f"{a}:{b}" for a, b in DEFAULT_CONTRASTS],
        help="Condition contrasts as condA:condB (default: retRating_sure:retRating_guess)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["Subject"] = df["Subject"].astype(str)

    summary_df = summarize_ids(df, args.rois)
    summary_path = out_dir / "confidence_absolute_id_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved mean ID summary to: {summary_path}")

    contrast_records: List[Dict[str, object]] = []
    for roi in args.rois:
        for contrast in args.contrast:
            if ":" not in contrast:
                continue
            cond_a, cond_b = contrast.split(":", 1)
            result = compute_contrast(df, roi, cond_a, cond_b)
            if result:
                contrast_records.append(result)

    contrast_df = pd.DataFrame(contrast_records)
    contrast_path = out_dir / "confidence_absolute_id_contrasts.csv"
    contrast_df.to_csv(contrast_path, index=False)
    print(f"Saved confidence contrasts to: {contrast_path}")


if __name__ == "__main__":
    main()
