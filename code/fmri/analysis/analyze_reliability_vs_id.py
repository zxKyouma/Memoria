#!/usr/bin/env python3
"""Summarize split-half reliability vs ID and test reliability-matched subsets."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROI_ORDER_DEFAULT = ["V1", "hV4", "LOIT", "phc", "hippocampus"]


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def compute_slopes(df: pd.DataFrame, roi_order: List[str]) -> pd.DataFrame:
    roi_index = {roi: idx for idx, roi in enumerate(roi_order)}
    records: List[Dict[str, object]] = []
    for (subject, condition), sub_df in df.groupby(["Subject", "Condition"]):
        sub_df = sub_df[sub_df["ROI"].isin(roi_index)]
        if len(sub_df) < 3:
            continue
        sub_df = sub_df.assign(ROI_index=sub_df["ROI"].map(roi_index))
        sub_df = sub_df.sort_values("ROI_index")
        x = sub_df["ROI_index"].to_numpy(dtype=float)
        y = sub_df["Intrinsic_Dim"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        records.append(
            {
                "Subject": subject,
                "Condition": condition,
                "Slope": float(slope),
                "Intercept": float(intercept),
            }
        )
    return pd.DataFrame(records)


def summarize_slopes(slopes_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for condition, sub_df in slopes_df.groupby("Condition"):
        values = sub_df["Slope"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        ci = 1.96 * sem if sem == sem else float("nan")
        rows.append(
            {
                "Condition": condition,
                "MeanSlope": mean,
                "SEM": sem,
                "CI_Low": mean - ci if ci == ci else float("nan"),
                "CI_High": mean + ci if ci == ci else float("nan"),
                "N": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reliability vs ID and reliability-matched slopes.")
    parser.add_argument("--reliability-csv", required=True, help="Path to pattern_reliability.csv")
    parser.add_argument("--id-csv", required=True, help="Path to group_id_full_results_by_condition.csv")
    parser.add_argument("--output-dir", default="./id_analysis_stats/reliability_vs_id", help="Output directory")
    parser.add_argument("--condition-a", default="Enc", help="Baseline condition name")
    parser.add_argument("--condition-b", default="ret", help="Comparison condition name")
    parser.add_argument("--threshold", type=float, default=0.02, help="Max |reliability diff| for matched subset")
    parser.add_argument(
        "--roi-order",
        nargs="+",
        default=ROI_ORDER_DEFAULT,
        help="Ordered ROI list used for slope computation",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rel_df = pd.read_csv(args.reliability_csv)
    id_df = pd.read_csv(args.id_csv)

    merged = rel_df.merge(
        id_df,
        on=["Subject", "Condition", "ROI"],
        how="inner",
        suffixes=("_rel", "_id"),
    )

    corr_rows: List[Dict[str, object]] = []
    for (condition, roi), sub_df in merged.groupby(["Condition", "ROI"]):
        r = pearsonr(
            sub_df["Reliability"].to_numpy(dtype=float),
            sub_df["Intrinsic_Dim"].to_numpy(dtype=float),
        )
        corr_rows.append(
            {
                "Condition": condition,
                "ROI": roi,
                "Reliability_ID_r": r,
                "N": int(len(sub_df)),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_path = out_dir / "reliability_id_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved reliability-ID correlations to: {corr_path}")

    rel_mean = (
        rel_df.groupby(["Subject", "Condition"])["Reliability"].mean().reset_index()
    )
    rel_wide = rel_mean.pivot(index="Subject", columns="Condition", values="Reliability")
    if args.condition_a not in rel_wide.columns or args.condition_b not in rel_wide.columns:
        raise ValueError(
            f"Missing conditions in reliability data: {args.condition_a} or {args.condition_b}"
        )
    rel_wide = rel_wide.dropna(subset=[args.condition_a, args.condition_b])
    rel_wide["Reliability_Diff"] = rel_wide[args.condition_b] - rel_wide[args.condition_a]
    matched = rel_wide[np.abs(rel_wide["Reliability_Diff"]) <= args.threshold]

    matched_path = out_dir / "reliability_matched_subjects.csv"
    matched.reset_index().to_csv(matched_path, index=False)
    print(f"Saved reliability-matched subjects to: {matched_path}")

    matched_subjects = set(matched.index.astype(str))
    matched_id = id_df[id_df["Subject"].astype(str).isin(matched_subjects)]
    matched_id = matched_id[matched_id["Condition"].isin([args.condition_a, args.condition_b])]

    slopes_df = compute_slopes(matched_id, args.roi_order)
    slopes_path = out_dir / "reliability_matched_slopes.csv"
    slopes_df.to_csv(slopes_path, index=False)
    print(f"Saved reliability-matched slopes to: {slopes_path}")

    if slopes_df.empty or "Condition" not in slopes_df.columns:
        print("No reliability-matched slopes available to summarize.")
        return

    summary_df = summarize_slopes(slopes_df)
    summary_path = out_dir / "reliability_matched_slope_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved reliability-matched slope summary to: {summary_path}")

    if not slopes_df.empty:
        slope_map = {
            cond: dict(zip(df["Subject"], df["Slope"]))
            for cond, df in slopes_df.groupby("Condition")
        }
        if args.condition_a in slope_map and args.condition_b in slope_map:
            common = sorted(set(slope_map[args.condition_a]) & set(slope_map[args.condition_b]))
            if common:
                diffs = np.array(
                    [slope_map[args.condition_b][sub] - slope_map[args.condition_a][sub] for sub in common],
                    dtype=float,
                )
                mean_diff = float(diffs.mean())
                sem = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
                ci = 1.96 * sem if sem == sem else float("nan")
                contrast = pd.DataFrame(
                    [
                        {
                            "ConditionA": args.condition_a,
                            "ConditionB": args.condition_b,
                            "MeanDelta": mean_diff,
                            "SEM": sem,
                            "CI_Low": mean_diff - ci if ci == ci else float("nan"),
                            "CI_High": mean_diff + ci if ci == ci else float("nan"),
                            "N": int(len(diffs)),
                        }
                    ]
                )
                contrast_path = out_dir / "reliability_matched_slope_contrast.csv"
                contrast.to_csv(contrast_path, index=False)
                print(f"Saved reliability-matched slope contrast to: {contrast_path}")


if __name__ == "__main__":
    main()
