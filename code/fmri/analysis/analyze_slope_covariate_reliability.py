#!/usr/bin/env python3
"""Analyze slope differences with reliability as a covariate."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_slopes(slopes_csv: Path, cond_a: str, cond_b: str) -> pd.DataFrame:
    slopes_df = pd.read_csv(slopes_csv)
    if "Condition" not in slopes_df.columns or "Subject" not in slopes_df.columns:
        raise ValueError("Slope CSV must include Subject and Condition columns.")
    slopes_df = slopes_df[slopes_df["Condition"].isin([cond_a, cond_b])].copy()
    slopes_df["Subject"] = slopes_df["Subject"].astype(str)
    return slopes_df


def load_reliability(reliability_csv: Path, cond_a: str, cond_b: str) -> pd.DataFrame:
    rel_df = pd.read_csv(reliability_csv)
    if "Condition" not in rel_df.columns or "Subject" not in rel_df.columns:
        raise ValueError("Reliability CSV must include Subject and Condition columns.")
    rel_df = rel_df[rel_df["Condition"].isin([cond_a, cond_b])].copy()
    rel_df["Subject"] = rel_df["Subject"].astype(str)
    rel_mean = rel_df.groupby(["Subject", "Condition"], as_index=False)["Reliability"].mean()
    return rel_mean


def fit_covariate_model(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) < 3:
        return np.array([float("nan"), float("nan")]), np.array([float("nan"), float("nan")])
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - 2
    if df <= 0:
        return beta, np.array([float("nan"), float("nan")])
    s2 = float((resid @ resid) / df)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjust slope differences for reliability covariate.")
    parser.add_argument("--slopes-csv", required=True, help="Path to subject_condition_slopes.csv")
    parser.add_argument("--reliability-csv", required=True, help="Path to pattern_reliability.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--condition-a", required=True, help="Condition A name")
    parser.add_argument("--condition-b", required=True, help="Condition B name")
    args = parser.parse_args()

    slopes_df = load_slopes(Path(args.slopes_csv), args.condition_a, args.condition_b)
    rel_df = load_reliability(Path(args.reliability_csv), args.condition_a, args.condition_b)

    merged = slopes_df.merge(rel_df, on=["Subject", "Condition"], how="inner")
    if merged.empty:
        raise SystemExit("No merged slope/reliability data available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "slopes_reliability_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged data to: {merged_path}")

    wide_slopes = merged.pivot(index="Subject", columns="Condition", values="Slope")
    wide_rel = merged.pivot(index="Subject", columns="Condition", values="Reliability")
    wide = wide_slopes.join(wide_rel, lsuffix="_Slope", rsuffix="_Reliability")
    slope_cols = [f"{args.condition_a}_Slope", f"{args.condition_b}_Slope"]
    wide = wide.dropna(subset=slope_cols)

    if wide.empty:
        raise SystemExit("No subjects have both conditions for covariate analysis.")

    slope_a = wide[f"{args.condition_a}_Slope"].to_numpy(dtype=float)
    slope_b = wide[f"{args.condition_b}_Slope"].to_numpy(dtype=float)
    rel_a = wide[f"{args.condition_a}_Reliability"].to_numpy(dtype=float)
    rel_b = wide[f"{args.condition_b}_Reliability"].to_numpy(dtype=float)

    slope_diff = slope_b - slope_a
    rel_diff = rel_b - rel_a

    beta, se = fit_covariate_model(slope_diff, rel_diff)
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    summary = pd.DataFrame(
        [
            {
                "ConditionA": args.condition_a,
                "ConditionB": args.condition_b,
                "N": int(len(slope_diff)),
                "MeanSlopeA": float(np.mean(slope_a)),
                "MeanSlopeB": float(np.mean(slope_b)),
                "MeanSlopeDiff": float(np.mean(slope_diff)),
                "MeanReliabilityA": float(np.mean(rel_a)),
                "MeanReliabilityB": float(np.mean(rel_b)),
                "MeanReliabilityDiff": float(np.mean(rel_diff)),
                "AdjIntercept": float(beta[0]),
                "AdjIntercept_SE": float(se[0]),
                "AdjIntercept_CI_Low": float(ci_low[0]),
                "AdjIntercept_CI_High": float(ci_high[0]),
                "ReliabilityBeta": float(beta[1]),
                "ReliabilityBeta_SE": float(se[1]),
                "ReliabilityBeta_CI_Low": float(ci_low[1]),
                "ReliabilityBeta_CI_High": float(ci_high[1]),
            }
        ]
    )

    summary_path = out_dir / "covariate_model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved covariate summary to: {summary_path}")

    wide_out = wide.reset_index()
    wide_out_path = out_dir / "slopes_reliability_wide.csv"
    wide_out.to_csv(wide_out_path, index=False)
    print(f"Saved wide data to: {wide_out_path}")


if __name__ == "__main__":
    main()
