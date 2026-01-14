#!/usr/bin/env python3
"""Analyze mean ID differences with reliability as a covariate."""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROI_ORDER_DEFAULT = ["V1", "hV4", "LOIT", "phc", "hippocampus"]


def load_id_means(id_csv: Path, roi_list: List[str], cond_a: str, cond_b: str) -> pd.DataFrame:
    id_df = pd.read_csv(id_csv)
    id_df = id_df[id_df["ROI"].isin(roi_list)].copy()
    id_df = id_df[id_df["Condition"].isin([cond_a, cond_b])].copy()
    id_df["Subject"] = id_df["Subject"].astype(str)
    id_mean = (
        id_df.groupby(["Subject", "Condition"], as_index=False)["Intrinsic_Dim"]
        .mean()
        .rename(columns={"Intrinsic_Dim": "Mean_ID"})
    )
    return id_mean


def load_reliability(reliability_csv: Path, cond_a: str, cond_b: str) -> pd.DataFrame:
    rel_df = pd.read_csv(reliability_csv)
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
    parser = argparse.ArgumentParser(description="Adjust mean ID differences for reliability covariate.")
    parser.add_argument("--id-csv", required=True, help="Path to group_id_full_results_by_condition.csv")
    parser.add_argument("--reliability-csv", required=True, help="Path to pattern_reliability.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--condition-a", required=True, help="Condition A name")
    parser.add_argument("--condition-b", required=True, help="Condition B name")
    parser.add_argument(
        "--roi-list",
        nargs="+",
        default=ROI_ORDER_DEFAULT,
        help="ROI list used to compute mean ID.",
    )
    args = parser.parse_args()

    id_mean = load_id_means(Path(args.id_csv), args.roi_list, args.condition_a, args.condition_b)
    rel_mean = load_reliability(Path(args.reliability_csv), args.condition_a, args.condition_b)

    merged = id_mean.merge(rel_mean, on=["Subject", "Condition"], how="inner")
    if merged.empty:
        raise SystemExit("No merged ID/reliability data available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "id_reliability_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged data to: {merged_path}")

    wide_id = merged.pivot(index="Subject", columns="Condition", values="Mean_ID")
    wide_rel = merged.pivot(index="Subject", columns="Condition", values="Reliability")
    wide = wide_id.join(wide_rel, lsuffix="_ID", rsuffix="_Reliability")
    id_cols = [f"{args.condition_a}_ID", f"{args.condition_b}_ID"]
    wide = wide.dropna(subset=id_cols)

    if wide.empty:
        raise SystemExit("No subjects have both conditions for covariate analysis.")

    id_a = wide[f"{args.condition_a}_ID"].to_numpy(dtype=float)
    id_b = wide[f"{args.condition_b}_ID"].to_numpy(dtype=float)
    rel_a = wide[f"{args.condition_a}_Reliability"].to_numpy(dtype=float)
    rel_b = wide[f"{args.condition_b}_Reliability"].to_numpy(dtype=float)

    id_diff = id_b - id_a
    rel_diff = rel_b - rel_a

    beta, se = fit_covariate_model(id_diff, rel_diff)
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    summary = pd.DataFrame(
        [
            {
                "ConditionA": args.condition_a,
                "ConditionB": args.condition_b,
                "N": int(len(id_diff)),
                "MeanIDA": float(np.mean(id_a)),
                "MeanIDB": float(np.mean(id_b)),
                "MeanIDDiff": float(np.mean(id_diff)),
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
    wide_out_path = out_dir / "id_reliability_wide.csv"
    wide_out.to_csv(wide_out_path, index=False)
    print(f"Saved wide data to: {wide_out_path}")


if __name__ == "__main__":
    main()
