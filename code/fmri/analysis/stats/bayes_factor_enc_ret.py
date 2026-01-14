#!/usr/bin/env python3
"""Compute Bayesian evidence for enc vs ret slope differences (paired)."""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def cauchy_pdf(x: float, r: float) -> float:
    return 1.0 / (np.pi * r * (1.0 + (x / r) ** 2))


def bf10_ttest(
    t_stat: float,
    n: int,
    r: float = 0.707,
    max_delta: float = 50.0,
    grid_points: int = 5001,
) -> float:
    """Jeffreys–Zellner–Siow Bayes factor for a one-sample t-test.

    Uses a Cauchy(0, r) prior over standardized effect size and integrates the
    noncentral t likelihood over the prior.
    """
    if n < 2:
        return float("nan")
    df = n - 1
    f0 = stats.t.pdf(t_stat, df)
    if f0 == 0:
        return float("nan")

    deltas = np.linspace(-max_delta, max_delta, grid_points)
    ncp = deltas * np.sqrt(n)
    likelihood = stats.nct.pdf(t_stat, df, ncp)
    prior = cauchy_pdf(deltas, r)
    m1 = float(np.trapezoid(likelihood * prior, deltas))
    return float(m1 / f0)


def load_slopes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Condition"].isin(["Enc", "ret"])]
    return df


def compute_delta(df: pd.DataFrame) -> np.ndarray:
    wide = df.pivot(index="Subject", columns="Condition", values="Slope").dropna()
    return (wide["ret"] - wide["Enc"]).to_numpy(dtype=float)


def summarize(deltas: np.ndarray, r: float) -> Dict[str, float]:
    n = len(deltas)
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
    bf10 = bf10_ttest(float(t_stat), n, r=r)
    return {
        "N": n,
        "Mean_Delta": mean,
        "SD_Delta": sd,
        "t": float(t_stat),
        "df": n - 1,
        "p_value": float(p_val),
        "BF10": bf10,
        "BF01": float(1.0 / bf10) if bf10 not in (0.0, np.nan) else float("nan"),
        "r_scale": r,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayes factors for enc vs ret slope deltas.")
    parser.add_argument(
        "--raw-slopes",
        default="id_analysis_stats/enc_vs_ret_cleaned/subject_condition_slopes.csv",
        help="Subject-level slopes CSV (raw ID).",
    )
    parser.add_argument(
        "--log10-slopes",
        default="id_analysis_stats/enc_vs_ret_cleaned_log10/subject_condition_slopes.csv",
        help="Subject-level slopes CSV (log10 ID).",
    )
    parser.add_argument(
        "--r-scale",
        type=float,
        default=0.707,
        help="Cauchy prior scale for effect size.",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/bayes_factor_enc_vs_ret",
        help="Output directory for summary CSV.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, path in [("raw", Path(args.raw_slopes)), ("log10", Path(args.log10_slopes))]:
        df = load_slopes(path)
        deltas = compute_delta(df)
        stats_row = summarize(deltas, r=args.r_scale)
        stats_row["Analysis"] = label
        stats_row["Input"] = str(path)
        rows.append(stats_row)

    summary = pd.DataFrame(rows)
    out_path = output_dir / "bayes_factor_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved Bayes factor summary to {out_path}")


if __name__ == "__main__":
    main()
