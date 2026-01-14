#!/usr/bin/env python3
"""Bin subjects by retrieval reliability and summarize enc vs ret slopes."""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sign_flip_pvalue(deltas: np.ndarray, permutations: int = 5000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    observed = float(np.mean(deltas))
    signs = rng.choice([-1, 1], size=(permutations, len(deltas)))
    null = (signs * deltas).mean(axis=1)
    p_val = (np.sum(np.abs(null) >= abs(observed)) + 1) / (permutations + 1)
    return float(p_val)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(sample.mean())
    lower = np.percentile(boot, 100 * alpha / 2)
    upper = np.percentile(boot, 100 * (1 - alpha / 2))
    return np.array([lower, upper])


def load_slopes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Condition"].isin(["Enc", "ret"])]
    return df


def load_reliability(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Condition"].isin(["Enc", "ret"])]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability-binned enc vs ret slopes.")
    parser.add_argument(
        "--slopes",
        default="id_analysis_stats/enc_vs_ret_cleaned/subject_condition_slopes.csv",
        help="Subject-level slopes CSV for enc vs ret.",
    )
    parser.add_argument(
        "--reliability",
        default="id_analysis_stats/pattern_reliability_enc_vs_ret_cleaned/pattern_reliability.csv",
        help="Split-half reliability CSV for enc vs ret.",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/reliability_binned_enc_vs_ret_cleaned",
        help="Output directory for summaries and plots.",
    )
    parser.add_argument("--bins", type=int, default=3, help="Number of reliability bins.")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slopes = load_slopes(Path(args.slopes))
    reliability = load_reliability(Path(args.reliability))

    rel_summary = (
        reliability.groupby(["Subject", "Condition"])["Reliability"]
        .mean()
        .reset_index()
    )
    ret_rel = rel_summary[rel_summary["Condition"] == "ret"].copy()

    merged = slopes.pivot(index="Subject", columns="Condition", values="Slope").dropna()
    merged = merged.reset_index()
    merged = merged.merge(ret_rel[["Subject", "Reliability"]], on="Subject", how="inner")
    merged["Delta"] = merged["ret"] - merged["Enc"]

    try:
        merged["Reliability_Bin"] = pd.qcut(
            merged["Reliability"], q=args.bins, labels=None, duplicates="drop"
        )
    except ValueError:
        merged["Reliability_Bin"] = pd.cut(
            merged["Reliability"], bins=args.bins, labels=None, include_lowest=True
        )
    merged["Reliability_Bin"] = merged["Reliability_Bin"].astype(str)

    subjects_path = output_dir / "reliability_binned_subjects.csv"
    merged.to_csv(subjects_path, index=False)

    slope_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []
    for bin_name, sub_df in merged.groupby("Reliability_Bin"):
        for condition in ["Enc", "ret"]:
            vals = sub_df[condition].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(vals)
            slope_rows.append(
                {
                    "Reliability_Bin": bin_name,
                    "Condition": condition,
                    "N": len(vals),
                    "MeanSlope": float(np.mean(vals)),
                    "SEM": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                    "CI_low": float(ci_low),
                    "CI_high": float(ci_high),
                }
            )
        deltas = sub_df["Delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas)
        p_perm = sign_flip_pvalue(deltas, permutations=args.permutations, seed=args.seed)
        delta_rows.append(
            {
                "Reliability_Bin": bin_name,
                "N": len(deltas),
                "MeanDelta": float(np.mean(deltas)),
                "SEM": float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else float("nan"),
                "CI_low": float(ci_low),
                "CI_high": float(ci_high),
                "P_perm": p_perm,
            }
        )

    slope_summary = pd.DataFrame(slope_rows)
    delta_summary = pd.DataFrame(delta_rows)
    slope_summary_path = output_dir / "reliability_binned_slope_summary.csv"
    delta_summary_path = output_dir / "reliability_binned_delta_summary.csv"
    slope_summary.to_csv(slope_summary_path, index=False)
    delta_summary.to_csv(delta_summary_path, index=False)

    bins_sorted = sorted(slope_summary["Reliability_Bin"].unique())
    x = np.arange(len(bins_sorted))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for idx, condition in enumerate(["Enc", "ret"]):
        sub = slope_summary[slope_summary["Condition"] == condition].set_index("Reliability_Bin")
        means = [sub.loc[bin_name, "MeanSlope"] for bin_name in bins_sorted]
        sems = [sub.loc[bin_name, "SEM"] for bin_name in bins_sorted]
        ax.bar(x + (idx - 0.5) * width, means, width=width, yerr=sems, capsize=3, label=condition)

    ax.axhline(0, color="0.8", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, bins_sorted)
    ax.set_xlabel("Retrieval reliability bin (mean across ROIs)")
    ax.set_ylabel("Ventral-stream slope")
    ax.set_title("Enc vs Ret slopes by reliability bin")
    ax.legend(frameon=False)
    fig.tight_layout()
    plot_path = output_dir / "reliability_binned_slopes.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved subject table to {subjects_path}")
    print(f"Saved slope summary to {slope_summary_path}")
    print(f"Saved delta summary to {delta_summary_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
