#!/usr/bin/env python3
"""
Plot ID(t) and activation(t) for retRating pilot (high vs low confidence).
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means.append(sample.mean())
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean, ci_low, ci_high


def plot_curves(df: pd.DataFrame, roi: str, value_col: str, out_path: Path, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for bin_name, color in [("high", "#1f77b4"), ("low", "#d62728")]:
        sub = df[(df["ROI"] == roi) & (df["Bin"] == bin_name)]
        times = sorted(sub["Time_s"].unique())
        means = []
        lows = []
        highs = []
        for t in times:
            vals = sub[sub["Time_s"] == t][value_col].to_numpy()
            mean, ci_low, ci_high = bootstrap_ci(vals)
            means.append(mean)
            lows.append(ci_low)
            highs.append(ci_high)
        ax.plot(times, means, label=bin_name, color=color)
        ax.fill_between(times, lows, highs, color=color, alpha=0.2)

    ax.set_xlabel("Time from retRating onset (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{roi} – {ylabel}")
    ax.legend(frameon=False)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retRating time-resolved pilot curves.")
    parser.add_argument("--id-subjects", default="id_analysis_stats/retRating_time_resolved_pilot/time_resolved_id_subjects.csv")
    parser.add_argument("--activation-subjects", default="id_analysis_stats/retRating_time_resolved_pilot/time_resolved_activation_subjects.csv")
    parser.add_argument("--output-dir", default="id_analysis_stats/retRating_time_resolved_pilot")
    parser.add_argument("--rois", nargs="+", default=["phc", "hippocampus"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_df = pd.read_csv(args.id_subjects)
    act_df = pd.read_csv(args.activation_subjects)

    for roi in args.rois:
        plot_curves(
            id_df,
            roi,
            value_col="ID",
            out_path=out_dir / f"id_timecourse_{roi}.png",
            ylabel="TwoNN ID",
        )
        plot_curves(
            act_df,
            roi,
            value_col="Mean_Activation",
            out_path=out_dir / f"mean_activation_timecourse_{roi}.png",
            ylabel="Mean Activation",
        )
        plot_curves(
            act_df,
            roi,
            value_col="RMS_Activation",
            out_path=out_dir / f"rms_activation_timecourse_{roi}.png",
            ylabel="RMS Activation",
        )


if __name__ == "__main__":
    main()
