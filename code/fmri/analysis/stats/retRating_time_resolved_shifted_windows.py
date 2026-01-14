#!/usr/bin/env python3
"""
Compute shifted-window collapse from time-resolved ID subjects file.
"""
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


EARLY_OFFSETS = [3, 4]
LATE_OFFSETS = [5, 6]


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def sign_flip_pvalue(values: np.ndarray, n_perm: int = 5000, seed: int = 42) -> float:
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
    parser = argparse.ArgumentParser(description="Shifted-window collapse from time-resolved ID.")
    parser.add_argument("--input", default="id_analysis_stats/retRating_time_resolved_pilot/time_resolved_id_subjects.csv")
    parser.add_argument("--output-dir", default="id_analysis_stats/retRating_time_resolved_pilot")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collapse_rows = []
    for (subj, roi, bin_name), sub_df in df.groupby(["Subject", "ROI", "Bin"]):
        early = sub_df[sub_df["TR_offset"].isin(EARLY_OFFSETS)]["ID"].mean()
        late = sub_df[sub_df["TR_offset"].isin(LATE_OFFSETS)]["ID"].mean()
        if np.isnan(early) or np.isnan(late):
            continue
        collapse_rows.append(
            {
                "Subject": subj,
                "ROI": roi,
                "Bin": bin_name,
                "ID_early": float(early),
                "ID_late": float(late),
                "Collapse": float(early - late),
            }
        )

    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df.to_csv(out_dir / "collapse_subjects_shifted.csv", index=False)

    summary = []
    for roi, roi_df in collapse_df.groupby("ROI"):
        high = roi_df[roi_df["Bin"] == "high"].set_index("Subject")
        low = roi_df[roi_df["Bin"] == "low"].set_index("Subject")
        common = high.index.intersection(low.index)
        if common.empty:
            continue
        delta = high.loc[common, "Collapse"] - low.loc[common, "Collapse"]
        mean, ci_low, ci_high = mean_ci(delta.to_numpy())
        p_perm = sign_flip_pvalue(delta.to_numpy(), n_perm=args.permutations)
        summary.append(
            {
                "ROI": roi,
                "N": len(delta),
                "DeltaCollapse_mean": mean,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "p_perm": p_perm,
            }
        )

    pd.DataFrame(summary).to_csv(out_dir / "collapse_summary_shifted.csv", index=False)


if __name__ == "__main__":
    main()
