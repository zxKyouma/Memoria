#!/usr/bin/env python3
"""
Summarize LOIT -> PHC -> Hippocampus ID drops from group-level results.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_drops(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (subject, condition), sub_df in df.groupby(["Subject", "Condition"]):
        roi_map = {row["ROI"]: row["Intrinsic_Dim"] for _, row in sub_df.iterrows()}
        if not all(roi in roi_map for roi in ("LOIT", "phc", "hippocampus")):
            continue
        loit = float(roi_map["LOIT"])
        phc = float(roi_map["phc"])
        hpc = float(roi_map["hippocampus"])
        records.append(
            {
                "Subject": subject,
                "Condition": condition,
                "Drop_LOIT_PHC": loit - phc,
                "Drop_PHC_HPC": phc - hpc,
                "Drop_LOIT_HPC": loit - hpc,
            }
        )
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for condition, sub_df in df.groupby("Condition"):
        n = len(sub_df)
        if n == 0:
            continue
        for metric in ["Drop_LOIT_PHC", "Drop_PHC_HPC", "Drop_LOIT_HPC"]:
            mean = sub_df[metric].mean()
            sem = sub_df[metric].sem(ddof=1)
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem
            summaries.append(
                {
                    "Condition": condition,
                    "Metric": metric,
                    "Mean": float(mean),
                    "SEM": float(sem),
                    "CI_low": float(ci_low),
                    "CI_high": float(ci_high),
                    "N": int(n),
                }
            )
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LOIT/PHC/HC ID drops.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to group_id_full_results_by_condition.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for drop summaries.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No data found in {input_path}")

    drops_df = compute_drops(df)
    drops_path = out_dir / "bottleneck_subject_drops.csv"
    drops_df.to_csv(drops_path, index=False)
    print(f"Saved subject-level drops to: {drops_path}")

    summary_df = summarize(drops_df)
    summary_path = out_dir / "bottleneck_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved drop summaries to: {summary_path}")


if __name__ == "__main__":
    main()
