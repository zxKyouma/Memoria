#!/usr/bin/env python3
"""
Summarize within-subject ID trends across retRating confidence bins.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CONFIDENCE_MAP: Dict[str, int] = {
    "guess": 1,
    "unsure": 2,
    "sure": 3,
}


def parse_confidence(condition: str) -> int:
    parts = condition.split("_")
    if len(parts) < 2:
        return 0
    return CONFIDENCE_MAP.get(parts[-1], 0)


def compute_trend_slopes(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for (subject, roi), sub_df in df.groupby(["Subject", "ROI"]):
        sub_df = sub_df.copy()
        sub_df["ConfidenceLevel"] = sub_df["Condition"].apply(parse_confidence)
        sub_df = sub_df[sub_df["ConfidenceLevel"] > 0]
        if sub_df.empty:
            continue
        x = sub_df["ConfidenceLevel"].to_numpy(dtype=float)
        y = sub_df["Intrinsic_Dim"].to_numpy(dtype=float)
        if len(x) < 2:
            continue
        slope, intercept = np.polyfit(x, y, deg=1)
        records.append(
            {
                "Subject": subject,
                "ROI": roi,
                "Slope": float(slope),
                "Intercept": float(intercept),
                "NumBins": int(len(x)),
            }
        )
    return pd.DataFrame(records)


def summarize_slopes(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for roi, sub_df in df.groupby("ROI"):
        n = len(sub_df)
        if n == 0:
            continue
        mean = sub_df["Slope"].mean()
        sem = sub_df["Slope"].sem(ddof=1)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
        summaries.append(
            {
                "ROI": roi,
                "MeanSlope": float(mean),
                "SEM": float(sem),
                "CI_low": float(ci_low),
                "CI_high": float(ci_high),
                "N": int(n),
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ID trends across confidence bins.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to group_id_full_results_by_condition.csv from retRating confidence analysis.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for trend summaries.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No data found in {input_path}")

    trend_df = compute_trend_slopes(df)
    trend_path = out_dir / "confidence_trend_subject_slopes.csv"
    trend_df.to_csv(trend_path, index=False)
    print(f"Saved subject-level trends to: {trend_path}")

    summary_df = summarize_slopes(trend_df)
    summary_path = out_dir / "confidence_trend_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved trend summaries to: {summary_path}")


if __name__ == "__main__":
    main()
