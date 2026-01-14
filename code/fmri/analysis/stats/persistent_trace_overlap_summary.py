#!/usr/bin/env python3
"""
Summarise hippocampal overlap between ret_hit and ret_miss_recog_correct.

Uses discriminability_results.csv to extract the between-condition correlation.
"""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise persistent-trace overlap in an ROI.")
    parser.add_argument(
        "--discriminability-results",
        default="id_analysis_stats/discriminability_ret_hit_miss_3way_cleaned/discriminability_results.csv",
        help="Path to discriminability_results.csv",
    )
    parser.add_argument("--roi", default="hippocampus", help="ROI name to summarise.")
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/persistent_trace_overlap",
        help="Output directory for summary CSV.",
    )
    args = parser.parse_args()

    rows = []
    with open(args.discriminability_results, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["ROI"] != args.roi:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found for ROI {args.roi}")

    by_subject = {}
    for row in rows:
        subj = row["Subject"]
        between = float(row["Between"])
        # Between is identical for both conditions; keep one.
        by_subject.setdefault(subj, between)

    values = np.array(list(by_subject.values()), dtype=float)
    mean, low, high = mean_ci(values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "persistent_trace_overlap_summary.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ROI", "N", "MeanBetween", "CI_low", "CI_high"])
        writer.writeheader()
        writer.writerow(
            {
                "ROI": args.roi,
                "N": int(len(values)),
                "MeanBetween": mean,
                "CI_low": low,
                "CI_high": high,
            }
        )

    print(f"Saved overlap summary to {out_path}")


if __name__ == "__main__":
    main()
