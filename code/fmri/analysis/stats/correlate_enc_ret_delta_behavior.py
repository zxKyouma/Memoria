#!/usr/bin/env python3
"""
Correlate enc->ret delta ID with behavioral accuracy by ROI.
"""
import argparse
import csv
from collections import defaultdict
from math import atanh, tanh, sqrt
from pathlib import Path
from typing import Dict, List

from scipy import stats


def load_behavior(path: Path) -> Dict[str, Dict[str, float]]:
    behavior = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            behavior[row["Subject"]] = {
                "Ret_Accuracy": float(row["Ret_Accuracy"]),
                "Recog_Accuracy": float(row["Recog_Accuracy"]),
            }
    return behavior


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-wise correlations for enc->ret delta ID.")
    parser.add_argument(
        "--id-results",
        default="id_analysis_results_by_condition_enc_vs_ret_cleaned_nosm/group_id_full_results_by_condition.csv",
        help="Path to enc/ret group ID results.",
    )
    parser.add_argument(
        "--behavior",
        default="id_analysis_stats/confidence_fingerprint_nosm_MATCHED/confidence_fingerprint_subjects.csv",
        help="Path to subject-level behavioral metrics (Ret_Accuracy, Recog_Accuracy).",
    )
    parser.add_argument(
        "--output",
        default="id_analysis_stats/enc_ret_delta_behavior_correlations.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    behavior = load_behavior(Path(args.behavior))

    vals: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with Path(args.id_results).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cond = row["Condition"]
            if cond not in ("Enc", "ret"):
                continue
            roi = row["ROI"]
            subj = row["Subject"]
            vals[roi][subj][cond] = float(row["Intrinsic_Dim"])

    rows = []
    for roi, by_subj in vals.items():
        deltas = {}
        for subj, conds in by_subj.items():
            if "Enc" in conds and "ret" in conds and subj in behavior:
                deltas[subj] = conds["ret"] - conds["Enc"]

        for metric in ("Ret_Accuracy", "Recog_Accuracy"):
            x = []
            y = []
            for subj, delta in deltas.items():
                if subj in behavior:
                    x.append(delta)
                    y.append(behavior[subj][metric])
            if len(x) < 4:
                continue
            r, p = stats.pearsonr(x, y)
            n = len(x)
            z = atanh(r)
            se = 1 / sqrt(n - 3)
            z_lo = z - 1.96 * se
            z_hi = z + 1.96 * se
            ci = (tanh(z_lo), tanh(z_hi))
            rows.append(
                {
                    "ROI": roi,
                    "Metric": metric,
                    "N": n,
                    "R": r,
                    "P": p,
                    "CI_low": ci[0],
                    "CI_high": ci[1],
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ROI", "Metric", "N", "R", "P", "CI_low", "CI_high"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved correlations to {out_path}")


if __name__ == "__main__":
    main()
