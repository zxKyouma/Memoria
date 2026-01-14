#!/usr/bin/env python3
"""
Correlate confidence-related ID drop with memory performance.

Defines ID drop as hippocampus ID (retRating_sure - retRating_guess).
Computes ret and recog accuracy from session-2 events.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TRIAL_TYPE_RE = r"^([A-Za-z]+)(\d+)(?:_([^_]+))?(?:_([^_]+))?$"


def parse_trial_type(trial_type: str) -> Tuple[str, str, str, str]:
    import re

    match = re.match(TRIAL_TYPE_RE, trial_type)
    if not match:
        return "", "", "", ""
    stage_raw, item_id, confidence, accuracy = match.groups()
    return stage_raw.lower(), item_id, (confidence or "").lower(), (accuracy or "").lower()


def load_events(events_path: Path) -> pd.DataFrame:
    if not events_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(events_path, sep="\t")
    parsed = df["trial_type"].fillna("").astype(str).apply(parse_trial_type)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["stage", "item_id", "confidence", "accuracy"])
    return parsed_df


def compute_accuracy(df: pd.DataFrame, stage: str) -> float:
    subset = df[df["stage"] == stage].copy()
    subset = subset[(subset["confidence"] != "noresp") & (subset["accuracy"] != "noresp")]
    if subset.empty:
        return float("nan")
    correct = (subset["accuracy"] == "correct").sum()
    total = len(subset)
    return float(correct / total) if total > 0 else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom != 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence-ID fingerprint analysis.")
    parser.add_argument(
        "--id-results",
        default="id_analysis_results_by_condition_ses-2_retRating_confidence/group_id_full_results_by_condition.csv",
        help="Path to retRating confidence group ID results.",
    )
    parser.add_argument(
        "--bids-root",
        default=".",
        help="Path to BIDS root (for events files).",
    )
    parser.add_argument(
        "--session",
        default="2",
        help="Session label (default: 2).",
    )
    parser.add_argument(
        "--roi",
        default="hippocampus",
        help="ROI for ID drop (default: hippocampus).",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/confidence_fingerprint",
        help="Output directory for summary CSVs.",
    )
    args = parser.parse_args()

    id_df = pd.read_csv(args.id_results)
    id_df = id_df[id_df["ROI"] == args.roi].copy()
    if id_df.empty:
        raise ValueError(f"No rows found for ROI {args.roi} in {args.id_results}")

    pivot = id_df.pivot_table(
        index="Subject",
        columns="Condition",
        values="Intrinsic_Dim",
        aggfunc="mean",
    )
    if "retRating_sure" not in pivot.columns or "retRating_guess" not in pivot.columns:
        raise ValueError("Missing retRating_sure or retRating_guess in ID results.")

    pivot = pivot.dropna(subset=["retRating_sure", "retRating_guess"])
    pivot["ID_Drop_SureMinusGuess"] = pivot["retRating_sure"] - pivot["retRating_guess"]

    behavior_rows = []
    for subject in pivot.index:
        events_path = Path(args.bids_root) / f"sub-{subject}" / f"ses-{args.session}" / "func" / (
            f"sub-{subject}_ses-{args.session}_task-forgetexp_events.tsv"
        )
        events_df = load_events(events_path)
        if events_df.empty:
            continue
        ret_acc = compute_accuracy(events_df, "ret")
        recog_acc = compute_accuracy(events_df, "recog")
        behavior_rows.append(
            {
                "Subject": subject,
                "ID_Drop_SureMinusGuess": float(pivot.loc[subject, "ID_Drop_SureMinusGuess"]),
                "Ret_Accuracy": ret_acc,
                "Recog_Accuracy": recog_acc,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not behavior_rows:
        raise ValueError("No subjects with both ID drop and behavior metrics.")

    summary_path = output_dir / "confidence_fingerprint_subjects.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Subject", "ID_Drop_SureMinusGuess", "Ret_Accuracy", "Recog_Accuracy"],
        )
        writer.writeheader()
        writer.writerows(behavior_rows)

    data = pd.DataFrame(behavior_rows)
    results = []
    for metric in ["Ret_Accuracy", "Recog_Accuracy"]:
        subset = data.dropna(subset=["ID_Drop_SureMinusGuess", metric])
        r = pearson(subset["ID_Drop_SureMinusGuess"].to_numpy(), subset[metric].to_numpy())
        results.append({"Metric": metric, "N": int(len(subset)), "r": r})

    results_path = output_dir / "confidence_fingerprint_summary.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "N", "r"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved subject data to {summary_path}")
    print(f"Saved correlation summary to {results_path}")


if __name__ == "__main__":
    main()
