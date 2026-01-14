import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROI_CONDITIONS = ["ret", "retRating", "retRatingFeedback", "faceRecog", "recog"]


def parse_trial_type(trial_type: str) -> Tuple[str, str, str]:
    parts = trial_type.split('_')
    if len(parts) == 1:  # baseline or other
        return parts[0], "", ""
    stage_raw = ''.join(ch for ch in parts[0] if ch.isalpha())
    confidence = ""
    accuracy = ""
    if len(parts) >= 3:
        confidence = parts[1]
        accuracy = parts[2]
    elif len(parts) == 2:
        confidence = parts[1]
    return stage_raw, confidence, accuracy


def compute_behavior_metrics(events_path: Path, stages: Tuple[str, ...]) -> Dict[str, float]:
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    stage_set = {stage.lower() for stage in stages}
    with events_path.open() as f:
        reader = csv.DictReader(f, delimiter='\t')
        recall_entries = []
        for row in reader:
            trial_type = row.get("trial_type", "")
            stage, confidence, accuracy = parse_trial_type(trial_type)
            stage_lower = stage.lower()
            if stage_lower not in stage_set:
                continue
            if confidence == "noResp":
                continue
            recall_entries.append((confidence.lower(), accuracy.lower()))
    if not recall_entries:
        raise ValueError(f"No recall trials with responses found in {events_path}")
    total = len(recall_entries)
    accuracy = sum(1 for conf, acc in recall_entries if acc == "correct") / total
    guess_rate = sum(1 for conf, acc in recall_entries if conf == "guess") / total
    sure_rate = sum(1 for conf, acc in recall_entries if conf == "sure") / total
    return {
        "Accuracy": accuracy,
        "Guess_Rate": guess_rate,
        "Sure_Rate": sure_rate,
    }


def load_slopes(slopes_csv: Path, condition: str) -> Dict[str, float]:
    slopes: Dict[str, float] = {}
    with slopes_csv.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Condition"] != condition:
                continue
            slopes[row["Subject"]] = float(row["Slope"])
    return slopes


def pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    r = np.corrcoef(x, y)[0, 1]
    # two-tailed p-value using t distribution
    df = len(x) - 2
    if df <= 0:
        return r, float("nan")
    t = abs(r) * math.sqrt(df / (1 - r ** 2)) if abs(r) != 1 else float('inf')
    # approximate p using survival function of t distribution via scipy formula replacement
    # Implement using np.abs and math.erfc? Instead approximate using scipy? Since SciPy may not be available,
    # we fall back to student-t CDF via numpy if SciPy absent. We'll use a simple approximation by numerical integration.
    # For df <= 30, we can use betainc relation: p = 2 * I_{df/(df+t^2)}(df/2, 0.5)
    # incomplete beta not available; approximate with SciPy? Instead use mpmath? Too heavy.
    # We'll just return NaN p-value to avoid dependency.
    return r, float("nan")


def main():
    parser = argparse.ArgumentParser(description="Correlate ventral slopes with behavioural metrics")
    parser.add_argument("subject_slopes_csv", help="Path to human_subject_slopes.csv")
    parser.add_argument("bids_root", help="Path to BIDS root (for events files)")
    parser.add_argument("--conditions", nargs="*", default=["ret", "retRatingFeedback"], help="Conditions to analyse")
    parser.add_argument("--stages", nargs="*", default=["ret"], help="Trial stages to include (e.g., ret, retRatingFeedback)")
    parser.add_argument("--session", default="1", help="Session ID (default: 1)")
    parser.add_argument("--task", default="forgetexp", help="Task name in events filenames")
    args = parser.parse_args()

    slopes_data: Dict[str, Dict[str, float]] = {
        cond: load_slopes(Path(args.subject_slopes_csv), cond) for cond in args.conditions
    }

    behavior_results: Dict[str, Dict[str, float]] = {}
    stages = tuple(s.lower() for s in args.stages)
    for subject in set(s for slopes in slopes_data.values() for s in slopes):
        events_path = Path(args.bids_root) / f"sub-{subject}" / f"ses-{args.session}" / "func" / f"sub-{subject}_ses-{args.session}_task-{args.task}_events.tsv"
        try:
            behavior_results[subject] = compute_behavior_metrics(events_path, stages)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping subject {subject}: {exc}")

    for cond, slopes in slopes_data.items():
        xs = []
        ys_acc = []
        ys_guess = []
        ys_sure = []
        for subject, slope in slopes.items():
            metrics = behavior_results.get(subject)
            if not metrics:
                continue
            xs.append(slope)
            ys_acc.append(metrics["Accuracy"])
            ys_guess.append(metrics["Guess_Rate"])
            ys_sure.append(metrics["Sure_Rate"])
        if len(xs) < 2:
            print(f"Condition {cond}: insufficient overlapping subjects for correlation")
            continue
        x_arr = np.array(xs, dtype=float)
        acc_arr = np.array(ys_acc, dtype=float)
        guess_arr = np.array(ys_guess, dtype=float)
        sure_arr = np.array(ys_sure, dtype=float)
        r_acc, _ = pearson(x_arr, acc_arr)
        r_guess, _ = pearson(x_arr, guess_arr)
        r_sure, _ = pearson(x_arr, sure_arr)
        print(f"Condition {cond}: n={len(x_arr)} | r(slope, Accuracy)={r_acc:.3f} | r(slope, Guess_Rate)={r_guess:.3f} | r(slope, Sure_Rate)={r_sure:.3f}")


if __name__ == "__main__":
    main()
