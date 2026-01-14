import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_trial_type(trial_type: str) -> Tuple[str, str, str]:
    parts = trial_type.split('_')
    if not parts:
        return "", "", ""
    stage_raw = ''.join(ch for ch in parts[0] if ch.isalpha())
    confidence = ""
    accuracy = ""
    if len(parts) >= 3:
        confidence = parts[1]
        accuracy = parts[2]
    elif len(parts) == 2:
        confidence = parts[1]
    return stage_raw.lower(), confidence.lower(), accuracy.lower()


def load_behavior_metrics(subject: str, bids_root: Path, session: str, task: str, stages: Tuple[str, ...]) -> Dict[str, float]:
    events_path = bids_root / f"sub-{subject}" / f"ses-{session}" / "func" / f"sub-{subject}_ses-{session}_task-{task}_events.tsv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events for {subject}: {events_path}")
    recall_entries: List[Tuple[str, str]] = []
    stage_set = {stage.lower() for stage in stages}
    with events_path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            stage, confidence, accuracy = parse_trial_type(row.get("trial_type", ""))
            if stage not in stage_set:
                continue
            if confidence == "noresp" or confidence == "":
                continue
            recall_entries.append((confidence, accuracy))
    if not recall_entries:
        raise ValueError(f"No recall trials with responses for {subject}")
    total = len(recall_entries)
    accuracy_prop = sum(1 for conf, acc in recall_entries if acc == "correct") / total
    guess_rate = sum(1 for conf, _ in recall_entries if conf == "guess") / total
    sure_rate = sum(1 for conf, _ in recall_entries if conf == "sure") / total
    return {
        "Accuracy": accuracy_prop,
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


def scatter_plot(condition: str, slopes: Dict[str, float], behavior: Dict[str, Dict[str, float]], metric: str, output_dir: Path) -> None:
    xs = []
    ys = []
    for subj, slope in slopes.items():
        metrics = behavior.get(subj)
        if not metrics:
            continue
        xs.append(slope)
        ys.append(metrics[metric])
    if len(xs) < 2:
        print(f"Skipping {condition} ({metric}): insufficient data (n={len(xs)})")
        return
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    r = np.corrcoef(x_arr, y_arr)[0, 1]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x_arr, y_arr, alpha=0.8, edgecolor='k')
    # best-fit line
    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='tab:red', linewidth=1.5)
    ax.set_xlabel("Ventral-stream slope")
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f"{condition}: r = {r:.2f}, n = {len(x_arr)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"scatter_{condition}_{metric}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot behaviour metrics against ventral slopes")
    parser.add_argument("subject_slopes_csv", help="Path to human_subject_slopes.csv")
    parser.add_argument("bids_root", help="Path to BIDS dataset root")
    parser.add_argument("--conditions", nargs="*", default=["ret", "retRatingFeedback"], help="Conditions to plot")
    parser.add_argument("--metrics", nargs="*", default=["Accuracy"], help="Behaviour metrics to plot")
    parser.add_argument("--stages", nargs="*", default=["ret"], help="Trial stages to include (e.g., ret, retRatingFeedback)")
    parser.add_argument("--session", default="1", help="Session label (default: 1)")
    parser.add_argument("--task", default="forgetexp", help="Task name in events filenames")
    parser.add_argument("--output-dir", default="id_analysis_stats/behavior_plots", help="Directory for plot PNGs")
    args = parser.parse_args()

    slopes_path = Path(args.subject_slopes_csv)
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)

    # gather behavior metrics for all subjects appearing in selected conditions
    slopes_data = {cond: load_slopes(slopes_path, cond) for cond in args.conditions}
    all_subjects = {sub for slopes in slopes_data.values() for sub in slopes}
    behavior: Dict[str, Dict[str, float]] = {}
    stages = tuple(s.lower() for s in args.stages)
    for subj in sorted(all_subjects):
        try:
            behavior[subj] = load_behavior_metrics(subj, bids_root, args.session, args.task, stages)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping subject {subj}: {exc}")

    for condition, slopes in slopes_data.items():
        for metric in args.metrics:
            scatter_plot(condition, slopes, behavior, metric, output_dir)


if __name__ == "__main__":
    main()
