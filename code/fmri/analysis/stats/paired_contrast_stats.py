#!/usr/bin/env python3
"""
Compute paired t-tests and effect sizes for condition contrasts by ROI.

Defaults match the hit/persistent/lost contrasts used in the manuscript.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scipy import stats


def load_values(path: Path, conditions: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    vals: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cond = row["Condition"]
            if cond not in conditions:
                continue
            roi = row["ROI"]
            subj = row["Subject"]
            vals[roi][subj][cond] = float(row["Intrinsic_Dim"])
    return vals


def paired_stats(a: List[float], b: List[float]) -> Tuple[float, float, float, float, float]:
    n = len(a)
    diffs = [b[i] - a[i] for i in range(n)]
    mean = sum(diffs) / n
    sd = (sum((x - mean) ** 2 for x in diffs) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    d = mean / sd if sd else float("nan")
    t, p = stats.ttest_rel(a, b)
    return mean, sd, d, t, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired contrasts by ROI.")
    parser.add_argument(
        "--input",
        default="id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned_controls_nosm/group_id_full_results_by_condition.csv",
        help="Path to group_id_full_results_by_condition.csv",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=[
            "ret_hit:ret_miss_recog_incorrect:hit_minus_miss_incorrect",
            "ret_miss_recog_correct:ret_miss_recog_incorrect:persistent_minus_miss_incorrect",
            "ret_hit:ret_miss_recog_correct:hit_minus_persistent",
        ],
        help="Triples as condA:condB:label (diff = condA - condB).",
    )
    parser.add_argument(
        "--output",
        default="id_analysis_stats/paired_contrast_stats.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    contrast_specs = []
    conditions = set()
    for item in args.contrasts:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid contrast spec: {item}")
        cond_a, cond_b, label = parts
        contrast_specs.append((cond_a, cond_b, label))
        conditions.update([cond_a, cond_b])

    vals = load_values(Path(args.input), sorted(conditions))

    out_rows = []
    for roi, by_subj in vals.items():
        for cond_a, cond_b, label in contrast_specs:
            a_vals = []
            b_vals = []
            for subj, conds in by_subj.items():
                if cond_a in conds and cond_b in conds:
                    a_vals.append(conds[cond_a])
                    b_vals.append(conds[cond_b])
            if len(a_vals) < 2:
                continue
            mean, sd, d, t, p = paired_stats(b_vals, a_vals)
            out_rows.append(
                {
                    "ROI": roi,
                    "Contrast": label,
                    "CondA": cond_a,
                    "CondB": cond_b,
                    "N": len(a_vals),
                    "MeanDiff": mean,
                    "SD": sd,
                    "CohensD": d,
                    "T": t,
                    "P": p,
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ROI", "Contrast", "CondA", "CondB", "N", "MeanDiff", "SD", "CohensD", "T", "P"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved paired contrast stats to {out_path}")


if __name__ == "__main__":
    main()
