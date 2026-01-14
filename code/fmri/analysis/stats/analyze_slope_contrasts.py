import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_CONTRASTS: Sequence[Tuple[str, str]] = (
    ("ret", "retRatingFeedback"),
    ("ret", "retRating"),
    ("ret", "faceRecog"),
)


def load_subject_slopes(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    condition_map: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = row["Condition"]
            subject = row["Subject"]
            slope = float(row["Slope"])
            condition_map[condition].append((subject, slope))
    return condition_map


def align_subjects(group_a: List[Tuple[str, float]], group_b: List[Tuple[str, float]]):
    map_a = {sub: slope for sub, slope in group_a}
    map_b = {sub: slope for sub, slope in group_b}
    common = sorted(set(map_a).intersection(map_b))
    slopes_a = np.array([map_a[sub] for sub in common], dtype=float)
    slopes_b = np.array([map_b[sub] for sub in common], dtype=float)
    return common, slopes_a, slopes_b


def bootstrap_diff(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000):
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    return lower, upper


def analyse_pairs(condition_map: Dict[str, List[Tuple[str, float]]], pairs: Sequence[Tuple[str, str]]):
    rng = np.random.default_rng(42)
    for cond_a, cond_b in pairs:
        if cond_a not in condition_map or cond_b not in condition_map:
            print(f"Skipping {cond_a} vs {cond_b}: missing data")
            continue
        common_subjects, slopes_a, slopes_b = align_subjects(condition_map[cond_a], condition_map[cond_b])
        if len(common_subjects) < 2:
            print(f"Skipping {cond_a} vs {cond_b}: insufficient overlap (n={len(common_subjects)})")
            continue
        diff = slopes_b - slopes_a
        mean_diff = float(diff.mean())
        sem = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        ci_low, ci_high = bootstrap_diff(diff, rng)
        print(f"{cond_a} -> {cond_b}: n={len(diff)} | mean Δslope={mean_diff:.4f} | SEM={sem:.4f} | 95% CI [{ci_low:.4f}, {ci_high:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Analyse condition contrasts for ventral-stream slopes")
    parser.add_argument("subject_slopes_csv", help="Path to human_subject_slopes.csv")
    parser.add_argument("--pairs", nargs="*", help="Pairs of conditions (format: condA:condB)")
    args = parser.parse_args()

    condition_map = load_subject_slopes(Path(args.subject_slopes_csv))
    if args.pairs:
        pairs = []
        for item in args.pairs:
            if ':' not in item:
                parser.error(f"Invalid pair specification: {item}")
            cond_a, cond_b = item.split(':', 1)
            pairs.append((cond_a, cond_b))
    else:
        pairs = list(DEFAULT_CONTRASTS)

    analyse_pairs(condition_map, pairs)


if __name__ == "__main__":
    main()
