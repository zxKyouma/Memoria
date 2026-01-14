import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROI_ORDER = ["V1", "hV4", "LOIT", "phc", "hippocampus"]


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["ROI"].isin(ROI_ORDER)].copy()
    df["ROI_index"] = df["ROI"].map({roi: idx for idx, roi in enumerate(ROI_ORDER)})
    return df


def compute_subject_slopes(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for (subject, condition), group in df.groupby(["Subject", "Condition"]):
        x = group["ROI_index"].values.astype(float)
        y = group["Intrinsic_Dim"].values.astype(float)
        if len(np.unique(x)) < 2:
            continue
        slope, intercept = np.polyfit(x, y, deg=1)
        records.append(
            {
                "Subject": subject,
                "Condition": condition,
                "Slope": slope,
                "Intercept": intercept,
                "MeanDim": y.mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(sample.mean())
    lower = np.percentile(boot, 100 * alpha / 2)
    upper = np.percentile(boot, 100 * (1 - alpha / 2))
    return np.array([lower, upper])


def summarise(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in slopes.groupby("Condition"):
        values = group["Slope"].values
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values))
        ci_low, ci_high = bootstrap_ci(values)
        rows.append(
            {
                "Condition": condition,
                "MeanSlope": mean,
                "SEM": sem,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "N": len(values),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute ventral-stream dimensionality slopes")
    parser.add_argument("csv", help="Path to group_id_full_results_by_condition.csv")
    parser.add_argument("--output-dir", default="id_analysis_stats")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)
    slopes = compute_subject_slopes(df)
    slopes_path = output_dir / "subject_condition_slopes.csv"
    slopes.to_csv(slopes_path, index=False)

    summary = summarise(slopes)
    summary_path = output_dir / "slope_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved subject-level slopes to", slopes_path)
    print("Saved summary statistics to", summary_path)


if __name__ == "__main__":
    main()
