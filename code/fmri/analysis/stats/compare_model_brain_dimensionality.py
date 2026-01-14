import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROI_ORDER: Sequence[str] = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
LAYER_ORDER: Sequence[str] = [
    "enc_layer_1",
    "enc_layer_2",
    "enc_layer_3",
    "enc_layer_4",
    "latent_z",
]


def compute_slope(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(y, y[0]):
        return 0.0
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def load_human_data(csv_path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("ROI") not in ROI_ORDER:
                continue
            records.append(row)
    return records


def summarise_human_slopes(
    records: List[Dict[str, str]],
    log_transform: bool,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    roi_index = {roi: idx for idx, roi in enumerate(ROI_ORDER)}
    grouped: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)

    for row in records:
        condition = row["Condition"]
        subject = row["Subject"]
        x = float(roi_index[row["ROI"]])
        y = float(row["Intrinsic_Dim"])
        if log_transform:
            if y <= 0:
                continue
            y = float(np.log10(y))
        grouped[(condition, subject)].append((x, y))

    subject_slopes: List[Dict[str, float]] = []
    for (condition, subject), pairs in grouped.items():
        pairs.sort(key=lambda p: p[0])
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        slope = compute_slope(x, y)
        subject_slopes.append({"Condition": condition, "Subject": subject, "Slope": slope})

    condition_map: Dict[str, List[float]] = defaultdict(list)
    for item in subject_slopes:
        condition_map[item["Condition"]].append(item["Slope"])

    rng = np.random.default_rng(42)
    summary_rows: List[Dict[str, float]] = []
    for condition, slopes in condition_map.items():
        values = np.array(slopes, dtype=float)
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        boot_means = []
        for _ in range(5000):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means.append(sample.mean())
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        summary_rows.append({
            "Condition": condition,
            "MeanSlope": mean,
            "SEM": sem,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "N": float(len(values)),
        })

    summary_rows.sort(key=lambda r: r["Condition"])
    subject_slopes.sort(key=lambda r: (r["Condition"], r["Subject"]))
    return summary_rows, subject_slopes


def compute_model_slope(csv_path: Path, log_transform: bool) -> float:
    layer_index = {layer: idx for idx, layer in enumerate(LAYER_ORDER)}
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            layer = row.get("layer")
            if layer not in layer_index:
                continue
            xs.append(layer_index[layer])
            ys.append(float(row["intrinsic_dim"]))
    order = np.argsort(xs)
    x_arr = np.array(xs, dtype=float)[order]
    y_arr = np.array(ys, dtype=float)[order]
    if log_transform:
        y_arr = np.log10(y_arr)
    return compute_slope(x_arr, y_arr)


def write_csv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, float]]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Compare human ventral slopes with BVAE dimensionality slopes")
    parser.add_argument("human_full_results", help="Path to group_id_full_results_by_condition.csv")
    parser.add_argument("model_id_csv", help="Path to BVAE id_results csv")
    parser.add_argument("--output-dir", default="comparison_outputs", help="Directory to save outputs")
    parser.add_argument(
        "--human-log10",
        action="store_true",
        help="Apply log10 transform to human Intrinsic_Dim before slope computation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_human_data(Path(args.human_full_results))
    summary_rows, subject_slopes = summarise_human_slopes(records, log_transform=args.human_log10)

    human_summary_path = output_dir / "human_slopes_summary.csv"
    subject_slopes_path = output_dir / "human_subject_slopes.csv"
    write_csv(human_summary_path, ["Condition", "MeanSlope", "SEM", "CI_low", "CI_high", "N"], summary_rows)
    write_csv(subject_slopes_path, ["Condition", "Subject", "Slope"], subject_slopes)

    model_slope_log = compute_model_slope(Path(args.model_id_csv), log_transform=True)
    model_slope_linear = compute_model_slope(Path(args.model_id_csv), log_transform=False)

    model_summary_rows = [
        {"System": "BVAE", "Scale": "log10", "Slope": model_slope_log},
        {"System": "BVAE", "Scale": "linear", "Slope": model_slope_linear},
    ]
    model_summary_path = output_dir / "model_slope_summary.csv"
    write_csv(model_summary_path, ["System", "Scale", "Slope"], model_summary_rows)

    combined_rows: List[Dict[str, float]] = []
    for row in summary_rows:
        combined = dict(row)
        combined["ModelSlope_log10"] = model_slope_log
        combined["ModelSlope_linear"] = model_slope_linear
        combined_rows.append(combined)
    combined_path = output_dir / "human_model_comparison.csv"
    write_csv(combined_path, ["Condition", "MeanSlope", "SEM", "CI_low", "CI_high", "N", "ModelSlope_log10", "ModelSlope_linear"], combined_rows)

    print(f"Saved human summary to {human_summary_path}")
    print(f"Saved subject slopes to {subject_slopes_path}")
    print(f"Saved model slope summary to {model_summary_path}")
    print(f"Saved combined table to {combined_path}")


if __name__ == "__main__":
    main()
