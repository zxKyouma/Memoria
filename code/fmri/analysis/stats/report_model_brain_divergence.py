import argparse
import csv
from pathlib import Path


def load_human_summary(path: Path):
    rows = []
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "Condition": row["Condition"],
                "MeanSlope": float(row["MeanSlope"]),
                "SEM": float(row["SEM"]),
                "CI_low": float(row["CI_low"]),
                "CI_high": float(row["CI_high"]),
                "N": float(row["N"]),
                "ModelSlope_log10": float(row["ModelSlope_log10"]),
                "ModelSlope_linear": float(row["ModelSlope_linear"]),
            })
    return rows


def compute_divergence(rows):
    summary = []
    for row in rows:
        diff = row["ModelSlope_log10"] - row["MeanSlope"]
        diverges = row["ModelSlope_log10"] < row["CI_low"] or row["ModelSlope_log10"] > row["CI_high"]
        summary.append({
            "Condition": row["Condition"],
            "HumanMeanSlope": row["MeanSlope"],
            "HumanCI_low": row["CI_low"],
            "HumanCI_high": row["CI_high"],
            "ModelSlope_log10": row["ModelSlope_log10"],
            "Difference": diff,
            "ModelOutsideHumanCI": diverges,
        })
    return summary


def write_csv(path: Path, rows):
    fieldnames = ["Condition", "HumanMeanSlope", "HumanCI_low", "HumanCI_high", "ModelSlope_log10", "Difference", "ModelOutsideHumanCI"]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarise divergence between BVAE and human slopes")
    parser.add_argument("combined_csv", help="Path to human_model_comparison.csv")
    parser.add_argument("--output", default="model_brain_divergence.csv")
    args = parser.parse_args()

    rows = load_human_summary(Path(args.combined_csv))
    summary = compute_divergence(rows)
    write_csv(Path(args.output), summary)

    print(f"Saved divergence summary to {args.output}")
    print("\nModel vs human slopes:")
    for item in summary:
        status = "OUTSIDE CI" if item["ModelOutsideHumanCI"] else "within CI"
        print(f"  {item['Condition']}: human={item['HumanMeanSlope']:.3f} (CI [{item['HumanCI_low']:.3f},{item['HumanCI_high']:.3f}]) | model={item['ModelSlope_log10']:.3f} -> diff={item['Difference']:.3f} => {status}")


if __name__ == '__main__':
    main()
