#!/usr/bin/env python3
"""
Individual-difference coupling across ROIs for enc and ret, plus delta coupling.
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


def compute_corr(x: pd.Series, y: pd.Series):
    r, p = pearsonr(x, y)
    return float(r), float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI coupling across subjects for enc/ret.")
    parser.add_argument(
        "--input",
        default="id_analysis_results_by_condition_enc_vs_ret_cleaned/group_id_full_results_by_condition.csv",
        help="Path to enc vs ret cleaned group results.",
    )
    parser.add_argument(
        "--roi-x",
        default="V1",
        help="ROI for x-axis (e.g., V1).",
    )
    parser.add_argument(
        "--roi-y",
        default="hippocampus",
        help="ROI for y-axis (e.g., hippocampus).",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/roi_coupling_traits",
        help="Output directory.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enc coupling
    enc = df[df["Condition"] == "Enc"].pivot_table(index="Subject", columns="ROI", values="Intrinsic_Dim")
    # Ret coupling
    ret = df[df["Condition"] == "ret"].pivot_table(index="Subject", columns="ROI", values="Intrinsic_Dim")

    rows = []

    if args.roi_x in enc.columns and args.roi_y in enc.columns:
        common = enc[[args.roi_x, args.roi_y]].dropna()
        r, p = compute_corr(common[args.roi_x], common[args.roi_y])
        rows.append({"analysis": "enc_v1_vs_hpc", "N": len(common), "PearsonR": r, "P_value": p})
        common.reset_index().to_csv(output_dir / "enc_v1_vs_hpc_subjects.csv", index=False)

    if args.roi_x in ret.columns and args.roi_y in ret.columns:
        common = ret[[args.roi_x, args.roi_y]].dropna()
        r, p = compute_corr(common[args.roi_x], common[args.roi_y])
        rows.append({"analysis": "ret_v1_vs_hpc", "N": len(common), "PearsonR": r, "P_value": p})
        common.reset_index().to_csv(output_dir / "ret_v1_vs_hpc_subjects.csv", index=False)

    # Delta coupling (ret - enc)
    if args.roi_x in enc.columns and args.roi_x in ret.columns and args.roi_y in enc.columns and args.roi_y in ret.columns:
        delta_x = (ret[args.roi_x] - enc[args.roi_x]).rename("delta_x")
        delta_y = (ret[args.roi_y] - enc[args.roi_y]).rename("delta_y")
        delta = pd.concat([delta_x, delta_y], axis=1).dropna()
        if not delta.empty:
            r, p = compute_corr(delta["delta_x"], delta["delta_y"])
            rows.append({"analysis": "delta_v1_vs_hpc", "N": len(delta), "PearsonR": r, "P_value": p})
            delta.reset_index().to_csv(output_dir / "delta_v1_vs_hpc_subjects.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "roi_coupling_trait_summary.csv", index=False)


if __name__ == "__main__":
    main()
