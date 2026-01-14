#!/usr/bin/env python3
"""
Plot human vs BVAE hierarchical ID in log10 space (divergence plot).
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROI_ORDER: List[str] = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
LAYER_ORDER: List[str] = ["enc_layer_1", "enc_layer_2", "enc_layer_3", "enc_layer_4", "latent_z"]


def load_human_curve(human_csv: Path, condition: str) -> np.ndarray:
    df = pd.read_csv(human_csv)
    df = df[df["Condition"] == condition]
    df = df[df["ROI"].isin(ROI_ORDER)]
    means = df.groupby("ROI")["Intrinsic_Dim"].mean().reindex(ROI_ORDER)
    return np.log10(means.values)


def load_persistent_point(persistent_csv: Path) -> float:
    df = pd.read_csv(persistent_csv)
    df = df[(df["Condition"] == "ret_miss_recog_correct") & (df["ROI"] == "hippocampus")]
    return float(np.log10(df["Intrinsic_Dim"].mean()))


def load_model_curve(model_csv: Path, stage: str) -> np.ndarray:
    df = pd.read_csv(model_csv)
    df = df[df["stage"] == stage]
    df = df[df["layer"].isin(LAYER_ORDER)]
    means = df.set_index("layer")["intrinsic_dim"].reindex(LAYER_ORDER)
    return np.log10(means.values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot human vs model ID divergence.")
    parser.add_argument("--human-csv", required=True, help="Human group_id_full_results_by_condition.csv")
    parser.add_argument("--human-condition", default="retRatingFeedback")
    parser.add_argument("--persistent-csv", required=True, help="Persistent-trace group_id_full_results_by_condition.csv")
    parser.add_argument("--model-csv", required=True, help="Metacog BVAE id_results_metacog.csv")
    parser.add_argument("--model-stage", default="feedback")
    parser.add_argument("--output-dir", default="id_analysis_stats/model_human_divergence")
    args = parser.parse_args()

    human_curve = load_human_curve(Path(args.human_csv), args.human_condition)
    model_curve = load_model_curve(Path(args.model_csv), args.model_stage)
    persistent_point = load_persistent_point(Path(args.persistent_csv))

    x = np.arange(len(ROI_ORDER))
    labels = ["V1\\nL1", "hV4\\nL2", "LOIT\\nL3", "PHC\\nL4", "HPC\\nZ"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(x, human_curve, marker="o", label="Human retRatingFeedback", color="#1f77b4")
    plt.plot(x, model_curve, marker="o", label="Metacog BVAE (feedback)", color="#d62728")
    plt.scatter([x[-1]], [persistent_point], marker="*", s=120, color="#2ca02c", label="Persistent Trace (HPC)")

    plt.xticks(x, labels)
    plt.ylabel("log10 Intrinsic Dimensionality")
    plt.xlabel("Hierarchy depth (Human ROIs / Model layers)")
    plt.title("Human vs BVAE Hierarchical ID (Log10)")
    plt.legend(frameon=False)
    plt.tight_layout()

    fig_path = output_dir / "divergence_plot.png"
    plt.savefig(fig_path, dpi=200)

    data_path = output_dir / "divergence_plot_values.csv"
    pd.DataFrame(
        {
            "depth_index": x,
            "roi": ROI_ORDER,
            "model_layer": LAYER_ORDER,
            "human_log10_id": human_curve,
            "model_log10_id": model_curve,
        }
    ).to_csv(data_path, index=False)

    meta_path = output_dir / "divergence_plot_meta.csv"
    pd.DataFrame(
        [
            {
                "human_condition": args.human_condition,
                "model_stage": args.model_stage,
                "persistent_log10_id": persistent_point,
            }
        ]
    ).to_csv(meta_path, index=False)

    print(f"Saved plot to {fig_path}")
    print(f"Saved values to {data_path}")
    print(f"Saved meta to {meta_path}")


if __name__ == "__main__":
    main()
