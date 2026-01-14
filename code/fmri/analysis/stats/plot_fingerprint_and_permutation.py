#!/usr/bin/env python3
"""
Generate ladder plots, fingerprint scatter, and a sign-flip permutation test.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_slopes(path: Path, condition: str) -> pd.Series:
    df = pd.read_csv(path)
    subset = df[df["Condition"] == condition][["Subject", "Slope"]].dropna()
    return subset.set_index("Subject")["Slope"]


def ladder_plot(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    title: str,
    out_path: Path,
) -> None:
    df = df.copy()
    df["Delta"] = df[col_b] - df[col_a]
    df = df.sort_values("Delta")

    fig, ax = plt.subplots(figsize=(5.5, 6.0))
    for _, row in df.iterrows():
        ax.plot([0, 1], [row[col_a], row[col_b]], color="0.5", linewidth=1.0)
        ax.scatter([0, 1], [row[col_a], row[col_b]], s=24, color="black")

    ax.axhline(0, color="0.8", linewidth=1.0, linestyle="--")
    ax.set_xticks([0, 1], [label_a, label_b])
    ax.set_title(title)
    ax.set_ylabel("Ventral-stream slope")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fingerprint_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
) -> Tuple[float, int]:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(x, y, s=36, color="black")
    ax.axhline(0, color="0.8", linewidth=1.0, linestyle="--")
    ax.axvline(0, color="0.8", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Δ enc→ret slope")
    ax.set_ylabel("Δ ret→retRatingFeedback slope")
    ax.set_title(f"{title}\nr = {r:.2f}, n = {len(x)}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return r, len(x)


def sign_flip_permutation(deltas: np.ndarray, permutations: int, seed: int) -> Tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed = float(deltas.mean())
    signs = rng.choice([-1, 1], size=(permutations, len(deltas)))
    null = (signs * deltas).mean(axis=1)
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (permutations + 1)
    return observed, p_value, null


def main() -> None:
    parser = argparse.ArgumentParser(description="Ladder plots, fingerprint scatter, and sign-flip permutation.")
    parser.add_argument(
        "--enc-ret-slopes",
        default="id_analysis_stats/enc_vs_ret_cleaned/subject_condition_slopes.csv",
        help="Subject slopes CSV for Enc vs ret cleaned analysis.",
    )
    parser.add_argument(
        "--metacog-slopes",
        default="id_analysis_stats/subject_condition_slopes.csv",
        help="Subject slopes CSV for ret/retRatingFeedback (session 2).",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/fingerprint_plots",
        help="Output directory for plots and stats.",
    )
    parser.add_argument("--permutations", type=int, default=5000, help="Number of sign-flip permutations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutations.")
    args = parser.parse_args()

    enc_ret_path = Path(args.enc_ret_slopes)
    metacog_path = Path(args.metacog_slopes)
    output_dir = Path(args.output_dir)

    enc = load_slopes(enc_ret_path, "Enc")
    ret_enc = load_slopes(enc_ret_path, "ret")
    enc_ret_df = pd.concat({"Enc": enc, "ret": ret_enc}, axis=1).dropna()
    enc_ret_df["Delta"] = enc_ret_df["ret"] - enc_ret_df["Enc"]

    ret_meta = load_slopes(metacog_path, "ret")
    meta = load_slopes(metacog_path, "retRatingFeedback")
    meta_df = pd.concat({"ret": ret_meta, "retRatingFeedback": meta}, axis=1).dropna()
    meta_df["Delta"] = meta_df["retRatingFeedback"] - meta_df["ret"]

    ladder_plot(
        enc_ret_df,
        "Enc",
        "ret",
        "Enc",
        "Ret",
        "Enc → Ret (paired slopes)",
        output_dir / "ladder_enc_ret.png",
    )
    ladder_plot(
        meta_df,
        "ret",
        "retRatingFeedback",
        "Ret",
        "RetRatingFeedback",
        "Ret → RetRatingFeedback (paired slopes)",
        output_dir / "ladder_ret_metacog.png",
    )

    fingerprint_df = enc_ret_df[["Delta"]].rename(columns={"Delta": "Delta_EncRet"}).join(
        meta_df[["Delta"]].rename(columns={"Delta": "Delta_RetMetacog"}),
        how="inner",
    )
    r_val, n_val = fingerprint_scatter(
        fingerprint_df,
        "Delta_EncRet",
        "Delta_RetMetacog",
        "Dimensionality Fingerprint",
        output_dir / "fingerprint_scatter.png",
    )

    observed, p_value, null = sign_flip_permutation(
        enc_ret_df["Delta"].to_numpy(dtype=float),
        permutations=args.permutations,
        seed=args.seed,
    )
    np.save(output_dir / "enc_ret_null_distribution.npy", null)
    summary_df = pd.DataFrame(
        [
            {
                "Contrast": "Enc_vs_Ret",
                "N": len(enc_ret_df),
                "Observed_Mean_Delta": observed,
                "Permutation_P": p_value,
                "Permutations": args.permutations,
                "Seed": args.seed,
                "Fingerprint_r": r_val,
                "Fingerprint_n": n_val,
            }
        ]
    )
    summary_path = output_dir / "fingerprint_permutation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved plots to {output_dir}")
    print(f"Permutation summary: {summary_path}")
    print(f"Observed Δ (Enc→Ret) mean = {observed:.4f}, p = {p_value:.4f}")
    print(f"Fingerprint r = {r_val:.3f}, n = {n_val}")


if __name__ == "__main__":
    main()
