#!/usr/bin/env python3
"""
Compute ROI-level tSNR for encoding vs retrieval and correlate with ID.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker


# --- USER CONFIGURATION ---
BIDS_DIR = "/home/zlow/by84/Brandon/memoria"
ROI_NAMES_IN_ORDER = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
ENC_SESSION = "1"
RET_SESSION = "2"
TASK_NAME = "forgetexp"
# --- END OF CONFIGURATION ---


def find_bold(subject_id: str, session_id: str) -> Path:
    func_dir = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref" / f"sub-{subject_id}" / f"ses-{session_id}" / "func"
    pattern = f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_acq-*_space-T1w_desc-preproc_bold.nii.gz"
    matches = sorted(func_dir.glob(pattern))
    if not matches:
        return Path()
    return matches[0]


def roi_mask_path(subject_id: str, session_id: str, roi_name: str) -> Path:
    return Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{session_id}_space-T1w_desc-{roi_name}_resampled-betaseries_mask.nii.gz"
    )


def compute_tsnr(img, mask_path: Path) -> float:
    masker = NiftiMasker(mask_img=str(mask_path), standardize=False)
    voxel_data = masker.fit_transform(img)
    mean_ts = voxel_data.mean(axis=0)
    std_ts = voxel_data.std(axis=0, ddof=1)
    valid = std_ts > 0
    if not np.any(valid):
        return float("nan")
    tsnr = np.mean(mean_ts[valid] / std_ts[valid])
    return float(tsnr)


def compute_tsnr_table(subjects: List[str]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for subject_id in subjects:
        enc_path = find_bold(subject_id, ENC_SESSION)
        ret_path = find_bold(subject_id, RET_SESSION)
        if not enc_path.exists() or not ret_path.exists():
            continue
        enc_img = load_img(str(enc_path))
        ret_img = load_img(str(ret_path))

        for roi in ROI_NAMES_IN_ORDER:
            enc_mask = roi_mask_path(subject_id, ENC_SESSION, roi)
            ret_mask = roi_mask_path(subject_id, RET_SESSION, roi)
            if not enc_mask.exists() or not ret_mask.exists():
                continue
            records.append(
                {
                    "Subject": subject_id,
                    "Condition": "Enc",
                    "ROI": roi,
                    "tSNR": compute_tsnr(enc_img, enc_mask),
                    "Num_TRs": enc_img.shape[-1],
                }
            )
            records.append(
                {
                    "Subject": subject_id,
                    "Condition": "ret",
                    "ROI": roi,
                    "tSNR": compute_tsnr(ret_img, ret_mask),
                    "Num_TRs": ret_img.shape[-1],
                }
            )
    return pd.DataFrame(records)


def correlate_tsnr_id(tsnr_df: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        tsnr_df,
        id_df[["Subject", "Condition", "ROI", "Intrinsic_Dim"]],
        on=["Subject", "Condition", "ROI"],
        how="inner",
    )
    records: List[Dict[str, object]] = []
    for (condition, roi), sub_df in merged.groupby(["Condition", "ROI"]):
        if len(sub_df) < 3:
            continue
        x = sub_df["tSNR"].to_numpy(dtype=float)
        y = sub_df["Intrinsic_Dim"].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        records.append(
            {
                "Condition": condition,
                "ROI": roi,
                "N": int(len(sub_df)),
                "r_tsnr_id": r,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROI tSNR and correlate with ID.")
    parser.add_argument(
        "--id-results",
        default="id_analysis_results_by_condition_enc_vs_ret_cleaned/group_id_full_results_by_condition.csv",
        help="Path to ID results CSV (Enc/ret conditions).",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/tsnr",
        help="Output directory for tSNR results.",
    )
    args = parser.parse_args()

    id_path = Path(args.id_results)
    if not id_path.exists():
        raise FileNotFoundError(f"Missing ID results: {id_path}")

    id_df = pd.read_csv(id_path)
    id_df["Subject"] = id_df["Subject"].astype(str)

    subjects = sorted(id_df["Subject"].unique().tolist())
    tsnr_df = compute_tsnr_table(subjects)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsnr_path = out_dir / "tsnr_by_roi.csv"
    tsnr_df.to_csv(tsnr_path, index=False)
    print(f"Saved tSNR table to: {tsnr_path}")

    corr_df = correlate_tsnr_id(tsnr_df, id_df)
    corr_path = out_dir / "tsnr_id_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"Saved tSNR/ID correlations to: {corr_path}")


if __name__ == "__main__":
    main()
