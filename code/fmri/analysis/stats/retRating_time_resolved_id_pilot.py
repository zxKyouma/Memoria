#!/usr/bin/env python3
"""
Pilot: time-resolved ID around retRating onset (high vs low confidence).

High = sure + unsure
Low = guess
"""
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from sklearn.neighbors import NearestNeighbors


BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
TASK_NAME = "forgetexp"
SESSION_ID = "2"
ROIS_DEFAULT = ["phc", "hippocampus"]
TR_OFFSETS = [0, 1, 2, 3, 4, 5, 6]  # TR0..TR6
EARLY_OFFSETS = [2, 3]
LATE_OFFSETS = [4, 5]
MIN_TRIALS = 15
RANDOM_SEED = 42


def find_subjects() -> List[str]:
    base = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref"
    return sorted(p.name.split("-")[1] for p in base.glob("sub-*") if p.is_dir())


def find_bold(subject_id: str) -> Path:
    func_dir = Path(BIDS_DIR) / "derivatives" / "fMRIprepNoSbref" / f"sub-{subject_id}" / f"ses-{SESSION_ID}" / "func"
    pattern = f"sub-{subject_id}_ses-{SESSION_ID}_task-{TASK_NAME}_acq-*_space-T1w_desc-preproc_bold.nii.gz"
    matches = sorted(func_dir.glob(pattern))
    return matches[0] if matches else Path()


def find_bold_json(subject_id: str) -> Path:
    func_dir = Path(BIDS_DIR) / f"sub-{subject_id}" / f"ses-{SESSION_ID}" / "func"
    pattern = f"sub-{subject_id}_ses-{SESSION_ID}_task-{TASK_NAME}_*_bold.json"
    matches = sorted(func_dir.glob(pattern))
    return matches[0] if matches else Path()


def find_events(subject_id: str) -> Path:
    return Path(BIDS_DIR) / f"sub-{subject_id}" / f"ses-{SESSION_ID}" / "func" / (
        f"sub-{subject_id}_ses-{SESSION_ID}_task-{TASK_NAME}_events.tsv"
    )


def load_roi_mask(subject_id: str, roi: str, ref_img: nib.Nifti1Image) -> np.ndarray:
    if ref_img.ndim == 4 and ref_img.shape[-1] >= 1:
        ref_data = ref_img.get_fdata()[..., 0]
        ref_img = nib.Nifti1Image(ref_data, ref_img.affine, ref_img.header)
    roi_path = Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{SESSION_ID}_space-T1w_desc-{roi}_resampled-betaseries_mask.nii.gz"
    )
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI mask: {roi_path}")
    roi_img = nib.load(str(roi_path))
    if roi_img.shape != ref_img.shape or not np.allclose(roi_img.affine, ref_img.affine):
        roi_img = resample_from_to(roi_img, ref_img, order=0)
    return roi_img.get_fdata() > 0


def parse_events(events_path: Path) -> pd.DataFrame:
    df = pd.read_csv(events_path, sep="\t")
    parsed = df["trial_type"].astype(str).str.extract(
        r"^(?P<stage>[A-Za-z]+)(?P<item>\d+)(?:_(?P<confidence>[^_]+))?(?:_(?P<accuracy>[^_]+))?$"
    )
    parsed["stage"] = parsed["stage"].str.lower()
    parsed["confidence"] = parsed["confidence"].fillna("").str.lower()
    parsed["accuracy"] = parsed["accuracy"].fillna("").str.lower()
    parsed = parsed.rename(columns={"item": "item_id"})
    df = df.join(parsed)
    return df


def get_tr(onset: float, tr: float) -> int:
    return int(round(onset / tr))


def build_trial_indices(
    onsets: np.ndarray,
    tr: float,
    offsets: List[int],
    n_trs: int,
) -> List[List[int]]:
    trial_indices = []
    for onset in onsets:
        base = get_tr(onset, tr)
        idxs = [base + o for o in offsets]
        if min(idxs) < 0 or max(idxs) >= n_trs:
            continue
        trial_indices.append(idxs)
    return trial_indices


def normalize_rows(data: np.ndarray) -> np.ndarray:
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (data - means) / stds


def twonn_id(data: np.ndarray, rng: np.random.Generator, max_samples: int = 5000) -> float:
    if data.shape[0] < 3:
        return float("nan")
    data = np.nan_to_num(data.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if max_samples and data.shape[0] > max_samples:
        idx = rng.choice(data.shape[0], size=max_samples, replace=False)
        data = data[idx]
    data += 1e-6 * rng.standard_normal(data.shape)
    try:
        nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
        nn.fit(data)
        dists, _ = nn.kneighbors(data, n_neighbors=3)
    except ValueError:
        return float("nan")
    d1 = dists[:, 1]
    d2 = dists[:, 2]
    mu = d2 / (d1 + 1e-12)
    mu = mu[np.isfinite(mu) & (mu > 0)]
    if mu.size == 0:
        return float("nan")
    return float(1.0 / np.mean(np.log(mu)))


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def sign_flip_pvalue(values: np.ndarray, n_perm: int = 5000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    more_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(values))
        perm_mean = float((values * signs).mean())
        if abs(perm_mean) >= abs(observed):
            more_extreme += 1
    return (more_extreme + 1) / (n_perm + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot time-resolved ID around retRating onset.")
    parser.add_argument("--rois", nargs="+", default=ROIS_DEFAULT)
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    parser.add_argument("--output-dir", default="id_analysis_stats/retRating_time_resolved_pilot")
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects()
    subject_rows = []
    collapse_rows = []
    excluded = {}

    for subject_id in subjects:
        bold_path = find_bold(subject_id)
        events_path = find_events(subject_id)
        if not bold_path.exists() or not events_path.exists():
            excluded.setdefault(subject_id, []).append("missing bold/events")
            continue

        json_path = find_bold_json(subject_id)
        tr = 2.7
        if json_path.exists():
            with json_path.open() as fh:
                meta = json.load(fh)
            tr = float(meta.get("RepetitionTime", tr))

        bold_img = nib.load(str(bold_path))
        bold_data = bold_img.get_fdata().astype(np.float32)
        n_trs = bold_data.shape[-1]

        events = parse_events(events_path)
        retrating = events[(events["stage"] == "retrating") & (events["confidence"] != "noresp") & (events["accuracy"] != "noresp")]
        if retrating.empty:
            excluded.setdefault(subject_id, []).append("no retrating trials")
            continue

        high = retrating[retrating["confidence"].isin(["sure", "unsure"])]
        low = retrating[retrating["confidence"] == "guess"]

        high_onsets = high["onset"].to_numpy()
        low_onsets = low["onset"].to_numpy()

        high_indices = build_trial_indices(high_onsets, tr, TR_OFFSETS, n_trs)
        low_indices = build_trial_indices(low_onsets, tr, TR_OFFSETS, n_trs)

        if len(high_indices) < args.min_trials or len(low_indices) < args.min_trials:
            excluded.setdefault(subject_id, []).append(
                f"min_trials (high={len(high_indices)}, low={len(low_indices)})"
            )
            continue

        rng = np.random.default_rng(RANDOM_SEED + int(subject_id))

        # Precompute ROI time series with voxel resampling to min vox count across ROIs
        roi_data = {}
        roi_counts = {}
        for roi in args.rois:
            mask = load_roi_mask(subject_id, roi, bold_img)
            vox = bold_data[mask].T  # (TR, vox)
            roi_data[roi] = vox
            roi_counts[roi] = vox.shape[1]
        min_vox = min(roi_counts.values())
        for roi in args.rois:
            if roi_data[roi].shape[1] > min_vox:
                idx = rng.choice(roi_data[roi].shape[1], size=min_vox, replace=False)
                roi_data[roi] = roi_data[roi][:, idx]

        for roi in args.rois:
            for bin_name, trial_idx in [("high", high_indices), ("low", low_indices)]:
                for offset_idx, offset in enumerate(TR_OFFSETS):
                    rows = []
                    for idxs in trial_idx:
                        tr_idx = idxs[offset_idx]
                        rows.append(roi_data[roi][tr_idx])
                    mat = np.stack(rows, axis=0)
                    mat = normalize_rows(mat)
                    id_val = twonn_id(mat, rng)
                    subject_rows.append(
                        {
                            "Subject": subject_id,
                            "ROI": roi,
                            "Bin": bin_name,
                            "TR_offset": offset,
                            "Time_s": offset * tr,
                            "ID": id_val,
                            "Num_Trials": len(trial_idx),
                            "Num_Voxels": mat.shape[1],
                        }
                    )

                # Collapse metric
                early_ids = [r["ID"] for r in subject_rows if r["Subject"] == subject_id and r["ROI"] == roi and r["Bin"] == bin_name and r["TR_offset"] in EARLY_OFFSETS]
                late_ids = [r["ID"] for r in subject_rows if r["Subject"] == subject_id and r["ROI"] == roi and r["Bin"] == bin_name and r["TR_offset"] in LATE_OFFSETS]
                if early_ids and late_ids:
                    collapse_rows.append(
                        {
                            "Subject": subject_id,
                            "ROI": roi,
                            "Bin": bin_name,
                            "ID_early": float(np.mean(early_ids)),
                            "ID_late": float(np.mean(late_ids)),
                            "Collapse": float(np.mean(early_ids) - np.mean(late_ids)),
                        }
                    )

    # Save subject-level time series
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(out_dir / "time_resolved_id_subjects.csv", index=False)

    # Group summary
    summary_rows = []
    for (roi, bin_name, offset), sub_df in subject_df.groupby(["ROI", "Bin", "TR_offset"]):
        vals = sub_df["ID"].to_numpy()
        mean, ci_low, ci_high = mean_ci(vals)
        summary_rows.append(
            {
                "ROI": roi,
                "Bin": bin_name,
                "TR_offset": offset,
                "Time_s": float(offset * tr),
                "N": len(vals),
                "Mean_ID": mean,
                "CI_low": ci_low,
                "CI_high": ci_high,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "time_resolved_id_summary.csv", index=False)

    # Collapse summaries
    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df.to_csv(out_dir / "collapse_subjects.csv", index=False)

    collapse_summary = []
    for roi, roi_df in collapse_df.groupby("ROI"):
        high = roi_df[roi_df["Bin"] == "high"].set_index("Subject")
        low = roi_df[roi_df["Bin"] == "low"].set_index("Subject")
        common = high.index.intersection(low.index)
        if common.empty:
            continue
        delta = high.loc[common, "Collapse"] - low.loc[common, "Collapse"]
        mean, ci_low, ci_high = mean_ci(delta.to_numpy())
        p_perm = sign_flip_pvalue(delta.to_numpy(), n_perm=args.permutations)
        collapse_summary.append(
            {
                "ROI": roi,
                "N": len(delta),
                "DeltaCollapse_mean": mean,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "p_perm": p_perm,
            }
        )
    pd.DataFrame(collapse_summary).to_csv(out_dir / "collapse_summary.csv", index=False)

    if excluded:
        excluded_path = out_dir / "excluded_subjects.csv"
        rows = [{"Subject": sub, "Reasons": "; ".join(reasons)} for sub, reasons in excluded.items()]
        pd.DataFrame(rows).to_csv(excluded_path, index=False)


if __name__ == "__main__":
    main()
