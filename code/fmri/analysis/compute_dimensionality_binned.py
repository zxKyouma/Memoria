#!/usr/bin/env python3
"""
Compute voxel-controlled intrinsic dimensionality (ID) for behaviorally binned trials.

Supports:
  - Enc (session 1) vs ret (session 2)
  - ret hit/miss
  - ret miss split by recog accuracy (3-way refinement)
  - retRating confidence bins (guess/unsure/sure)
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from nilearn.image import load_img, index_img
from nilearn.maskers import NiftiMasker
import skdim
from tqdm import tqdm


# --- USER CONFIGURATION ---
BIDS_DIR = os.environ.get("MEMORIA_ROOT", ".")
TASK_NAME = "forgetexp"
ROI_NAMES_DEFAULT = ["V1", "hV4", "LOIT", "phc", "hippocampus"]
ROI_NAMES_IN_ORDER = ROI_NAMES_DEFAULT
RANDOM_SEED = 42
MIN_TRIALS = 4
VOXEL_RESAMPLES = 1
TRIAL_RESAMPLES = 1
ID_ESTIMATOR = "twonn"
MLE_K = 10
NORMALIZATION = "none"
DEFAULT_BETASERIES_ROOT = "derivatives/nibetaseries/nibetaseries"
# --- END OF CONFIGURATION ---

TRIAL_TYPE_RE = re.compile(r"^([A-Za-z]+)(\d+)(?:_([^_]+))?(?:_([^_]+))?$")
DESC_RE = re.compile(r"_desc-([A-Za-z]+)(\d+)_betaseries\.nii\.gz$")

ANALYSES = {
    "enc_vs_ret": {
        "output_dir": "./id_analysis_results_by_condition_enc_vs_ret",
        "bins": [
            {"name": "Enc", "session": "1", "stage": "Enc", "include_noresp": True},
            {"name": "ret", "session": "2", "stage": "ret"},
        ],
    },
    "enc_vs_ret_cleaned": {
        "output_dir": "./id_analysis_results_by_condition_enc_vs_ret_cleaned",
        "bins": [
            {"name": "Enc", "session": "1", "stage": "Enc", "include_noresp": True},
            {"name": "ret", "session": "2", "stage": "ret"},
        ],
        "match_items_bins": ["Enc", "ret"],
        "count_match_groups": [["Enc", "ret"]],
    },
    "ret_hit_miss": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_ret_hit_miss",
        "bins": [
            {"name": "ret_hit", "session": "2", "stage": "ret", "accuracy": "correct"},
            {"name": "ret_miss", "session": "2", "stage": "ret", "accuracy": "incorrect"},
        ],
    },
    "ret_hit_miss_cleaned": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_ret_hit_miss_cleaned",
        "bins": [
            {"name": "ret_hit", "session": "2", "stage": "ret", "accuracy": "correct"},
            {"name": "ret_miss", "session": "2", "stage": "ret", "accuracy": "incorrect"},
        ],
        "count_match_groups": [["ret_hit", "ret_miss"]],
    },
    "ret_hit_miss_3way": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_ret_hit_miss_3way",
        "bins": [
            {"name": "ret_hit", "session": "2", "stage": "ret", "accuracy": "correct"},
            {
                "name": "ret_miss_recog_correct",
                "session": "2",
                "stage": "ret",
                "accuracy": "incorrect",
                "recog_accuracy": "correct",
            },
            {
                "name": "ret_miss_recog_incorrect",
                "session": "2",
                "stage": "ret",
                "accuracy": "incorrect",
                "recog_accuracy": "incorrect",
            },
        ],
    },
    "ret_hit_miss_3way_cleaned": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned",
        "bins": [
            {"name": "ret_hit", "session": "2", "stage": "ret", "accuracy": "correct"},
            {
                "name": "ret_miss_recog_correct",
                "session": "2",
                "stage": "ret",
                "accuracy": "incorrect",
                "recog_accuracy": "correct",
            },
            {
                "name": "ret_miss_recog_incorrect",
                "session": "2",
                "stage": "ret",
                "accuracy": "incorrect",
                "recog_accuracy": "incorrect",
            },
        ],
        "count_match_groups": [["ret_hit", "ret_miss_recog_correct"]],
    },
    "retRating_confidence": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_retRating_confidence",
        "bins": [
            {"name": "retRating_guess", "session": "2", "stage": "retRating", "confidence": "guess"},
            {"name": "retRating_unsure", "session": "2", "stage": "retRating", "confidence": "unsure"},
            {"name": "retRating_sure", "session": "2", "stage": "retRating", "confidence": "sure"},
        ],
    },
    "retRating_sure_guess_matched": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_retRating_sure_guess_matched",
        "bins": [
            {"name": "retRating_guess", "session": "2", "stage": "retRating", "confidence": "guess"},
            {"name": "retRating_sure", "session": "2", "stage": "retRating", "confidence": "sure"},
        ],
        "count_match_groups": [["retRating_guess", "retRating_sure"]],
    },
    "ret_hit_confidence": {
        "output_dir": "./id_analysis_results_by_condition_ses-2_ret_hit_confidence",
        "bins": [
            {"name": "ret_hit_unsure", "session": "2", "stage": "ret", "accuracy": "correct", "confidence": "unsure"},
            {"name": "ret_hit_sure", "session": "2", "stage": "ret", "accuracy": "correct", "confidence": "sure"},
        ],
        "count_match_groups": [["ret_hit_unsure", "ret_hit_sure"]],
    },
}


def load_events(subject_id: str, session_id: str) -> pd.DataFrame:
    events_path = Path(BIDS_DIR) / f"sub-{subject_id}" / f"ses-{session_id}" / "func" / (
        f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_events.tsv"
    )
    if not events_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(events_path, sep="\t")
    parsed_df = df["trial_type"].astype(str).str.extract(TRIAL_TYPE_RE)
    parsed_df.columns = ["stage", "item_id", "confidence", "accuracy"]
    parsed_df["trial_type"] = df["trial_type"]
    parsed_df = parsed_df.dropna(subset=["stage", "item_id"])
    parsed_df["confidence"] = parsed_df["confidence"].fillna("").str.lower()
    parsed_df["accuracy"] = parsed_df["accuracy"].fillna("").str.lower()
    return parsed_df


def load_single_volume_img(path: Path):
    img = load_img(path)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        return index_img(img, 0)
    return img


def build_beta_index(
    subject_id: str,
    session_id: str,
    betaseries_root: Path,
) -> Dict[Tuple[str, str], Path]:
    func_dir = betaseries_root / f"sub-{subject_id}" / f"ses-{session_id}" / "func"
    index: Dict[Tuple[str, str], Path] = {}
    if not func_dir.exists():
        return index
    pattern = f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_acq-*_space-T1w_desc-*_betaseries.nii.gz"
    for path in func_dir.glob(pattern):
        match = DESC_RE.search(path.name)
        if not match:
            continue
        stage, item_id = match.groups()
        index[(stage, item_id)] = path
    return index


def filter_trials(
    df: pd.DataFrame,
    stage: str,
    confidence: Optional[Iterable[str]] = None,
    accuracy: Optional[Iterable[str]] = None,
    include_noresp: bool = False,
) -> pd.DataFrame:
    subset = df[df["stage"] == stage].copy()
    if not include_noresp:
        subset = subset[(subset["confidence"] != "noresp") & (subset["accuracy"] != "noresp")]
    if confidence is not None:
        allowed = set([confidence] if isinstance(confidence, str) else confidence)
        subset = subset[subset["confidence"].isin(allowed)]
    if accuracy is not None:
        allowed = set([accuracy] if isinstance(accuracy, str) else accuracy)
        subset = subset[subset["accuracy"].isin(allowed)]
    return subset


def build_recog_accuracy_map(df: pd.DataFrame) -> Dict[str, str]:
    recog_df = df[df["stage"] == "recog"].copy()
    recog_df = recog_df[(recog_df["confidence"] != "noresp") & (recog_df["accuracy"] != "noresp")]
    recog_map: Dict[str, str] = {}
    for _, row in recog_df.iterrows():
        item_id = row["item_id"]
        if item_id in recog_map:
            continue
        if row["accuracy"]:
            recog_map[item_id] = row["accuracy"]
    return recog_map


def get_beta_paths(
    trial_df: pd.DataFrame,
    stage: str,
    beta_index: Dict[Tuple[str, str], Path],
    recog_accuracy: Optional[str] = None,
    recog_map: Optional[Dict[str, str]] = None,
) -> List[Path]:
    paths: List[Path] = []
    for item_id in sorted(set(trial_df["item_id"].tolist())):
        if recog_accuracy is not None:
            if recog_map is None:
                continue
            if recog_map.get(item_id) != recog_accuracy:
                continue
        path = beta_index.get((stage, item_id))
        if path is not None:
            paths.append(path)
    return paths


def get_beta_paths_from_items(
    stage: str,
    item_ids: List[str],
    beta_index: Dict[Tuple[str, str], Path],
) -> List[Path]:
    paths: List[Path] = []
    for item_id in item_ids:
        path = beta_index.get((stage, item_id))
        if path is not None:
            paths.append(path)
    return paths


def get_roi_mask_path(subject_id: str, session_id: str, roi_name: str) -> Path:
    return Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{session_id}_space-T1w_desc-{roi_name}_resampled-betaseries_mask.nii.gz"
    )


def compute_roi_voxel_counts(
    sample_img, subject_id: str, session_id: str, roi_names: List[str]
) -> Dict[str, int]:
    roi_counts: Dict[str, int] = {}
    for roi_name in roi_names:
        roi_mask_path = get_roi_mask_path(subject_id, session_id, roi_name)
        if not roi_mask_path.exists():
            continue
        try:
            masker = NiftiMasker(mask_img=str(roi_mask_path))
            n_voxels = masker.fit_transform(sample_img).shape[1]
            if n_voxels > 0:
                roi_counts[roi_name] = n_voxels
        except Exception:
            continue
    return roi_counts


def estimate_id(data: np.ndarray, estimator: str, mle_k: int) -> float:
    if estimator == "twonn":
        return float(skdim.id.TwoNN().fit(data).dimension_)
    if estimator == "pr":
        if data.shape[0] < 2:
            return float("nan")
        centered = data - np.mean(data, axis=0, keepdims=True)
        singular_vals = np.linalg.svd(centered, compute_uv=False)
        if singular_vals.size == 0:
            return float("nan")
        s2 = singular_vals**2
        denom = float(np.sum(s2**2))
        if denom == 0.0:
            return float("nan")
        return float((np.sum(s2) ** 2) / denom)
    if estimator == "mle":
        estimator_obj = skdim.id.MLE()
        estimator_obj.fit(data, n_neighbors=mle_k)
        return float(estimator_obj.dimension_)
    raise ValueError(f"Unknown estimator '{estimator}'.")


def normalize_data(data: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return data
    data = data.astype(np.float64, copy=False)
    if mode == "trial_z":
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    if mode == "voxel_z":
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    if mode == "trial_l2":
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return data / norms
    raise ValueError(f"Unknown normalization mode '{mode}'.")


def compute_id_for_bin(
    beta_paths: List[Path],
    subject_id: str,
    session_id: str,
    condition_name: str,
    min_voxel_count: int,
    voxel_resamples: int,
    voxel_rng: np.random.Generator,
    estimator: str,
    mle_k: int,
    roi_names: List[str],
    normalization: str,
) -> List[Dict[str, object]]:
    beta_imgs = [load_single_volume_img(path) for path in beta_paths]
    results: List[Dict[str, object]] = []
    for roi_name in roi_names:
        roi_mask_path = get_roi_mask_path(subject_id, session_id, roi_name)
        if not roi_mask_path.exists():
            continue
        masker = NiftiMasker(mask_img=str(roi_mask_path), standardize=False)
        try:
            full_voxel_data = masker.fit_transform(beta_imgs)
            n_samples, n_original_features = full_voxel_data.shape
            voxel_indices = np.arange(n_original_features)
            dims: List[float] = []
            for _ in range(voxel_resamples):
                subsampled_indices = voxel_rng.choice(
                    voxel_indices, size=min_voxel_count, replace=False
                )
                subsampled_voxel_data = full_voxel_data[:, subsampled_indices]
                normalized_data = normalize_data(subsampled_voxel_data, normalization)
                dims.append(float(estimate_id(normalized_data, estimator, mle_k)))
            intrinsic_dim = float(np.mean(dims))
            dim_std = float(np.std(dims, ddof=1)) if voxel_resamples > 1 else 0.0
            results.append(
                {
                    "Subject": subject_id,
                    "Session": session_id,
                    "Condition": condition_name,
                    "ROI": roi_name,
                    "Intrinsic_Dim": intrinsic_dim,
                    "Intrinsic_Dim_Std": dim_std,
                    "Estimator": estimator,
                    "Normalization": normalization,
                    "Voxel_Resamples": voxel_resamples,
                    "Num_Trials": n_samples,
                    "Original_Voxels": n_original_features,
                    "Used_Voxels": min_voxel_count,
                }
            )
        except Exception as exc:
            print(f"  - ERROR sub-{subject_id} {condition_name} ROI-{roi_name}: {exc}")
    return results


def add_exclusion(excluded: Dict[str, List[str]], subject_id: str, reason: str) -> None:
    reasons = excluded.setdefault(subject_id, [])
    if reason not in reasons:
        reasons.append(reason)


def find_subjects(betaseries_root: Path) -> List[str]:
    subject_dirs = [p for p in betaseries_root.glob("sub-*") if p.is_dir()]
    return sorted(p.name.split("-")[1] for p in subject_dirs)


def run_analysis(
    analysis_key: str,
    min_trials: int,
    output_dir: str,
    voxel_resamples: int,
    trial_resamples: int,
    estimator: str,
    mle_k: int,
    roi_names: List[str],
    normalization: str,
    betaseries_root: Path,
) -> None:
    if analysis_key not in ANALYSES:
        raise ValueError(f"Unknown analysis '{analysis_key}'. Options: {sorted(ANALYSES)}")
    config = ANALYSES[analysis_key]
    bins = config["bins"]
    match_items_bins = config.get("match_items_bins", [])
    count_match_groups = config.get("count_match_groups", [])
    out_dir = Path(output_dir or config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(betaseries_root)
    all_results: List[Dict[str, object]] = []
    excluded: Dict[str, List[str]] = {}

    session_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    for subject_id in tqdm(subjects, desc="Processing Subjects"):
        base_bin_items: Dict[str, List[str]] = {}
        bin_meta: Dict[str, Dict[str, str]] = {}

        # Preload events per session
        session_events: Dict[str, pd.DataFrame] = {}
        session_recog_maps: Dict[str, Dict[str, str]] = {}
        for bin_spec in bins:
            session_id = bin_spec["session"]
            if session_id not in session_events:
                session_events[session_id] = load_events(subject_id, session_id)
            if session_events[session_id].empty:
                continue

        # Collect candidate items per bin
        for bin_spec in bins:
            session_id = bin_spec["session"]
            stage = bin_spec["stage"]
            confidence = bin_spec.get("confidence")
            accuracy = bin_spec.get("accuracy")
            include_noresp = bin_spec.get("include_noresp", False)
            recog_accuracy = bin_spec.get("recog_accuracy")

            parsed_df = session_events.get(session_id, pd.DataFrame())
            if parsed_df.empty:
                add_exclusion(excluded, subject_id, f"{bin_spec['name']}: missing events")
                continue

            recog_map = None
            if stage == "ret" and recog_accuracy is not None:
                if session_id not in session_recog_maps:
                    session_recog_maps[session_id] = build_recog_accuracy_map(parsed_df)
                recog_map = session_recog_maps[session_id]

            trial_df = filter_trials(
                parsed_df,
                stage=stage,
                confidence=confidence,
                accuracy=accuracy,
                include_noresp=include_noresp,
            )
            if trial_df.empty:
                add_exclusion(excluded, subject_id, f"{bin_spec['name']}: no trials")
                continue

            item_ids = sorted(set(trial_df["item_id"].tolist()))
            if recog_accuracy is not None:
                item_ids = [item_id for item_id in item_ids if recog_map and recog_map.get(item_id) == recog_accuracy]

            base_bin_items[bin_spec["name"]] = item_ids
            bin_meta[bin_spec["name"]] = {"session": session_id, "stage": stage}

        # Match items across specified bins
        if match_items_bins:
            intersect = None
            for cond in match_items_bins:
                items = set(base_bin_items.get(cond, []))
                if not items:
                    intersect = set()
                    break
                intersect = items if intersect is None else intersect & items
            intersect = intersect or set()
            for cond in match_items_bins:
                if cond in base_bin_items:
                    base_bin_items[cond] = sorted(intersect)

        # Pre-compute count-matching targets (deterministic given matched items)
        count_match_targets: Dict[Tuple[str, ...], int] = {}
        for group in count_match_groups:
            group_counts = [
                len(base_bin_items.get(cond, [])) for cond in group if cond in base_bin_items
            ]
            if not group_counts:
                continue
            target = min(group_counts)
            count_match_targets[tuple(group)] = target
            if target < min_trials:
                for cond in group:
                    add_exclusion(
                        excluded,
                        subject_id,
                        f"{cond}: count-matched target {target} (<{min_trials})",
                    )
                    base_bin_items[cond] = []

        trial_resamples = max(1, trial_resamples)
        for resample_idx in range(trial_resamples):
            rng = np.random.default_rng(RANDOM_SEED + int(subject_id) + resample_idx * 10000)
            voxel_rng = np.random.default_rng(
                RANDOM_SEED + int(subject_id) + 1000 + resample_idx * 20000
            )
            bin_items = {key: list(values) for key, values in base_bin_items.items()}

            # Match counts across groups (stochastic)
            for group in count_match_groups:
                target = count_match_targets.get(tuple(group))
                if target is None or target < min_trials:
                    continue
                for cond in group:
                    items = bin_items.get(cond, [])
                    if len(items) <= target:
                        continue
                    sampled = rng.choice(items, size=target, replace=False)
                    bin_items[cond] = sorted(sampled)

            # Compute ID per bin
            for bin_spec in bins:
                cond_name = bin_spec["name"]
                item_ids = bin_items.get(cond_name, [])
                if len(item_ids) < min_trials:
                    add_exclusion(
                        excluded,
                        subject_id,
                        f"{cond_name}: {len(item_ids)} trials (<{min_trials})",
                    )
                    continue

                session_id = bin_meta[cond_name]["session"]
                stage = bin_meta[cond_name]["stage"]

                beta_index = session_cache.get((subject_id, session_id), {}).get("beta_index")
                if beta_index is None:
                    beta_index = build_beta_index(subject_id, session_id, betaseries_root)
                    session_cache[(subject_id, session_id)] = {"beta_index": beta_index}

                beta_paths = get_beta_paths_from_items(stage, item_ids, beta_index)
                if len(beta_paths) < min_trials:
                    add_exclusion(
                        excluded,
                        subject_id,
                        f"{cond_name}: {len(beta_paths)} beta files (<{min_trials})",
                    )
                    continue

                session_key = (subject_id, session_id)
                cached = session_cache.get(session_key, {})
                if "min_voxel_count" not in cached:
                    sample_img = load_single_volume_img(beta_paths[0])
                    roi_counts = compute_roi_voxel_counts(
                        sample_img, subject_id, session_id, roi_names
                    )
                    if not roi_counts:
                        add_exclusion(excluded, subject_id, f"{cond_name}: missing ROI masks")
                        continue
                    cached["min_voxel_count"] = min(roi_counts.values())
                    cached["roi_counts"] = roi_counts
                    session_cache[session_key] = cached

                min_voxel_count = cached["min_voxel_count"]
                results = compute_id_for_bin(
                    beta_paths,
                    subject_id,
                    session_id,
                    cond_name,
                    min_voxel_count,
                    voxel_resamples,
                    voxel_rng,
                    estimator,
                    mle_k,
                    roi_names,
                    normalization,
                )
                for row in results:
                    row["Trial_Resample"] = resample_idx + 1
                all_results.extend(results)

    if not all_results:
        print("No results generated. Check logs for missing files or trial filters.")
        return

    results_df = pd.DataFrame(all_results)
    if "Trial_Resample" not in results_df.columns:
        results_df["Trial_Resample"] = 1
    results_df["ROI"] = pd.Categorical(results_df["ROI"], categories=roi_names, ordered=True)
    results_df = results_df.sort_values(["Subject", "Condition", "ROI"])

    full_results_path = out_dir / "group_id_full_results_by_condition.csv"
    results_df.to_csv(full_results_path, index=False)
    print(f"Full results saved to: {full_results_path}")

    if trial_resamples > 1:
        subject_means = (
            results_df.groupby(["Subject", "Condition", "ROI", "Estimator", "Normalization"])[
                "Intrinsic_Dim"
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        subject_means = subject_means.rename(
            columns={"mean": "Intrinsic_Dim", "std": "Intrinsic_Dim_Std"}
        )
        subject_means["Trial_Resamples"] = trial_resamples
        subject_means_path = out_dir / "group_id_subject_mean_by_condition.csv"
        subject_means.to_csv(subject_means_path, index=False)
        print(f"Subject mean stats saved to: {subject_means_path}")

        group_stats = (
            subject_means.groupby(["Condition", "ROI", "Estimator", "Normalization"])[
                "Intrinsic_Dim"
            ]
            .agg(["mean", "std", "sem", "count"])
            .reset_index()
        )
    else:
        group_stats = (
            results_df.groupby(["Condition", "ROI", "Estimator", "Normalization"])[
                "Intrinsic_Dim"
            ]
            .agg(["mean", "std", "sem", "count"])
            .reset_index()
        )
    summary_path = out_dir / "group_id_summary_stats_by_condition.csv"
    group_stats.to_csv(summary_path, index=False)
    print(f"Summary stats saved to: {summary_path}")

    if excluded:
        print("\n--- Exclusions ---")
        for sub_id, reasons in excluded.items():
            print(f"sub-{sub_id}: {', '.join(reasons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute binned intrinsic dimensionality (ID).")
    parser.add_argument(
        "--analysis",
        choices=sorted(ANALYSES.keys()),
        required=True,
        help="Which analysis to run.",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=MIN_TRIALS,
        help="Minimum trials required per bin.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output directory.",
    )
    parser.add_argument(
        "--voxel-resamples",
        type=int,
        default=VOXEL_RESAMPLES,
        help="Number of voxel resamples for ROI-size normalization.",
    )
    parser.add_argument(
        "--trial-resamples",
        type=int,
        default=TRIAL_RESAMPLES,
        help="Number of trial resamples for count-matched analyses.",
    )
    parser.add_argument(
        "--estimator",
        choices=["twonn", "pr", "mle"],
        default=ID_ESTIMATOR,
        help="ID estimator (twonn, pr, or mle).",
    )
    parser.add_argument(
        "--mle-k",
        type=int,
        default=MLE_K,
        help="Neighbor count for MLE estimator.",
    )
    parser.add_argument(
        "--roi-list",
        nargs="+",
        default=ROI_NAMES_DEFAULT,
        help="List of ROI names to analyze (default: ventral stream ROIs).",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "trial_z", "voxel_z", "trial_l2"],
        default=NORMALIZATION,
        help="Normalization to apply before ID estimation.",
    )
    parser.add_argument(
        "--betaseries-root",
        default=DEFAULT_BETASERIES_ROOT,
        help="Path to betaseries root (relative to BIDS_DIR unless absolute).",
    )
    args = parser.parse_args()
    betaseries_root = Path(args.betaseries_root)
    if not betaseries_root.is_absolute():
        betaseries_root = Path(BIDS_DIR) / betaseries_root

    run_analysis(
        args.analysis,
        args.min_trials,
        args.output_dir,
        args.voxel_resamples,
        args.trial_resamples,
        args.estimator,
        args.mle_k,
        args.roi_list,
        args.normalization,
        betaseries_root,
    )


if __name__ == "__main__":
    main()
