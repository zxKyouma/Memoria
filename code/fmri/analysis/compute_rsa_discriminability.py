#!/usr/bin/env python3
"""Compute RSA-style discriminability (within vs between condition correlations)."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

import compute_dimensionality_binned as dim

ROI_NAMES = getattr(dim, "ROI_NAMES_IN_ORDER", dim.ROI_NAMES_DEFAULT)


def zscore_rows(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = np.nan
    zdata = (data - mean) / std
    valid = ~np.isnan(zdata).any(axis=1)
    return zdata[valid]


def mean_within_corr(zdata: np.ndarray) -> float:
    n_trials = zdata.shape[0]
    if n_trials < 2:
        return float("nan")
    corr = (zdata @ zdata.T) / zdata.shape[1]
    triu = np.triu_indices(n_trials, k=1)
    return float(np.mean(corr[triu]))


def mean_between_corr(zdata_a: np.ndarray, zdata_b: np.ndarray) -> float:
    if zdata_a.size == 0 or zdata_b.size == 0:
        return float("nan")
    corr = (zdata_a @ zdata_b.T) / zdata_a.shape[1]
    return float(np.mean(corr))


def prepare_bins(
    subject_id: str,
    bins: List[Dict[str, object]],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    bin_items: Dict[str, List[str]] = {}
    bin_meta: Dict[str, Dict[str, str]] = {}
    excluded: Dict[str, List[str]] = {}

    session_events: Dict[str, pd.DataFrame] = {}
    session_recog_maps: Dict[str, Dict[str, str]] = {}

    for bin_spec in bins:
        session_id = bin_spec["session"]
        if session_id not in session_events:
            session_events[session_id] = dim.load_events(subject_id, session_id)

    for bin_spec in bins:
        session_id = bin_spec["session"]
        stage = bin_spec["stage"]
        confidence = bin_spec.get("confidence")
        accuracy = bin_spec.get("accuracy")
        include_noresp = bin_spec.get("include_noresp", False)
        recog_accuracy = bin_spec.get("recog_accuracy")

        parsed_df = session_events.get(session_id, pd.DataFrame())
        if parsed_df.empty:
            excluded.setdefault(subject_id, []).append(f"{bin_spec['name']}: missing events")
            continue

        recog_map = None
        if stage == "ret" and recog_accuracy is not None:
            if session_id not in session_recog_maps:
                session_recog_maps[session_id] = dim.build_recog_accuracy_map(parsed_df)
            recog_map = session_recog_maps[session_id]

        trial_df = dim.filter_trials(
            parsed_df,
            stage=stage,
            confidence=confidence,
            accuracy=accuracy,
            include_noresp=include_noresp,
        )
        if trial_df.empty:
            excluded.setdefault(subject_id, []).append(f"{bin_spec['name']}: no trials")
            continue

        item_ids = sorted(set(trial_df["item_id"].tolist()))
        if recog_accuracy is not None:
            item_ids = [item_id for item_id in item_ids if recog_map and recog_map.get(item_id) == recog_accuracy]

        bin_items[bin_spec["name"]] = item_ids
        bin_meta[bin_spec["name"]] = {"session": session_id, "stage": stage}

    return bin_items, bin_meta, excluded


def apply_item_and_count_matching(
    bin_items: Dict[str, List[str]],
    match_items_bins: List[str],
    count_match_groups: List[List[str]],
    min_trials: int,
    rng: np.random.Generator,
    excluded: Dict[str, List[str]],
    subject_id: str,
) -> None:
    if match_items_bins:
        intersect = None
        for cond in match_items_bins:
            items = set(bin_items.get(cond, []))
            if not items:
                intersect = set()
                break
            intersect = items if intersect is None else intersect & items
        intersect = intersect or set()
        for cond in match_items_bins:
            if cond in bin_items:
                bin_items[cond] = sorted(intersect)

    for group in count_match_groups:
        group_counts = [len(bin_items.get(cond, [])) for cond in group if cond in bin_items]
        if not group_counts:
            continue
        target = min(group_counts)
        if target < min_trials:
            for cond in group:
                excluded.setdefault(subject_id, []).append(
                    f"{cond}: count-matched target {target} (<{min_trials})"
                )
                bin_items[cond] = []
            continue
        for cond in group:
            items = bin_items.get(cond, [])
            if len(items) <= target:
                continue
            sampled = rng.choice(items, size=target, replace=False)
            bin_items[cond] = sorted(sampled)


def run_analysis(
    analysis_key: str,
    cond_a: str,
    cond_b: str,
    min_trials: int,
    output_dir: str,
    voxel_subsample: str,
    id_results: str,
) -> None:
    if analysis_key not in dim.ANALYSES:
        raise ValueError(f"Unknown analysis '{analysis_key}'. Options: {sorted(dim.ANALYSES)}")
    config = dim.ANALYSES[analysis_key]
    bins = config["bins"]
    match_items_bins = config.get("match_items_bins", [])
    count_match_groups = config.get("count_match_groups", [])

    cond_names = {spec["name"] for spec in bins}
    if cond_a not in cond_names or cond_b not in cond_names:
        raise ValueError(f"Conditions must be in bins: {sorted(cond_names)}")

    out_dir = Path(output_dir or f"./id_analysis_stats/discriminability_{analysis_key}")
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = dim.find_subjects()
    all_results: List[Dict[str, object]] = []
    excluded: Dict[str, List[str]] = {}

    session_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    for subject_id in subjects:
        rng = np.random.default_rng(dim.RANDOM_SEED + int(subject_id))
        voxel_rng = np.random.default_rng(dim.RANDOM_SEED + int(subject_id) + 3000)

        bin_items, bin_meta, exc = prepare_bins(subject_id, bins)
        if exc:
            for sub_id, reasons in exc.items():
                excluded.setdefault(sub_id, []).extend(reasons)

        apply_item_and_count_matching(
            bin_items,
            match_items_bins,
            count_match_groups,
            min_trials,
            rng,
            excluded,
            subject_id,
        )

        items_a = bin_items.get(cond_a, [])
        items_b = bin_items.get(cond_b, [])
        if len(items_a) < min_trials or len(items_b) < min_trials:
            excluded.setdefault(subject_id, []).append(
                f"{cond_a}/{cond_b}: insufficient trials"
            )
            continue

        session_a = bin_meta[cond_a]["session"]
        session_b = bin_meta[cond_b]["session"]
        stage_a = bin_meta[cond_a]["stage"]
        stage_b = bin_meta[cond_b]["stage"]

        beta_index_a = session_cache.get((subject_id, session_a), {}).get("beta_index")
        if beta_index_a is None:
            beta_index_a = dim.build_beta_index(subject_id, session_a)
            session_cache[(subject_id, session_a)] = {"beta_index": beta_index_a}
        beta_index_b = session_cache.get((subject_id, session_b), {}).get("beta_index")
        if beta_index_b is None:
            beta_index_b = dim.build_beta_index(subject_id, session_b)
            session_cache[(subject_id, session_b)] = {"beta_index": beta_index_b}

        beta_paths_a = dim.get_beta_paths_from_items(stage_a, items_a, beta_index_a)
        beta_paths_b = dim.get_beta_paths_from_items(stage_b, items_b, beta_index_b)
        if len(beta_paths_a) < min_trials or len(beta_paths_b) < min_trials:
            excluded.setdefault(subject_id, []).append(
                f"{cond_a}/{cond_b}: missing beta files"
            )
            continue

        if voxel_subsample == "min":
            for session_id, beta_paths in [(session_a, beta_paths_a), (session_b, beta_paths_b)]:
                cache_key = (subject_id, session_id)
                cached = session_cache.get(cache_key, {})
                if "min_voxel_count" in cached:
                    continue
                sample_img = dim.load_single_volume_img(beta_paths[0])
                roi_counts = dim.compute_roi_voxel_counts(sample_img, subject_id, session_id, ROI_NAMES)
                if not roi_counts:
                    excluded.setdefault(subject_id, []).append(
                        f"{cond_a}/{cond_b}: missing ROI masks"
                    )
                    continue
                cached["min_voxel_count"] = min(roi_counts.values())
                cached["roi_counts"] = roi_counts
                session_cache[cache_key] = cached

        beta_imgs_a = [dim.load_single_volume_img(path) for path in beta_paths_a]
        beta_imgs_b = [dim.load_single_volume_img(path) for path in beta_paths_b]

        for roi_name in ROI_NAMES:
            roi_mask_a = dim.get_roi_mask_path(subject_id, session_a, roi_name)
            roi_mask_b = dim.get_roi_mask_path(subject_id, session_b, roi_name)
            if not roi_mask_a.exists() or not roi_mask_b.exists():
                continue

            masker_a = NiftiMasker(mask_img=str(roi_mask_a), standardize=False)
            masker_b = NiftiMasker(mask_img=str(roi_mask_b), standardize=False)
            try:
                data_a = masker_a.fit_transform(beta_imgs_a)
                data_b = masker_b.fit_transform(beta_imgs_b)
                n_trials_a, n_vox_a = data_a.shape
                n_trials_b, n_vox_b = data_b.shape
                used_voxels = min(n_vox_a, n_vox_b)
                if voxel_subsample == "min":
                    min_vox_a = session_cache[(subject_id, session_a)]["min_voxel_count"]
                    min_vox_b = session_cache[(subject_id, session_b)]["min_voxel_count"]
                    min_vox = min(min_vox_a, min_vox_b)
                    if n_vox_a >= min_vox and n_vox_b >= min_vox:
                        data_a = data_a[:, voxel_rng.choice(np.arange(n_vox_a), size=min_vox, replace=False)]
                        data_b = data_b[:, voxel_rng.choice(np.arange(n_vox_b), size=min_vox, replace=False)]
                        used_voxels = min_vox

                z_a = zscore_rows(data_a)
                z_b = zscore_rows(data_b)
                within_a = mean_within_corr(z_a)
                within_b = mean_within_corr(z_b)
                between = mean_between_corr(z_a, z_b)

                all_results.append(
                    {
                        "Subject": subject_id,
                        "Condition": cond_a,
                        "ROI": roi_name,
                        "Within": within_a,
                        "Between": between,
                        "Discriminability": within_a - between,
                        "Num_Trials": n_trials_a,
                        "Used_Voxels": used_voxels,
                    }
                )
                all_results.append(
                    {
                        "Subject": subject_id,
                        "Condition": cond_b,
                        "ROI": roi_name,
                        "Within": within_b,
                        "Between": between,
                        "Discriminability": within_b - between,
                        "Num_Trials": n_trials_b,
                        "Used_Voxels": used_voxels,
                    }
                )
            except Exception as exc:
                excluded.setdefault(subject_id, []).append(
                    f"{cond_a}/{cond_b} ROI-{roi_name}: {exc}"
                )

    if not all_results:
        print("No discriminability results generated. Check logs for missing files or trial filters.")
        return

    results_df = pd.DataFrame(all_results)
    results_df["Subject"] = results_df["Subject"].astype(str)
    results_df["ROI"] = pd.Categorical(
        results_df["ROI"], categories=ROI_NAMES, ordered=True
    )
    results_df = results_df.sort_values(["Subject", "Condition", "ROI"])

    full_results_path = out_dir / "discriminability_results.csv"
    results_df.to_csv(full_results_path, index=False)
    print(f"Discriminability results saved to: {full_results_path}")

    summary = (
        results_df.groupby(["Condition", "ROI"])
        .agg({
            "Within": "mean",
            "Between": "mean",
            "Discriminability": "mean",
            "Subject": "count",
        })
        .rename(columns={"Subject": "N"})
        .reset_index()
    )
    summary_path = out_dir / "discriminability_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Discriminability summary saved to: {summary_path}")

    if id_results:
        id_df = pd.read_csv(id_results)
        id_df["Subject"] = id_df["Subject"].astype(str)
        merged = results_df.merge(
            id_df,
            on=["Subject", "Condition", "ROI"],
            how="inner",
        )
        merge_path = out_dir / "discriminability_id_merged.csv"
        merged.to_csv(merge_path, index=False)
        print(f"Merged discriminability/ID saved to: {merge_path}")

        corr_rows: List[Dict[str, object]] = []
        for (condition, roi), sub_df in merged.groupby(["Condition", "ROI"]):
            x = sub_df["Discriminability"].to_numpy(dtype=float)
            y = sub_df["Intrinsic_Dim"].to_numpy(dtype=float)
            if len(x) < 3:
                r = float("nan")
            else:
                x = x - x.mean()
                y = y - y.mean()
                denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
                r = float(np.sum(x * y) / denom) if denom != 0 else float("nan")
            corr_rows.append(
                {
                    "Condition": condition,
                    "ROI": roi,
                    "Discriminability_ID_r": r,
                    "N": int(len(sub_df)),
                }
            )
        corr_df = pd.DataFrame(corr_rows)
        corr_path = out_dir / "discriminability_id_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        print(f"Discriminability/ID correlations saved to: {corr_path}")

    if excluded:
        print("\n--- Exclusions ---")
        for sub_id, reasons in excluded.items():
            print(f"sub-{sub_id}: {', '.join(reasons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RSA-style discriminability metrics.")
    parser.add_argument(
        "--analysis",
        choices=sorted(dim.ANALYSES.keys()),
        required=True,
        help="Which analysis bins to use.",
    )
    parser.add_argument("--condition-a", required=True, help="First condition name")
    parser.add_argument("--condition-b", required=True, help="Second condition name")
    parser.add_argument(
        "--min-trials",
        type=int,
        default=dim.MIN_TRIALS,
        help="Minimum trials required per bin.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override output directory.",
    )
    parser.add_argument(
        "--voxel-subsample",
        choices=["all", "min"],
        default="all",
        help="Use all ROI voxels or subsample to the minimum ROI size.",
    )
    parser.add_argument(
        "--id-results",
        default="",
        help="Optional ID results CSV to correlate with discriminability.",
    )
    args = parser.parse_args()

    run_analysis(
        args.analysis,
        args.condition_a,
        args.condition_b,
        args.min_trials,
        args.output_dir,
        args.voxel_subsample,
        args.id_results,
    )


if __name__ == "__main__":
    main()
