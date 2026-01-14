#!/usr/bin/env python3
"""
Compute split-half pattern reliability (odd/even trials) for ROI patterns by condition.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

import compute_dimensionality_binned as dim


def split_indices(n_trials: int, split: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if split == "odd-even":
        idx = np.arange(n_trials)
        return idx[::2], idx[1::2]
    indices = np.arange(n_trials)
    rng.shuffle(indices)
    half = n_trials // 2
    return indices[:half], indices[half:]


def compute_reliability(patterns: np.ndarray, split: str, rng: np.random.Generator) -> float:
    n_trials = patterns.shape[0]
    if n_trials < 2:
        return float("nan")
    idx_a, idx_b = split_indices(n_trials, split, rng)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    mean_a = patterns[idx_a].mean(axis=0)
    mean_b = patterns[idx_b].mean(axis=0)
    if np.std(mean_a) == 0 or np.std(mean_b) == 0:
        return float("nan")
    return float(np.corrcoef(mean_a, mean_b)[0, 1])


def prepare_bins(
    subject_id: str,
    bins: List[Dict[str, object]],
    min_trials: int,
    rng: np.random.Generator,
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
    min_trials: int,
    output_dir: str,
    split: str,
    voxel_subsample: str,
    betaseries_root: Path,
) -> None:
    if analysis_key not in dim.ANALYSES:
        raise ValueError(f"Unknown analysis '{analysis_key}'. Options: {sorted(dim.ANALYSES)}")
    config = dim.ANALYSES[analysis_key]
    bins = config["bins"]
    match_items_bins = config.get("match_items_bins", [])
    count_match_groups = config.get("count_match_groups", [])

    out_dir = Path(output_dir or f"./id_analysis_stats/pattern_reliability_{analysis_key}")
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = dim.find_subjects(betaseries_root)
    all_results: List[Dict[str, object]] = []
    excluded: Dict[str, List[str]] = {}

    session_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    for subject_id in subjects:
        rng = np.random.default_rng(dim.RANDOM_SEED + int(subject_id))
        voxel_rng = np.random.default_rng(dim.RANDOM_SEED + int(subject_id) + 2000)

        bin_items, bin_meta, exc = prepare_bins(subject_id, bins, min_trials, rng)
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

        for bin_spec in bins:
            cond_name = bin_spec["name"]
            item_ids = bin_items.get(cond_name, [])
            if len(item_ids) < min_trials:
                excluded.setdefault(subject_id, []).append(
                    f"{cond_name}: {len(item_ids)} trials (<{min_trials})"
                )
                continue

            session_id = bin_meta[cond_name]["session"]
            stage = bin_meta[cond_name]["stage"]

            beta_index = session_cache.get((subject_id, session_id), {}).get("beta_index")
            if beta_index is None:
                beta_index = dim.build_beta_index(subject_id, session_id, betaseries_root)
                session_cache[(subject_id, session_id)] = {"beta_index": beta_index}

            beta_paths = dim.get_beta_paths_from_items(stage, item_ids, beta_index)
            if len(beta_paths) < min_trials:
                excluded.setdefault(subject_id, []).append(
                    f"{cond_name}: {len(beta_paths)} beta files (<{min_trials})"
                )
                continue

            session_key = (subject_id, session_id)
            cached = session_cache.get(session_key, {})
            if voxel_subsample == "min" and "min_voxel_count" not in cached:
                sample_img = dim.load_single_volume_img(beta_paths[0])
                roi_counts = dim.compute_roi_voxel_counts(sample_img, subject_id, session_id)
                if not roi_counts:
                    excluded.setdefault(subject_id, []).append(f"{cond_name}: missing ROI masks")
                    continue
                cached["min_voxel_count"] = min(roi_counts.values())
                cached["roi_counts"] = roi_counts
                session_cache[session_key] = cached

            for roi_name in dim.ROI_NAMES_IN_ORDER:
                roi_mask_path = dim.get_roi_mask_path(subject_id, session_id, roi_name)
                if not roi_mask_path.exists():
                    continue

                masker = NiftiMasker(mask_img=str(roi_mask_path), standardize=False)
                try:
                    beta_imgs = [dim.load_single_volume_img(path) for path in beta_paths]
                    voxel_data = masker.fit_transform(beta_imgs)
                    n_trials, n_voxels = voxel_data.shape
                    used_voxels = n_voxels
                    if voxel_subsample == "min":
                        min_voxel_count = cached.get("min_voxel_count")
                        if min_voxel_count and n_voxels >= min_voxel_count:
                            voxel_indices = np.arange(n_voxels)
                            subsampled = voxel_rng.choice(
                                voxel_indices, size=min_voxel_count, replace=False
                            )
                            voxel_data = voxel_data[:, subsampled]
                            used_voxels = min_voxel_count
                    reliability = compute_reliability(voxel_data, split, rng)
                    all_results.append(
                        {
                            "Subject": subject_id,
                            "Session": session_id,
                            "Condition": cond_name,
                            "ROI": roi_name,
                            "Reliability": reliability,
                            "Num_Trials": n_trials,
                            "Original_Voxels": n_voxels,
                            "Used_Voxels": used_voxels,
                            "Split": split,
                            "Voxel_Subsample": voxel_subsample,
                        }
                    )
                except Exception as exc:
                    excluded.setdefault(subject_id, []).append(
                        f"{cond_name} ROI-{roi_name}: {exc}"
                    )

    if not all_results:
        print("No reliability results generated. Check logs for missing files or trial filters.")
        return

    results_df = pd.DataFrame(all_results)
    results_df["ROI"] = pd.Categorical(
        results_df["ROI"], categories=dim.ROI_NAMES_IN_ORDER, ordered=True
    )
    results_df = results_df.sort_values(["Subject", "Condition", "ROI"])

    full_results_path = out_dir / "pattern_reliability.csv"
    results_df.to_csv(full_results_path, index=False)
    print(f"Reliability results saved to: {full_results_path}")

    group_stats = (
        results_df.groupby(["Condition", "ROI"])["Reliability"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    group_stats["sem"] = group_stats["std"] / np.sqrt(group_stats["count"])
    summary_path = out_dir / "pattern_reliability_summary.csv"
    group_stats.to_csv(summary_path, index=False)
    print(f"Reliability summary saved to: {summary_path}")

    if excluded:
        print("\n--- Exclusions ---")
        for sub_id, reasons in excluded.items():
            print(f"sub-{sub_id}: {', '.join(reasons)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute split-half reliability for ROI patterns.")
    parser.add_argument(
        "--analysis",
        choices=sorted(dim.ANALYSES.keys()),
        required=True,
        help="Which analysis bins to use.",
    )
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
        "--split",
        choices=["odd-even", "random"],
        default="odd-even",
        help="Split strategy for trials.",
    )
    parser.add_argument(
        "--voxel-subsample",
        choices=["all", "min"],
        default="all",
        help="Use all ROI voxels or subsample to the minimum ROI size.",
    )
    parser.add_argument(
        "--betaseries-root",
        default=dim.DEFAULT_BETASERIES_ROOT,
        help="Path to betaseries root (relative to BIDS_DIR unless absolute).",
    )
    args = parser.parse_args()
    betaseries_root = Path(args.betaseries_root)
    if not betaseries_root.is_absolute():
        betaseries_root = Path(dim.BIDS_DIR) / betaseries_root

    run_analysis(
        args.analysis,
        args.min_trials,
        args.output_dir,
        args.split,
        args.voxel_subsample,
        betaseries_root,
    )


if __name__ == "__main__":
    main()
