#!/usr/bin/env python3
"""
Mean activation control: compute mean beta amplitude per ROI/condition and fit
ID ~ Condition + Mean_Beta + (1|Subject) for cleaned Enc vs Ret and persistent trace.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import nibabel as nib
from nibabel.processing import resample_from_to
import re

BIDS_DIR = "/home/zlow/by84/Brandon/memoria"
TASK_NAME = "forgetexp"
MIN_TRIALS = 4
RANDOM_SEED = 42
DEFAULT_BETASERIES_ROOT = "derivatives/nibetaseries/nibetaseries"

TRIAL_TYPE_RE = re.compile(r"^([A-Za-z]+)(\d+)(?:_([^_]+))?(?:_([^_]+))?$")
DESC_RE = re.compile(r"_desc-([A-Za-z]+)(\d+)_betaseries\.nii\.gz$")

ANALYSES = {
    "enc_vs_ret_cleaned": {
        "bins": [
            {"name": "Enc", "session": "1", "stage": "Enc", "include_noresp": True},
            {"name": "ret", "session": "2", "stage": "ret"},
        ],
        "match_items_bins": ["Enc", "ret"],
        "count_match_groups": [["Enc", "ret"]],
    },
    "ret_hit_miss_3way_cleaned": {
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


def filter_trials(
    df: pd.DataFrame,
    stage: str,
    confidence=None,
    accuracy=None,
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


def find_subjects(betaseries_root: Path) -> List[str]:
    subject_dirs = [p for p in betaseries_root.glob("sub-*") if p.is_dir()]
    return sorted(p.name.split("-")[1] for p in subject_dirs)


def load_roi_mask(subject_id: str, session_id: str, roi: str, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    roi_path = Path(BIDS_DIR) / "derivatives" / "roi_masks" / f"sub-{subject_id}" / "anat" / (
        f"sub-{subject_id}_ses-{session_id}_space-T1w_desc-{roi}_resampled-betaseries_mask.nii.gz"
    )
    if not roi_path.exists():
        raise FileNotFoundError(f"Missing ROI mask: {roi_path}")
    roi_img = nib.load(str(roi_path))
    if roi_img.shape != ref_img.shape or not np.allclose(roi_img.affine, ref_img.affine):
        roi_img = resample_from_to(roi_img, ref_img, order=0)
    return roi_img


def load_single_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data


def compute_mean_beta_multi(beta_paths: List[Path], roi_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    sums = {roi: 0.0 for roi in roi_masks}
    counts = {roi: 0 for roi in roi_masks}
    for path in beta_paths:
        data = load_single_volume(path)
        for roi, mask in roi_masks.items():
            if data.shape != mask.shape:
                raise ValueError(f"Beta shape {data.shape} does not match ROI mask {mask.shape} for {path}")
            vals = data[mask]
            if vals.size == 0:
                continue
            sums[roi] += float(vals.mean())
            counts[roi] += 1
    means = {}
    for roi, total in sums.items():
        means[roi] = float(total / counts[roi]) if counts[roi] else float("nan")
    return means


def collect_mean_betas(
    analysis_key: str,
    roi_names: List[str],
    min_trials: int,
    betaseries_root: Path,
) -> pd.DataFrame:
    config = ANALYSES[analysis_key]
    bins = config["bins"]
    match_items_bins = config.get("match_items_bins", [])
    count_match_groups = config.get("count_match_groups", [])

    subjects = find_subjects(betaseries_root)
    rows = []

    session_cache: Dict[Tuple[str, str], Dict[str, object]] = {}

    for subject_id in subjects:
        bin_items: Dict[str, List[str]] = {}
        bin_meta: Dict[str, Dict[str, str]] = {}

        rng = np.random.default_rng(RANDOM_SEED + int(subject_id))

        session_events: Dict[str, pd.DataFrame] = {}
        session_recog_maps: Dict[str, Dict[str, str]] = {}
        for bin_spec in bins:
            session_id = bin_spec["session"]
            if session_id not in session_events:
                session_events[session_id] = load_events(subject_id, session_id)

        for bin_spec in bins:
            session_id = bin_spec["session"]
            stage = bin_spec["stage"]
            confidence = bin_spec.get("confidence")
            accuracy = bin_spec.get("accuracy")
            include_noresp = bin_spec.get("include_noresp", False)
            recog_accuracy = bin_spec.get("recog_accuracy")

            parsed_df = session_events.get(session_id, pd.DataFrame())
            if parsed_df.empty:
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
                continue

            item_ids = sorted(set(trial_df["item_id"].tolist()))
            if recog_accuracy is not None:
                item_ids = [item_id for item_id in item_ids if recog_map and recog_map.get(item_id) == recog_accuracy]

            bin_items[bin_spec["name"]] = item_ids
            bin_meta[bin_spec["name"]] = {"session": session_id, "stage": stage}

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
                    bin_items[cond] = []
                continue
            for cond in group:
                items = bin_items.get(cond, [])
                if len(items) <= target:
                    continue
                sampled = rng.choice(items, size=target, replace=False)
                bin_items[cond] = sorted(sampled)

        for bin_spec in bins:
            cond_name = bin_spec["name"]
            item_ids = bin_items.get(cond_name, [])
            if len(item_ids) < min_trials:
                continue
            session_id = bin_meta[cond_name]["session"]
            stage = bin_meta[cond_name]["stage"]

            beta_index = session_cache.get((subject_id, session_id), {}).get("beta_index")
            if beta_index is None:
                beta_index = build_beta_index(subject_id, session_id, betaseries_root)
                session_cache[(subject_id, session_id)] = {"beta_index": beta_index}

            beta_paths = get_beta_paths_from_items(stage, item_ids, beta_index)
            if len(beta_paths) < min_trials:
                continue

            ref_img = nib.load(str(beta_paths[0]))
            if ref_img.ndim == 4 and ref_img.shape[-1] == 1:
                ref_data = ref_img.get_fdata()[..., 0]
                ref_img = nib.Nifti1Image(ref_data, ref_img.affine, ref_img.header)

            roi_masks: Dict[str, np.ndarray] = {}
            for roi in roi_names:
                try:
                    roi_img = load_roi_mask(subject_id, session_id, roi, ref_img)
                except FileNotFoundError:
                    continue
                roi_masks[roi] = roi_img.get_fdata() > 0

            if not roi_masks:
                continue

            mean_betas = compute_mean_beta_multi(beta_paths, roi_masks)
            for roi, mean_beta in mean_betas.items():
                rows.append(
                    {
                        "Subject": subject_id,
                        "Condition": cond_name,
                        "ROI": roi,
                        "Mean_Beta": mean_beta,
                        "Num_Trials": len(beta_paths),
                    }
                )

    return pd.DataFrame(rows)


def run_lmm(
    df: pd.DataFrame,
    condition_ref: str,
    condition_test: str,
    output_dir: Path,
    analysis_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for roi in sorted(df["ROI"].unique()):
        sub = df[df["ROI"] == roi].copy()
        if sub.empty:
            continue
        formula = (
            "Intrinsic_Dim ~ C(Condition, Treatment(reference='{}')) + Mean_Beta"
        ).format(condition_ref)
        model = smf.mixedlm(formula, data=sub, groups=sub["Subject"])
        fit_note = ""
        try:
            result = model.fit(reml=False, method="lbfgs")
        except Exception as exc:
            fit_note = f"mixedlm_failed: {exc}"
            result = smf.ols(formula, data=sub).fit()

        condition_term = f"C(Condition, Treatment(reference='{condition_ref}'))[T.{condition_test}]"
        summary_rows.append(
            {
                "analysis": analysis_label,
                "ROI": roi,
                "N": sub["Subject"].nunique(),
                "condition_term": condition_term,
                "condition_beta": result.params.get(condition_term, float("nan")),
                "condition_pvalue": result.pvalues.get(condition_term, float("nan")),
                "mean_beta_coef": result.params.get("Mean_Beta", float("nan")),
                "mean_beta_pvalue": result.pvalues.get("Mean_Beta", float("nan")),
                "fit_note": fit_note,
            }
        )

        summary_path = output_dir / f"mixedlm_summary_{analysis_label}_{roi}.txt"
        summary_path.write_text(result.summary().as_text())

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / f"mean_beta_lmm_summary_{analysis_label}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean beta activation control LMMs.")
    parser.add_argument(
        "--analysis",
        choices=["enc_vs_ret_cleaned", "persistent_trace"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/mean_beta_activation",
    )
    parser.add_argument(
        "--betaseries-root",
        default=DEFAULT_BETASERIES_ROOT,
        help="Path to betaseries root (relative to BIDS_DIR unless absolute).",
    )
    parser.add_argument(
        "--roi-names",
        nargs="+",
        default=["V1", "hV4", "LOIT", "phc", "hippocampus"],
    )
    parser.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    args = parser.parse_args()
    betaseries_root = Path(args.betaseries_root)
    if not betaseries_root.is_absolute():
        betaseries_root = Path(BIDS_DIR) / betaseries_root

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis == "enc_vs_ret_cleaned":
        analysis_key = "enc_vs_ret_cleaned"
        id_results_path = Path("id_analysis_results_by_condition_enc_vs_ret_cleaned/group_id_full_results_by_condition.csv")
        conditions = ["Enc", "ret"]
        condition_ref = "Enc"
        condition_test = "ret"
        analysis_label = "enc_vs_ret_cleaned"
    else:
        analysis_key = "ret_hit_miss_3way_cleaned"
        id_results_path = Path("id_analysis_results_by_condition_ses-2_ret_hit_miss_3way_cleaned/group_id_full_results_by_condition.csv")
        conditions = ["ret_hit", "ret_miss_recog_correct"]
        condition_ref = "ret_hit"
        condition_test = "ret_miss_recog_correct"
        analysis_label = "ret_hit_vs_persistent"

    mean_beta_df = collect_mean_betas(analysis_key, args.roi_names, args.min_trials, betaseries_root)
    mean_beta_df["Subject"] = mean_beta_df["Subject"].astype(str)
    mean_beta_df = mean_beta_df[mean_beta_df["Condition"].isin(conditions)].copy()
    mean_beta_df.to_csv(output_dir / f"mean_beta_values_{analysis_label}.csv", index=False)

    id_df = pd.read_csv(id_results_path)
    id_df["Subject"] = id_df["Subject"].astype(str)
    id_df = id_df[id_df["Condition"].isin(conditions)].copy()

    merged = id_df.merge(mean_beta_df, on=["Subject", "Condition", "ROI"], how="inner")
    merged.to_csv(output_dir / f"mean_beta_merged_{analysis_label}.csv", index=False)

    run_lmm(merged, condition_ref, condition_test, output_dir, analysis_label)


if __name__ == "__main__":
    main()
