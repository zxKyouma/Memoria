#!/usr/bin/env python3
"""
Mixed-effects model for Condition x ROI with control ROIs.
Reports interaction terms comparing each ROI to the hippocampus
for the retRatingFeedback - ret contrast.
"""
import argparse
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit ID ~ Condition * ROI mixed model and report control contrasts."
    )
    parser.add_argument(
        "--input",
        default="id_analysis_results_by_condition_ses-2_forgetexp_controls/group_id_full_results_by_condition.csv",
        help="Path to group-level ID results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="id_analysis_stats/control_roi_condition_lmm",
        help="Directory to write model summary and contrast table.",
    )
    parser.add_argument(
        "--condition-ref",
        default="ret",
        help="Reference condition for the model.",
    )
    parser.add_argument(
        "--condition-test",
        default="retRatingFeedback",
        help="Condition used in the interaction contrast.",
    )
    parser.add_argument(
        "--roi-ref",
        default="hippocampus",
        help="Reference ROI for the model.",
    )
    parser.add_argument(
        "--contrast-rois",
        nargs="+",
        default=["A1", "M1", "S1", "MT", "V1"],
        help="ROIs to report interaction contrasts against hippocampus.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["ret", "retRatingFeedback"],
        help="Conditions to include in the model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df[df["Condition"].isin(args.conditions)].copy()

    formula = (
        "Intrinsic_Dim ~ C(Condition, Treatment(reference='{}'))"
        " * C(ROI, Treatment(reference='{}'))"
    ).format(args.condition_ref, args.roi_ref)

    model = smf.mixedlm(formula, data=df, groups=df["Subject"])
    result = model.fit(reml=False, method="lbfgs")

    summary_path = output_dir / "mixedlm_summary.txt"
    summary_path.write_text(result.summary().as_text())

    contrasts = []
    for roi in args.contrast_rois:
        if roi == args.roi_ref:
            continue
        term = (
            "C(Condition, Treatment(reference='{}'))[T.{}]:"
            "C(ROI, Treatment(reference='{}'))[T.{}]"
        ).format(args.condition_ref, args.condition_test, args.roi_ref, roi)
        if term not in result.params:
            contrasts.append(
                {
                    "ROI": roi,
                    "term": term,
                    "delta_diff_roi_minus_ref": float("nan"),
                    "se": float("nan"),
                    "z": float("nan"),
                    "pvalue": float("nan"),
                    "note": "term_not_found",
                }
            )
            continue
        est = result.params[term]
        se = result.bse[term]
        z = est / se if se else float("nan")
        pval = result.pvalues[term]
        contrasts.append(
            {
                "ROI": roi,
                "term": term,
                "delta_diff_roi_minus_ref": est,
                "se": se,
                "z": z,
                "pvalue": pval,
                "note": "",
            }
        )

    contrasts_df = pd.DataFrame(contrasts)
    contrasts_path = output_dir / "interaction_contrasts.csv"
    contrasts_df.to_csv(contrasts_path, index=False)

    condition_term = f"C(Condition, Treatment(reference='{args.condition_ref}'))[T.{args.condition_test}]"
    baseline_path = output_dir / "hippocampus_condition_contrast.csv"
    baseline_label = f"delta_{args.condition_test}_minus_{args.condition_ref}_at_{args.roi_ref}"
    if condition_term in result.params:
        baseline = pd.DataFrame(
            [
                {
                    "term": condition_term,
                    baseline_label: result.params[condition_term],
                    "se": result.bse[condition_term],
                    "z": result.tvalues[condition_term],
                    "pvalue": result.pvalues[condition_term],
                }
            ]
        )
        baseline.to_csv(baseline_path, index=False)
    else:
        baseline_path.write_text(f"Missing term: {condition_term}\n")


if __name__ == "__main__":
    main()
