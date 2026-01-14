#!/usr/bin/env python3
"""
Fit ROI gradient mixed models:
  log10(ID) ~ ROI_index * Condition + (1 + ROI_index | Subject)
Fallback to random-intercept if random slope fails or is singular.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROI_ORDER = ["V1", "hV4", "LOIT", "phc", "hippocampus"]


def fit_model(df: pd.DataFrame, formula: str, re_formula: str):
    model = smf.mixedlm(formula, data=df, groups=df["Subject"], re_formula=re_formula)
    return model.fit(reml=False, method="lbfgs")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI gradient mixed model (log10 ID).")
    parser.add_argument("--input", required=True, help="Path to group_id_full_results_by_condition.csv")
    parser.add_argument("--conditions", nargs="+", required=True, help="Conditions to include")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--condition-ref", required=True, help="Reference condition")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[df["Condition"].isin(args.conditions)].copy()
    df = df[df["ROI"].isin(ROI_ORDER)].copy()
    df["ROI_index"] = df["ROI"].map({roi: idx for idx, roi in enumerate(ROI_ORDER)}).astype(float)
    df = df[df["Intrinsic_Dim"] > 0].copy()
    df["log10_ID"] = np.log10(df["Intrinsic_Dim"])
    df["Condition"] = pd.Categorical(df["Condition"], categories=args.conditions, ordered=True)
    df["ROI_index_c"] = df["ROI_index"] - df["ROI_index"].mean()

    formula = "log10_ID ~ ROI_index_c * C(Condition, Treatment(reference='{}'))".format(args.condition_ref)

    # Try random slope first
    fit_note = ""
    used_re_formula = "1 + ROI_index_c"
    try:
        result = fit_model(df, formula, re_formula=used_re_formula)
        if result.cov_re is not None and np.any(np.diag(result.cov_re) < 1e-6):
            fit_note = "random_slope_singular"
    except Exception as exc:
        used_re_formula = "1"
        result = fit_model(df, formula, re_formula=used_re_formula)
        fit_note = f"random_slope_failed: {exc}"

    summary_path = output_dir / "roi_gradient_mixedlm_summary.txt"
    summary_path.write_text(result.summary().as_text())

    meta_path = output_dir / "roi_gradient_mixedlm_meta.csv"
    pd.DataFrame(
        [{
            "fit_note": fit_note,
            "re_formula": used_re_formula,
            "n_rows": len(df),
            "n_subjects": df["Subject"].nunique(),
        }]
    ).to_csv(meta_path, index=False)

    coef_path = output_dir / "roi_gradient_mixedlm_coefficients.csv"
    pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
            "se": result.bse.values,
            "pvalue": result.pvalues.values,
        }
    ).to_csv(coef_path, index=False)

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote coefficients to {coef_path}")
    if fit_note:
        print(f"Fit note: {fit_note}")


if __name__ == "__main__":
    main()
