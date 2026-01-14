#!/usr/bin/env python3
"""
OLS regression of confidence fingerprint (Sure-Guess ID drop) vs behavior.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Regress confidence fingerprint vs behavior.")
    parser.add_argument(
        "--subjects",
        default="id_analysis_stats/confidence_fingerprint_nosm_MATCHED/confidence_fingerprint_subjects.csv",
        help="Path to confidence_fingerprint_subjects.csv.",
    )
    parser.add_argument(
        "--output",
        default="id_analysis_stats/confidence_fingerprint_regression.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = []
    with Path(args.subjects).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    x = np.array([float(r["ID_Drop_SureMinusGuess"]) for r in rows])
    X = np.column_stack([np.ones(len(x)), x])

    out_rows = []
    for metric in ("Ret_Accuracy", "Recog_Accuracy"):
        y = np.array([float(r[metric]) for r in rows])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        resid = y - yhat
        n, p = X.shape
        dof = n - p
        s2 = (resid @ resid) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tvals = beta / se
        pvals = 2 * stats.t.sf(np.abs(tvals), dof)
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = (resid ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        out_rows.append(
            {
                "Metric": metric,
                "N": n,
                "Intercept": beta[0],
                "Intercept_SE": se[0],
                "Intercept_t": tvals[0],
                "Intercept_p": pvals[0],
                "Beta": beta[1],
                "Beta_SE": se[1],
                "Beta_t": tvals[1],
                "Beta_p": pvals[1],
                "R2": r2,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Metric",
                "N",
                "Intercept",
                "Intercept_SE",
                "Intercept_t",
                "Intercept_p",
                "Beta",
                "Beta_SE",
                "Beta_t",
                "Beta_p",
                "R2",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved regression summary to {out_path}")


if __name__ == "__main__":
    main()
