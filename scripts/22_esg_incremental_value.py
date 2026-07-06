"""
Script 22: ESG Incremental Value Analysis
==========================================
Tests whether ESG_composite provides predictive power beyond sector membership.
Critical for addressing the ESG data provenance concern (Phase 1, Issue C3).

Methodology:
1. Sector-only model: Regress financial metrics on sector dummies (R²_sector)
2. ESG-augmented model: Regress on sector dummies + ESG_composite (R²_full)
3. Incremental R² = R²_full - R²_sector
4. F-test for significance of ESG_composite coefficient
5. Within-sector ESG variance analysis (if ESG is just sector average,
   within-sector variance should be near zero)

Input:  data/processed/indexed_data.csv
Outputs:
  reports/tables/esg_incremental_value.csv
  reports/tables/esg_provenance_summary.csv
  reports/tables/esg_within_sector_variance.csv
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests  # BH FDR correction
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, ensure_dir
from src.constants import SCORE_COLUMNS

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Analysis 1: ESG Provenance Summary
# ---------------------------------------------------------------------------
def provenance_summary(df: pd.DataFrame) -> None:
    """Summarise ESG data provenance if a source-tracking file exists."""
    provenance_path = TABLES / "esg_data_provenance.csv"
    if not provenance_path.exists():
        print("Provenance file not found — skipping provenance summary")
        return

    prov_df = pd.read_csv(provenance_path)
    print("\n=== ESG Data Provenance Summary ===")

    prov_cols = [
        c for c in prov_df.columns
        if c.endswith("_source") or "_provenance" in c.lower()
    ]
    if not prov_cols:
        print("  No provenance columns detected in file")
        return

    provenance_counts: dict[str, int] = {}
    for col in prov_cols:
        for source, count in prov_df[col].value_counts().items():
            provenance_counts[source] = provenance_counts.get(source, 0) + count

    total_cells = sum(provenance_counts.values())
    prov_summary = pd.DataFrame([
        {"source": src, "count": cnt, "pct": round(cnt / total_cells * 100, 1)}
        for src, cnt in sorted(provenance_counts.items(), key=lambda x: -x[1])
    ])
    prov_summary.to_csv(TABLES / "esg_provenance_summary.csv", index=False)
    print(prov_summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Analysis 2: Within-Sector ESG Variance
# ---------------------------------------------------------------------------
def within_sector_variance(df: pd.DataFrame, esg_col: str) -> None:
    """Decompose ESG variance into between- and within-sector components."""
    print("\n=== Within-Sector ESG Variance ===")

    if "sector" not in df.columns:
        print("  No 'sector' column found — skipping variance decomposition")
        return

    sector_var = df.groupby("sector")[esg_col].agg(["mean", "std", "count", "var"])
    sector_var.columns = ["sector_mean", "sector_std", "n", "sector_var"]

    # Overall variance
    total_var = df[esg_col].var()

    # Between-sector variance (variance of group means, weighted by n)
    grand_mean = df[esg_col].mean()
    between_var = sum(
        row["n"] * (row["sector_mean"] - grand_mean) ** 2
        for _, row in sector_var.iterrows()
    ) / (len(df) - 1)

    # Within-sector variance (pooled within-group variance)
    n_groups = sector_var.shape[0]
    within_var = sum(
        (row["n"] - 1) * row["sector_var"]
        for _, row in sector_var.iterrows()
        if row["n"] > 1
    ) / (len(df) - n_groups)

    # Eta-squared: proportion of total variance explained by sector
    eta_sq = between_var / total_var if total_var > 0 else 0.0

    sector_var["eta_sq_contribution"] = sector_var.apply(
        lambda r: (
            r["n"] * (r["sector_mean"] - grand_mean) ** 2
            / (total_var * (len(df) - 1))
            if total_var > 0 else 0.0
        ),
        axis=1,
    )

    print(f"  Total ESG variance:       {total_var:.4f}")
    print(f"  Between-sector variance:  {between_var:.4f}")
    print(f"  Within-sector variance:   {within_var:.4f}")
    print(f"  Eta-squared (sector):     {eta_sq:.4f} ({eta_sq * 100:.1f}%)")
    print(f"  If eta² > 0.80, ESG is essentially a sector indicator")

    sector_var.to_csv(TABLES / "esg_within_sector_variance.csv")


# ---------------------------------------------------------------------------
# Analysis 3: Incremental R² Analysis
# ---------------------------------------------------------------------------
def incremental_r2(df: pd.DataFrame, esg_col: str) -> None:
    """Test whether ESG adds predictive power beyond sector dummies."""
    print("\n=== ESG Incremental R² Analysis ===")

    if "sector" not in df.columns:
        print("  No 'sector' column found — cannot perform incremental R² analysis")
        return

    # Dependent variables: composite scores that ESG should predict
    candidate_dep = [
        "financial_score", "risk_adjusted_score", "stability_score",
        # Also try direct financial metrics
        "roa", "roe", "price_volatility", "sharpe_ratio_1y",
    ]
    dependent_vars = [c for c in candidate_dep if c in df.columns]

    if not dependent_vars:
        print("  No dependent variables found — skipping incremental R² analysis")
        return

    sector_dummies = pd.get_dummies(df["sector"], drop_first=True, dtype=float)

    results: list[dict] = []
    for dep_var in dependent_vars:
        y = df[dep_var].dropna()
        esg_vals = df.loc[y.index, esg_col].dropna()
        valid_idx = y.index.intersection(esg_vals.index)

        y_clean = y.loc[valid_idx]
        X_sector = sector_dummies.loc[valid_idx].values
        X_full = np.column_stack([X_sector, df.loc[valid_idx, esg_col].values])

        if len(y_clean) < 20:
            continue

        # Model 1: Sector dummies only
        model_sector = LinearRegression().fit(X_sector, y_clean)
        r2_sector = model_sector.score(X_sector, y_clean)

        # Model 2: Sector dummies + ESG_composite
        model_full = LinearRegression().fit(X_full, y_clean)
        r2_full = model_full.score(X_full, y_clean)

        # Incremental R²
        incr_r2 = r2_full - r2_sector

        # F-test for the significance of the ESG coefficient
        n = len(y_clean)
        p_full = X_full.shape[1]
        p_sector = X_sector.shape[1]
        df_num = p_full - p_sector        # = 1 (ESG_composite)
        df_den = n - p_full - 1

        if df_den > 0 and (1 - r2_full) > 0:
            f_stat = (incr_r2 / df_num) / ((1 - r2_full) / df_den)
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
        else:
            f_stat = np.nan
            p_value = np.nan

        esg_coef = model_full.coef_[-1]

        results.append({
            "dependent_variable": dep_var,
            "n": n,
            "r2_sector_only": round(r2_sector, 4),
            "r2_sector_plus_esg": round(r2_full, 4),
            "incremental_r2": round(incr_r2, 4),
            "esg_coefficient": round(esg_coef, 4),
            "f_statistic": round(f_stat, 4) if not np.isnan(f_stat) else np.nan,
            "p_value": round(p_value, 6) if not np.isnan(p_value) else np.nan,
            "significant_at_005": p_value < 0.05 if not np.isnan(p_value) else False,
        })

        sig = (
            "***" if (not np.isnan(p_value) and p_value < 0.01)
            else ("**" if (not np.isnan(p_value) and p_value < 0.05) else "")
        )
        print(
            f"  {dep_var}: R²(sector)={r2_sector:.4f}, "
            f"R²(+ESG)={r2_full:.4f}, ΔR²={incr_r2:.4f}, "
            f"F={f_stat:.2f}, p={p_value:.4f} {sig}"
        )

    if not results:
        print("  No valid regressions could be run")
        return

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # BH FDR correction for multiple testing (Benjamini & Hochberg, 1995)
    # Testing k dependent variables simultaneously inflates Type I error:
    #   P(≥1 false positive) = 1 - (1-α)^k  (e.g., 30% for k=7, α=0.05)
    # BH FDR controls the expected proportion of false discoveries.
    # ------------------------------------------------------------------
    raw_pvals = results_df["p_value"].values
    valid_mask = ~np.isnan(raw_pvals)
    if valid_mask.sum() >= 2:
        reject, pvals_corrected, _, _ = multipletests(
            raw_pvals[valid_mask], alpha=0.05, method="fdr_bh"
        )
        fdr_p = np.full(len(raw_pvals), np.nan)
        fdr_sig = np.full(len(raw_pvals), False)
        fdr_p[valid_mask] = np.round(pvals_corrected, 6)
        fdr_sig[valid_mask] = reject
        results_df["p_value_fdr"] = fdr_p
        results_df["significant_fdr_005"] = fdr_sig
        print(f"\n  BH FDR correction applied to {int(valid_mask.sum())} simultaneous tests")
    else:
        results_df["p_value_fdr"] = results_df["p_value"]
        results_df["significant_fdr_005"] = results_df["significant_at_005"]

    results_df.to_csv(TABLES / "esg_incremental_value.csv", index=False)

    # ---- Summary interpretation ----
    avg_incr_r2 = results_df["incremental_r2"].mean()
    n_significant = int(results_df["significant_at_005"].sum())
    n_significant_fdr = int(results_df["significant_fdr_005"].sum())

    print(f"\n=== Summary ===")
    print(f"  Average incremental R²: {avg_incr_r2:.4f}")
    print(f"  Significant at p<0.05 (raw):  {n_significant}/{len(results)} dependent variables")
    print(f"  Significant at p<0.05 (FDR):  {n_significant_fdr}/{len(results)} dependent variables")

    if avg_incr_r2 < 0.01:
        print(f"  WARNING: ESG adds < 1% incremental R² beyond sector membership.")
        print(f"  ESG composite may be predominantly a sector membership indicator.")
        print(f"  This MUST be disclosed in the thesis.")
    elif avg_incr_r2 < 0.05:
        print(f"  MODERATE: ESG adds {avg_incr_r2 * 100:.1f}% incremental R² beyond sector.")
        print(f"  ESG provides some information beyond sector membership.")
    else:
        print(f"  STRONG: ESG adds {avg_incr_r2 * 100:.1f}% incremental R² beyond sector.")
        print(f"  ESG composite captures substantial within-sector variation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_indexed_data(PROJECT_ROOT, include_benchmarks=False)
    print(f"Loaded {len(df)} companies for ESG incremental value analysis")

    # Resolve ESG column name
    esg_col: str | None = None
    for candidate in ("ESG_composite", "esg_composite"):
        if candidate in df.columns:
            esg_col = candidate
            break
    if esg_col is None:
        print("ERROR: No ESG composite column found in indexed data")
        sys.exit(1)

    provenance_summary(df)
    within_sector_variance(df, esg_col)
    incremental_r2(df, esg_col)

    print("\nDone. Results saved to reports/tables/")


if __name__ == "__main__":
    main()
