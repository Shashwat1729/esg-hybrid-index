"""
Step 24: Geographic Normalization Bias Assessment
==================================================
US and Indian companies are normalized together in a single pool.
Different accounting standards (US GAAP vs Ind AS), market structures,
and ESG disclosure regimes may systematically advantage one geography.

This script tests whether pooled cross-country normalization introduces
systematic geographic bias by:

1. Splitting the mid-cap universe into US and India subsets
2. For each factor score, running:
   a. Mann-Whitney U test (non-parametric) for US vs India difference
   b. Cohen's d effect size to quantify the magnitude of any gap
   c. Within-geography rank correlation vs pooled ranking (Spearman)
3. Testing whether country is a significant predictor of residual factor
   scores after controlling for sector (OLS with sector dummies)
4. Robustness check: re-normalizing within each geography independently
   and comparing the resulting rankings to the pooled rankings
5. Outputting summary tables for inclusion in the thesis

Input:  data/processed/indexed_data.csv
Outputs:
  reports/tables/geographic_robustness_mann_whitney.csv
  reports/tables/geographic_robustness_effect_sizes.csv
  reports/tables/geographic_robustness_rank_correlation.csv
  reports/tables/geographic_robustness_sector_residuals.csv
  reports/tables/geographic_robustness_renorm_comparison.csv
  reports/tables/geographic_robustness_summary.csv
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, ensure_dir
from src.constants import SCORE_COLUMNS

TABLES = ensure_dir(PROJECT_ROOT / "reports" / "tables")

# Minimum number of companies per geography to run a test
MIN_GEO_N = 5

# Bonferroni-corrected significance threshold (adjusted in main based on
# the actual number of tests)
ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. Mann-Whitney U tests: US vs India per factor
# ---------------------------------------------------------------------------

def mann_whitney_tests(
    df_us: pd.DataFrame,
    df_in: pd.DataFrame,
    score_cols: list[str],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Non-parametric two-sample test for each factor score.

    Returns a DataFrame with U-statistic, p-value, and significance flag
    (Bonferroni-corrected).
    """
    n_tests = len(score_cols)
    corrected_alpha = alpha / max(n_tests, 1)

    rows = []
    for col in score_cols:
        us_vals = df_us[col].dropna()
        in_vals = df_in[col].dropna()

        if len(us_vals) < MIN_GEO_N or len(in_vals) < MIN_GEO_N:
            rows.append({
                "factor": col,
                "n_us": len(us_vals),
                "n_india": len(in_vals),
                "U_statistic": np.nan,
                "p_value": np.nan,
                "significant_bonferroni": False,
                "us_median": np.nan,
                "india_median": np.nan,
                "median_diff": np.nan,
            })
            continue

        u_stat, p_val = sp_stats.mannwhitneyu(
            us_vals, in_vals, alternative="two-sided"
        )

        rows.append({
            "factor": col,
            "n_us": len(us_vals),
            "n_india": len(in_vals),
            "U_statistic": round(u_stat, 2),
            "p_value": round(p_val, 6),
            "significant_bonferroni": p_val < corrected_alpha,
            "us_median": round(us_vals.median(), 3),
            "india_median": round(in_vals.median(), 3),
            "median_diff": round(us_vals.median() - in_vals.median(), 3),
        })

    result = pd.DataFrame(rows)
    result["bonferroni_alpha"] = round(corrected_alpha, 6)
    return result


# ---------------------------------------------------------------------------
# 2. Cohen's d effect size
# ---------------------------------------------------------------------------

def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Compute Cohen's d (pooled SD denominator)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_std


def effect_size_table(
    df_us: pd.DataFrame,
    df_in: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """Cohen's d for each factor, with conventional interpretation."""
    rows = []
    for col in score_cols:
        us_vals = df_us[col].dropna()
        in_vals = df_in[col].dropna()
        d = cohens_d(us_vals, in_vals)

        abs_d = abs(d) if not np.isnan(d) else np.nan
        if np.isnan(abs_d):
            interpretation = "insufficient data"
        elif abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        rows.append({
            "factor": col,
            "cohens_d": round(d, 4) if not np.isnan(d) else np.nan,
            "abs_cohens_d": round(abs_d, 4) if not np.isnan(abs_d) else np.nan,
            "interpretation": interpretation,
            "us_mean": round(us_vals.mean(), 3) if len(us_vals) else np.nan,
            "india_mean": round(in_vals.mean(), 3) if len(in_vals) else np.nan,
            "us_std": round(us_vals.std(), 3) if len(us_vals) else np.nan,
            "india_std": round(in_vals.std(), 3) if len(in_vals) else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Within-geography rank correlation vs pooled ranking
# ---------------------------------------------------------------------------

def rank_correlation_analysis(
    df: pd.DataFrame,
    df_us: pd.DataFrame,
    df_in: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """Percentile distortion: pooled vs within-geography percentile.

    For each geography, compute each company's percentile rank in the
    pooled sample and in their own geography.  If pooling introduces bias,
    one geography's companies will systematically have higher (or lower)
    pooled percentiles than their within-geography percentiles would
    suggest.

    Reports:
    - Spearman rho between pooled and within-geography ranks (should be
      1.0 by construction when no ties change — serves as a sanity check)
    - Mean and max absolute percentile shift
    - Mean signed percentile shift (positive = geography advantaged by
      pooling, negative = disadvantaged)
    """
    rows = []
    n_total = len(df)

    for col in score_cols:
        # Pooled percentile (0–100, higher = better)
        pooled_pctile = df[col].rank(pct=True) * 100

        for label, sub_df in [("US", df_us), ("India", df_in)]:
            vals = sub_df[col].dropna()
            n_geo = len(vals)
            if n_geo < MIN_GEO_N:
                rows.append({
                    "factor": col,
                    "geography": label,
                    "n": n_geo,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "mean_pooled_pctile": np.nan,
                    "mean_within_pctile": np.nan,
                    "mean_pctile_shift": np.nan,
                    "mean_abs_pctile_shift": np.nan,
                    "max_abs_pctile_shift": np.nan,
                })
                continue

            within_pctile = vals.rank(pct=True) * 100
            pooled_sub = pooled_pctile.loc[vals.index]

            # Rank-order correlation (sanity check — should be ~1.0)
            within_rank = vals.rank(ascending=False)
            pooled_rank_sub = df[col].rank(ascending=False).loc[vals.index]
            rho, p = sp_stats.spearmanr(within_rank, pooled_rank_sub)

            # Percentile shift: pooled - within
            # Positive means company looks better in the pooled context
            pctile_shift = pooled_sub.values - within_pctile.values

            rows.append({
                "factor": col,
                "geography": label,
                "n": n_geo,
                "spearman_rho": round(rho, 4),
                "spearman_p": round(p, 6),
                "mean_pooled_pctile": round(pooled_sub.mean(), 2),
                "mean_within_pctile": round(within_pctile.mean(), 2),
                "mean_pctile_shift": round(np.mean(pctile_shift), 2),
                "mean_abs_pctile_shift": round(np.mean(np.abs(pctile_shift)), 2),
                "max_abs_pctile_shift": round(np.max(np.abs(pctile_shift)), 2),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Sector-controlled residual test: is country still significant?
# ---------------------------------------------------------------------------

def sector_controlled_country_test(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """OLS regression: score ~ country + sector_dummies.

    Tests whether country has a statistically significant coefficient after
    controlling for sector composition. Uses dummy-variable regression
    implemented with numpy/scipy to avoid a statsmodels dependency.
    """
    if "sector" not in df.columns or "country" not in df.columns:
        return pd.DataFrame()

    # Encode country as binary (US=1, India=0)
    country_code = (df["country"] == "US").astype(float).values

    # Sector dummies (drop first to avoid collinearity)
    sector_dummies = pd.get_dummies(df["sector"], drop_first=True, dtype=float)

    rows = []
    n_tests = len(score_cols)
    corrected_alpha = ALPHA / max(n_tests, 1)

    for col in score_cols:
        y = df[col].values
        valid_mask = ~np.isnan(y)
        y_valid = y[valid_mask]
        n = valid_mask.sum()

        if n < MIN_GEO_N + sector_dummies.shape[1] + 2:
            rows.append({
                "factor": col,
                "country_coef": np.nan,
                "country_t_stat": np.nan,
                "country_p_value": np.nan,
                "significant_bonferroni": False,
                "r_squared": np.nan,
                "n": n,
            })
            continue

        # Build design matrix: [intercept, country, sector_dummies]
        X = np.column_stack([
            np.ones(n),
            country_code[valid_mask],
            sector_dummies.values[valid_mask],
        ])

        # OLS via normal equations: beta = (X'X)^-1 X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y_valid
            beta = np.linalg.solve(XtX, Xty)

            y_hat = X @ beta
            residuals = y_valid - y_hat
            dof = n - X.shape[1]

            if dof <= 0:
                raise ValueError("Insufficient degrees of freedom")

            mse = (residuals ** 2).sum() / dof
            # Variance-covariance matrix of coefficients
            cov_beta = mse * np.linalg.inv(XtX)

            # Country coefficient is index 1
            country_coef = beta[1]
            country_se = np.sqrt(cov_beta[1, 1])
            t_stat = country_coef / country_se if country_se > 0 else np.nan

            # Two-sided p-value from t-distribution
            if not np.isnan(t_stat):
                p_val = 2 * sp_stats.t.sf(abs(t_stat), df=dof)
            else:
                p_val = np.nan

            # R-squared
            ss_res = (residuals ** 2).sum()
            ss_tot = ((y_valid - y_valid.mean()) ** 2).sum()
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            rows.append({
                "factor": col,
                "country_coef": round(country_coef, 4),
                "country_se": round(country_se, 4),
                "country_t_stat": round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                "country_p_value": round(p_val, 6) if not np.isnan(p_val) else np.nan,
                "significant_bonferroni": (
                    p_val < corrected_alpha if not np.isnan(p_val) else False
                ),
                "r_squared": round(r_sq, 4) if not np.isnan(r_sq) else np.nan,
                "n": n,
            })
        except np.linalg.LinAlgError:
            rows.append({
                "factor": col,
                "country_coef": np.nan,
                "country_se": np.nan,
                "country_t_stat": np.nan,
                "country_p_value": np.nan,
                "significant_bonferroni": False,
                "r_squared": np.nan,
                "n": n,
            })

    result = pd.DataFrame(rows)
    result["bonferroni_alpha"] = round(corrected_alpha, 6)
    return result


# ---------------------------------------------------------------------------
# 5. Robustness: re-normalize within each geography, compare rankings
# ---------------------------------------------------------------------------

def within_geography_renorm(
    df: pd.DataFrame,
    df_us: pd.DataFrame,
    df_in: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """Re-normalize scores within each geography and compare to pooled.

    For each geography:
    1. Z-score within that subset -> map to mean=50, std=10, clip [0,100]
    2. Rank companies within geography on the re-normalized scores
    3. Rank companies within geography on the original pooled scores
    4. Compute Spearman rho between the two rank orderings

    High correlation means the pooled normalization does not distort
    within-geography orderings; low correlation means it does.
    """
    rows = []

    for col in score_cols:
        for label, sub_df in [("US", df_us), ("India", df_in)]:
            vals = sub_df[col].dropna()
            if len(vals) < MIN_GEO_N:
                rows.append({
                    "factor": col,
                    "geography": label,
                    "n": len(vals),
                    "spearman_rho_pooled_vs_renorm": np.nan,
                    "spearman_p": np.nan,
                    "pooled_mean": np.nan,
                    "pooled_std": np.nan,
                    "renorm_mean": np.nan,
                    "renorm_std": np.nan,
                    "mean_rank_shift": np.nan,
                    "max_rank_shift": np.nan,
                })
                continue

            geo_mean = vals.mean()
            geo_std = vals.std()

            if geo_std == 0 or pd.isna(geo_std):
                # All values identical in this geography — rank is arbitrary
                rows.append({
                    "factor": col,
                    "geography": label,
                    "n": len(vals),
                    "spearman_rho_pooled_vs_renorm": np.nan,
                    "spearman_p": np.nan,
                    "pooled_mean": round(geo_mean, 3),
                    "pooled_std": 0.0,
                    "renorm_mean": 50.0,
                    "renorm_std": 0.0,
                    "mean_rank_shift": np.nan,
                    "max_rank_shift": np.nan,
                })
                continue

            # Re-normalize within geography
            renorm = ((vals - geo_mean) / geo_std) * 10 + 50
            renorm = renorm.clip(0, 100)

            # Rank on pooled (original) scores vs re-normalized scores
            rank_pooled = vals.rank(ascending=False)
            rank_renorm = renorm.rank(ascending=False)

            rho, p = sp_stats.spearmanr(rank_pooled, rank_renorm)
            rank_diff = (rank_pooled - rank_renorm).abs()

            rows.append({
                "factor": col,
                "geography": label,
                "n": len(vals),
                "spearman_rho_pooled_vs_renorm": round(rho, 4),
                "spearman_p": round(p, 6),
                "pooled_mean": round(vals.mean(), 3),
                "pooled_std": round(vals.std(), 3),
                "renorm_mean": round(renorm.mean(), 3),
                "renorm_std": round(renorm.std(), 3),
                "mean_rank_shift": round(rank_diff.mean(), 2),
                "max_rank_shift": int(rank_diff.max()),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("STEP 24: GEOGRAPHIC NORMALIZATION BIAS ASSESSMENT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = load_indexed_data(PROJECT_ROOT, include_benchmarks=False)
    print(f"[OK] Loaded {len(df)} mid-cap companies")

    if "country" not in df.columns:
        print("[ERROR] 'country' column not found — cannot run geographic "
              "robustness analysis.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Split by geography
    # ------------------------------------------------------------------
    df_us = df[df["country"] == "US"].copy()
    df_in = df[df["country"] == "India"].copy()

    n_us, n_in = len(df_us), len(df_in)
    n_other = len(df) - n_us - n_in
    print(f"[OK] US companies  : {n_us}")
    print(f"[OK] India companies: {n_in}")
    if n_other > 0:
        print(f"[WARN] {n_other} companies with other/missing country — excluded "
              "from geographic tests")

    if n_us < MIN_GEO_N or n_in < MIN_GEO_N:
        print(f"[ERROR] Need at least {MIN_GEO_N} companies per geography. "
              f"Got US={n_us}, India={n_in}. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Identify score columns
    # ------------------------------------------------------------------
    score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
    pref_cols = sorted(c for c in df.columns if c.startswith("pref_"))
    all_score_cols = score_cols + pref_cols
    print(f"[OK] {len(score_cols)} core factors + {len(pref_cols)} preference "
          f"profiles = {len(all_score_cols)} score columns")

    if "sector" in df.columns:
        print(f"[OK] {df['sector'].nunique()} sectors available for controls")
        # Show sector x country distribution
        cross = pd.crosstab(df["sector"], df["country"])
        print("\nSector x Country distribution:")
        print(cross.to_string())
        print()

    # ==================================================================
    # ANALYSIS 1: Mann-Whitney U tests
    # ==================================================================
    print("-" * 70)
    print("TEST 1: Mann-Whitney U (non-parametric location test)")
    print("-" * 70)
    mw_results = mann_whitney_tests(df_us, df_in, all_score_cols)
    mw_results.to_csv(TABLES / "geographic_robustness_mann_whitney.csv",
                      index=False)

    n_sig = mw_results["significant_bonferroni"].sum()
    print(f"  Factors with significant US-India difference "
          f"(Bonferroni α={mw_results['bonferroni_alpha'].iloc[0]:.4f}): "
          f"{n_sig}/{len(all_score_cols)}")
    for _, row in mw_results[mw_results["significant_bonferroni"]].iterrows():
        print(f"    {row['factor']:30s}  p={row['p_value']:.6f}  "
              f"US median={row['us_median']:.1f}  "
              f"India median={row['india_median']:.1f}")
    print()

    # ==================================================================
    # ANALYSIS 2: Cohen's d effect sizes
    # ==================================================================
    print("-" * 70)
    print("TEST 2: Cohen's d effect sizes")
    print("-" * 70)
    es_results = effect_size_table(df_us, df_in, all_score_cols)
    es_results.to_csv(TABLES / "geographic_robustness_effect_sizes.csv",
                      index=False)

    print(f"  {'Factor':<30s} {'d':>8s} {'|d|':>8s} {'Interp.':<12s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*12}")
    for _, row in es_results.iterrows():
        d_str = f"{row['cohens_d']:.3f}" if not pd.isna(row["cohens_d"]) else "N/A"
        abs_str = f"{row['abs_cohens_d']:.3f}" if not pd.isna(row["abs_cohens_d"]) else "N/A"
        print(f"  {row['factor']:<30s} {d_str:>8s} {abs_str:>8s} "
              f"{row['interpretation']:<12s}")

    n_medium_or_large = es_results["abs_cohens_d"].apply(
        lambda x: x >= 0.5 if not pd.isna(x) else False
    ).sum()
    print(f"\n  Factors with medium/large effect: {n_medium_or_large}/{len(all_score_cols)}")
    print()

    # ==================================================================
    # ANALYSIS 3: Within-geography rank correlation vs pooled
    # ==================================================================
    print("-" * 70)
    print("TEST 3: Within-geography rank correlation vs pooled ranking")
    print("-" * 70)
    rc_results = rank_correlation_analysis(df, df_us, df_in, all_score_cols)
    rc_results.to_csv(TABLES / "geographic_robustness_rank_correlation.csv",
                      index=False)

    for geo in ["US", "India"]:
        sub = rc_results[rc_results["geography"] == geo]
        valid_rho = sub["spearman_rho"].dropna()
        valid_shift = sub["mean_pctile_shift"].dropna()
        if len(valid_rho) > 0:
            print(f"  {geo:6s}: rank ρ = {valid_rho.mean():.3f} (sanity check)")
            mean_shift = valid_shift.mean() if len(valid_shift) else 0
            abs_shift = sub["mean_abs_pctile_shift"].dropna()
            direction = "advantaged" if mean_shift > 0 else "disadvantaged"
            print(f"  {' ':6s}  mean percentile shift = {mean_shift:+.1f}pp "
                  f"({direction} by pooling)")
            if len(abs_shift) > 0:
                print(f"  {' ':6s}  mean |shift| = {abs_shift.mean():.1f}pp, "
                      f"max |shift| = {sub['max_abs_pctile_shift'].dropna().max():.1f}pp")
        else:
            print(f"  {geo:6s}: insufficient data")
    print()

    # ==================================================================
    # ANALYSIS 4: Sector-controlled country residual test
    # ==================================================================
    print("-" * 70)
    print("TEST 4: Country coefficient after sector control (OLS)")
    print("-" * 70)
    sr_results = sector_controlled_country_test(df, all_score_cols)

    if len(sr_results) > 0:
        sr_results.to_csv(
            TABLES / "geographic_robustness_sector_residuals.csv",
            index=False,
        )
        n_sig_ols = sr_results["significant_bonferroni"].sum()
        print(f"  Factors where country is significant after sector control: "
              f"{n_sig_ols}/{len(all_score_cols)}")
        for _, row in sr_results.iterrows():
            sig_flag = " ***" if row["significant_bonferroni"] else ""
            t_str = (f"{row['country_t_stat']:.2f}"
                     if not pd.isna(row["country_t_stat"]) else "N/A")
            p_str = (f"{row['country_p_value']:.4f}"
                     if not pd.isna(row["country_p_value"]) else "N/A")
            coef_str = (f"{row['country_coef']:+.2f}"
                        if not pd.isna(row["country_coef"]) else "N/A")
            print(f"    {row['factor']:<30s}  coef={coef_str:>7s}  "
                  f"t={t_str:>7s}  p={p_str:>8s}{sig_flag}")
    else:
        print("  [SKIP] Could not run sector-controlled test "
              "(missing sector/country column)")
    print()

    # ==================================================================
    # ANALYSIS 5: Re-normalization robustness check
    # ==================================================================
    print("-" * 70)
    print("TEST 5: Within-geography re-normalization robustness")
    print("-" * 70)
    rn_results = within_geography_renorm(df, df_us, df_in, all_score_cols)
    rn_results.to_csv(
        TABLES / "geographic_robustness_renorm_comparison.csv",
        index=False,
    )

    for geo in ["US", "India"]:
        sub = rn_results[rn_results["geography"] == geo]
        valid = sub["spearman_rho_pooled_vs_renorm"].dropna()
        if len(valid) > 0:
            print(f"  {geo:6s}: mean ρ(pooled vs renorm) = {valid.mean():.3f}, "
                  f"min = {valid.min():.3f}")
            mean_shift = sub["mean_rank_shift"].dropna().mean()
            max_shift = sub["max_rank_shift"].dropna().max()
            print(f"  {' ':6s}  avg rank shift = {mean_shift:.1f}, "
                  f"worst rank shift = {max_shift:.0f}")
        else:
            print(f"  {geo:6s}: insufficient data")
    print()

    # ==================================================================
    # Summary table
    # ==================================================================
    print("=" * 70)
    print("GEOGRAPHIC ROBUSTNESS SUMMARY")
    print("=" * 70)

    # Aggregate summary
    valid_es = es_results["abs_cohens_d"].dropna()
    valid_rho_us = rc_results.loc[
        rc_results["geography"] == "US", "mean_pctile_shift"
    ].dropna()
    valid_rho_in = rc_results.loc[
        rc_results["geography"] == "India", "mean_pctile_shift"
    ].dropna()
    valid_renorm = rn_results["spearman_rho_pooled_vs_renorm"].dropna()

    summary_rows = [{
        "metric": "N companies (US)",
        "value": n_us,
    }, {
        "metric": "N companies (India)",
        "value": n_in,
    }, {
        "metric": "N factors tested",
        "value": len(all_score_cols),
    }, {
        "metric": "Mann-Whitney significant (Bonferroni)",
        "value": int(n_sig),
    }, {
        "metric": "Mean |Cohen's d|",
        "value": round(valid_es.mean(), 4) if len(valid_es) else np.nan,
    }, {
        "metric": "Max |Cohen's d|",
        "value": round(valid_es.max(), 4) if len(valid_es) else np.nan,
    }, {
        "metric": "Factors with medium/large effect (|d|>=0.5)",
        "value": int(n_medium_or_large),
    }, {
        "metric": "Mean percentile shift from pooling (US)",
        "value": round(valid_rho_us.mean(), 4) if len(valid_rho_us) else np.nan,
    }, {
        "metric": "Mean percentile shift from pooling (India)",
        "value": round(valid_rho_in.mean(), 4) if len(valid_rho_in) else np.nan,
    }, {
        "metric": "Country significant after sector control (Bonferroni)",
        "value": int(n_sig_ols) if len(sr_results) else np.nan,
    }, {
        "metric": "Mean renorm rank corr (pooled vs within-geo)",
        "value": round(valid_renorm.mean(), 4) if len(valid_renorm) else np.nan,
    }]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "geographic_robustness_summary.csv", index=False)

    for _, row in summary_df.iterrows():
        val = row["value"]
        val_str = f"{val}" if isinstance(val, int) else (
            f"{val:.4f}" if not pd.isna(val) else "N/A"
        )
        print(f"  {row['metric']:<50s} {val_str}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)

    # Determine verdict based on multiple criteria
    bias_flags = 0
    total_checks = 3

    # Check 1: proportion of significant Mann-Whitney tests
    sig_ratio = n_sig / max(len(all_score_cols), 1)
    if sig_ratio > 0.5:
        bias_flags += 1

    # Check 2: average absolute Cohen's d
    mean_abs_d = valid_es.mean() if len(valid_es) else 0
    if mean_abs_d >= 0.5:
        bias_flags += 1

    # Check 3: country significant after sector control
    if len(sr_results) > 0:
        ols_sig_ratio = n_sig_ols / max(len(all_score_cols), 1)
        if ols_sig_ratio > 0.3:
            bias_flags += 1

    if bias_flags == 0:
        verdict = (
            "NO SYSTEMATIC GEOGRAPHIC BIAS DETECTED. "
            "Pooled normalization does not systematically advantage either "
            "geography. Cross-country differences are small and largely "
            "explained by sector composition."
        )
    elif bias_flags == 1:
        verdict = (
            "MILD GEOGRAPHIC EFFECTS DETECTED. "
            "Some factors show statistically significant US-India differences, "
            "but effect sizes are small or explained by sector composition. "
            "Pooled normalization is acceptable with acknowledgement."
        )
    elif bias_flags == 2:
        verdict = (
            "MODERATE GEOGRAPHIC BIAS DETECTED. "
            "Multiple factors show meaningful US-India differences that persist "
            "after sector control. Consider within-geography normalization or "
            "country-adjusted scoring for affected factors."
        )
    else:
        verdict = (
            "SUBSTANTIAL GEOGRAPHIC BIAS DETECTED. "
            "Pooled normalization systematically advantages one geography "
            "across most factors. Within-geography normalization is strongly "
            "recommended."
        )

    print(f"VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # File output confirmation
    # ------------------------------------------------------------------
    print()
    output_files = [
        "geographic_robustness_mann_whitney.csv",
        "geographic_robustness_effect_sizes.csv",
        "geographic_robustness_rank_correlation.csv",
        "geographic_robustness_sector_residuals.csv",
        "geographic_robustness_renorm_comparison.csv",
        "geographic_robustness_summary.csv",
    ]
    for fname in output_files:
        fpath = TABLES / fname
        if fpath.exists():
            print(f"[OK] Saved  {fpath}")

    print("\n[DONE] Geographic robustness analysis complete.")


if __name__ == "__main__":
    main()
