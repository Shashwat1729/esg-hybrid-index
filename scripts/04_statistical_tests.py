"""
Step 04: Comprehensive Statistical Tests
==========================================
Performs every statistical test relevant for a research paper:
  1.  Descriptive statistics (mean, std, skew, kurtosis)
  2.  Normality tests (Shapiro-Wilk, Jarque-Bera, Lilliefors K-S)
  3.  Correlation analysis (Pearson, Spearman, Kendall)
  4.  ESG-Financial relationship (OLS regression with controls)
  5.  Sector differences (ANOVA + Welch's ANOVA + Kruskal-Wallis + effect sizes)
  6.  Country differences (t-tests, Mann-Whitney, Cohen's d)
  7.  ESG pillar inter-correlations
  8.  Quintile analysis (score quintiles vs. financial performance)
  9.  Factor contribution analysis
  10. Multicollinearity check (VIF)
  11. Rank correlation between profiles
  12. Top/Bottom analysis (20 companies)
  13. Multiple regression with controls
  14. Heteroscedasticity test (Breusch-Pagan via statsmodels)
  15. Decile analysis
  16. Subgroup analysis (size, sector, country)
  17. Non-parametric tests (Friedman, Wilcoxon)
  18. Binary variable analysis (Chi-square, Point-Biserial)
  19. Ordinal variable analysis (Spearman, Kruskal-Wallis)
  20. Non-parametric robustness tests
  21. Factor score pairwise correlations (H2 overlap)
  22. Factor correlation matrix & VIF diagnostics

Input:  data/processed/indexed_data.csv
Output: reports/tables/*.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro, jarque_bera, kstest,
    pearsonr, spearmanr, kendalltau,
    mannwhitneyu, kruskal, wilcoxon,
    f_oneway, ttest_ind, friedmanchisquare,
    alexandergovern,
)
from scipy.stats import norm as _norm, nct as _nct
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import lilliefors as lilliefors_test
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data
from src.constants import BINARY_VARS, ORDINAL_VARS

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

CORE_SCORES = ["ESG_composite", "E_score", "S_score", "G_score",
               "financial_score", "market_score", "operational_score",
               "pref_balanced", "pref_esg_first", "pref_financial_first"]

EXTENDED_SCORES = CORE_SCORES + [
    "risk_adjusted_score", "value_score", "growth_score", "stability_score",
]


# ---------------------------------------------------------------------------
# Helper: Fisher z-transform utilities for correlation CIs and power
# ---------------------------------------------------------------------------
def _fisher_z(r):
    """Arctanh transform (Fisher z) of a correlation coefficient."""
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def _fisher_z_inv(z):
    """Inverse Fisher z (tanh)."""
    return np.tanh(z)


def _spearman_ci(r, n, alpha=0.05):
    """95% confidence interval for Spearman r via Fisher z-transform.

    The SE of z = arctanh(r) is approximately 1/sqrt(n-3).
    """
    if n <= 3:
        return (np.nan, np.nan)
    z = _fisher_z(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = _norm.ppf(1 - alpha / 2)
    lo = _fisher_z_inv(z - z_crit * se)
    hi = _fisher_z_inv(z + z_crit * se)
    return (lo, hi)


def _correlation_power(r, n, alpha=0.05):
    """Statistical power to detect correlation *r* with sample size *n*.

    Uses the Fisher z-transform approach: under H1 the test statistic
    z_obs = arctanh(r) * sqrt(n-3) is ~ N(arctanh(rho)*sqrt(n-3), 1).
    """
    if n <= 3:
        return np.nan
    z_crit = _norm.ppf(1 - alpha / 2)
    noncentrality = abs(_fisher_z(r)) * np.sqrt(n - 3)
    power = 1 - _norm.cdf(z_crit - noncentrality) + _norm.cdf(-z_crit - noncentrality)
    return power


def _min_detectable_r(n, alpha=0.05, power_target=0.80):
    """Minimum |r| detectable at given alpha and power with sample size n."""
    if n <= 3:
        return np.nan
    z_crit = _norm.ppf(1 - alpha / 2)
    z_beta = _norm.ppf(power_target)
    ncp_needed = z_crit + z_beta
    z_r = ncp_needed / np.sqrt(n - 3)
    return _fisher_z_inv(z_r)


def _ttest_power(d, n1, n2, alpha=0.05):
    """Power for an independent two-sample t-test (unequal n) via noncentral t.

    Parameters
    ----------
    d  : Cohen's d effect size
    n1, n2 : group sample sizes
    alpha : significance level (two-sided)
    """
    df = n1 + n2 - 2
    ncp = abs(d) * np.sqrt(n1 * n2 / (n1 + n2))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - _nct.cdf(t_crit, df, ncp) + _nct.cdf(-t_crit, df, ncp)
    return power


def _min_detectable_d(n1, n2, alpha=0.05, power_target=0.80):
    """Minimum Cohen's d detectable for two-sample t-test at given power."""
    from scipy.optimize import brentq
    try:
        d_min = brentq(lambda d: _ttest_power(d, n1, n2, alpha) - power_target,
                       0.001, 5.0, xtol=1e-4)
        return d_min
    except ValueError:
        return np.nan


def _n_for_correlation(r, alpha=0.05, power_target=0.80):
    """Sample size needed to detect correlation r at given alpha/power."""
    if abs(r) < 1e-6:
        return np.inf
    z_crit = _norm.ppf(1 - alpha / 2)
    z_beta = _norm.ppf(power_target)
    ncp_needed = z_crit + z_beta
    z_r = _fisher_z(r)
    n_needed = (ncp_needed / abs(z_r)) ** 2 + 3
    return int(np.ceil(n_needed))


def load_data():
    df = load_indexed_data(PROJECT_ROOT)
    print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns")
    return df


def _cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size for two independent groups.

    Cliff's delta is a non-parametric effect size measure that quantifies
    the degree to which values in one group tend to be larger than values
    in the other group.  It is the appropriate companion to the
    Mann-Whitney U test.

    Returns
    -------
    delta : float
        Value in [-1, 1].  +1 means every x > every y.
    size : str
        Qualitative label using Romano et al. (2006) thresholds:
        |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    delta = (more - less) / (n_x * n_y)
    # Interpret: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large
    abs_d = abs(delta)
    if abs_d < 0.147:
        size = "negligible"
    elif abs_d < 0.33:
        size = "small"
    elif abs_d < 0.474:
        size = "medium"
    else:
        size = "large"
    return delta, size


# 1. Descriptive Statistics
def test_descriptive(df):
    print("\n--- Test 1: Descriptive Statistics ---")
    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    desc = df[avail].describe().T
    desc["skewness"] = df[avail].skew()
    desc["kurtosis"] = df[avail].kurtosis()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["cv"] = desc["std"] / desc["mean"].abs().clip(1e-10)
    desc.to_csv(TABLES / "descriptive_statistics.csv", encoding="utf-8")
    print(f"  [OK] Saved descriptive_statistics.csv ({len(avail)} variables)")
    return desc


# 2. Normality Tests
def test_normality(df):
    print("\n--- Test 2: Normality Tests (Shapiro, Jarque-Bera, Lilliefors) ---")
    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    for col in avail:
        data = df[col].dropna()
        if len(data) < 8:
            continue
        sw_stat, sw_p = shapiro(data[:5000])
        jb_stat, jb_p = jarque_bera(data)
        # Lilliefors correction for composite null hypothesis (estimated parameters)
        ks_stat, ks_p = lilliefors_test(data, dist='norm')
        rows.append({
            "variable": col,
            "n": len(data),
            "shapiro_stat": sw_stat, "shapiro_p": sw_p,
            "jarque_bera_stat": jb_stat, "jarque_bera_p": jb_p,
            "ks_stat": ks_stat, "ks_p": ks_p,
            "normal_shapiro": sw_p > 0.05,
            "normal_jb": jb_p > 0.05,
            "normal_ks": ks_p > 0.05,
        })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "normality_tests.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved normality_tests.csv")
    return result


# 3. Correlation Analysis
def test_correlations(df):
    print("\n--- Test 3: Correlation Analysis ---")
    financial_extras = ["roa", "roe", "market_cap", "total_revenue",
                        "debt_to_equity", "net_margin", "current_ratio",
                        "revenue_growth", "dividend_yield"]
    avail = [c for c in EXTENDED_SCORES + financial_extras
             if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    pearson_corr = df[avail].corr(method="pearson")
    pearson_corr.to_csv(TABLES / "correlation_pearson.csv", encoding="utf-8")

    spearman_corr = df[avail].corr(method="spearman")
    spearman_corr.to_csv(TABLES / "correlation_spearman.csv", encoding="utf-8")

    # Pairwise significance for core scores
    core_avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    for i, c1 in enumerate(core_avail):
        for c2 in core_avail[i + 1:]:
            d = df[[c1, c2]].dropna()
            if len(d) < 5:
                continue
            pr, pp = pearsonr(d[c1], d[c2])
            sr, sp = spearmanr(d[c1], d[c2])
            kr, kp = kendalltau(d[c1], d[c2])
            n_pair = len(d)
            sr_ci_lo, sr_ci_hi = _spearman_ci(sr, n_pair)
            pr_ci_lo, pr_ci_hi = _spearman_ci(pr, n_pair)  # Fisher z CI works for Pearson too
            rows.append({
                "var1": c1, "var2": c2, "n": n_pair,
                "pearson_r": pr, "pearson_p": pp,
                "pearson_ci_lo": pr_ci_lo, "pearson_ci_hi": pr_ci_hi,
                "spearman_r": sr, "spearman_p": sp,
                "spearman_ci_lo": sr_ci_lo, "spearman_ci_hi": sr_ci_hi,
                "kendall_tau": kr, "kendall_p": kp,
                "sig_pearson": pp < 0.05,
                "sig_spearman": sp < 0.05,
            })
    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(TABLES / "correlation_significance.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved correlation tables ({len(avail)} variables, {len(rows)} pairs)")
    return pearson_corr, spearman_corr


# 4. ESG-Financial Regression (with sector controls)
def test_esg_financial_regression(df):
    """Test association between ESG composite and financial metrics.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  SYNTHETIC DATA LIMITATION                                         ║
    ║                                                                    ║
    ║  The ESG scores used here are PREDOMINANTLY SYNTHETIC with         ║
    ║  sector-specific profiles. corr_weight was removed — ESG-financial ║
    ║  correlation is now emergent from sector patterns only.            ║
    ║  Real data is limited to Yahoo governance risk scores              ║
    ║  (5 indicators) and financial proxies (5 indicators).              ║
    ║                                                                    ║
    ║  ESG-financial regressions should be interpreted with caution:     ║
    ║  any association may reflect sector-level confounding rather       ║
    ║  than a genuine ESG-performance link.                              ║
    ║                                                                    ║
    ║  Cross-sectional validation uses ex-market variants to mitigate    ║
    ║  circularity. With real ESG provider data, these regressions       ║
    ║  would test genuine empirical associations.                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print("\n--- Test 4: ESG-Financial Regression ---")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │ CAVEAT: ESG data is predominantly synthetic with sector-       │")
    print("  │ specific profiles (corr_weight removed — no built-in blend).   │")
    print("  │ ESG-financial correlation is emergent from sector patterns     │")
    print("  │ only. Interpret regressions with caution.                      │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    rows = []
    if "ESG_composite" not in df.columns:
        return pd.DataFrame()

    dep_vars = ["roa", "roe", "net_margin", "financial_score", "revenue_growth",
                "operating_margins", "current_ratio"]
    for dv in dep_vars:
        if dv not in df.columns:
            continue
        mask = df[["ESG_composite", dv]].dropna().index
        if len(mask) < 10:
            continue
        x = df.loc[mask, "ESG_composite"].values
        y = df.loc[mask, dv].values
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        rows.append({
            "dependent_var": dv, "independent_var": "ESG_composite",
            "slope": slope, "intercept": intercept,
            "r_squared": r_val ** 2, "p_value": p_val, "std_error": std_err,
            "n": len(mask),
        })

    # Also test individual pillars
    for pillar in ["E_score", "S_score", "G_score"]:
        if pillar not in df.columns:
            continue
        for dv in ["roa", "financial_score"]:
            if dv not in df.columns:
                continue
            mask = df[[pillar, dv]].dropna().index
            if len(mask) < 10:
                continue
            x = df.loc[mask, pillar].values
            y = df.loc[mask, dv].values
            slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
            rows.append({
                "dependent_var": dv, "independent_var": pillar,
                "slope": slope, "intercept": intercept,
                "r_squared": r_val ** 2, "p_value": p_val, "std_error": std_err,
                "n": len(mask),
            })

    result = pd.DataFrame(rows)

    # Prepend a metadata row documenting the synthetic data limitation
    caveat_row = pd.DataFrame([{
        "dependent_var": "# SYNTHETIC DATA CAVEAT",
        "independent_var": "ESG data is predominantly synthetic with sector-specific profiles. "
                          "corr_weight was removed — ESG-financial correlation is emergent "
                          "from sector patterns only, not a built-in parameter.",
        "slope": None, "intercept": None,
        "r_squared": None, "p_value": None, "std_error": None, "n": None,
    }])
    result_with_caveat = pd.concat([caveat_row, result], ignore_index=True)
    result_with_caveat.to_csv(TABLES / "esg_financial_regression.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved esg_financial_regression.csv ({len(result)} regressions + caveat row)")
    return result


# 5. Sector Differences (ANOVA + effect sizes)
def test_sector_differences(df):
    print("\n--- Test 5: Sector Differences (ANOVA + Effect Sizes) ---")
    if "sector" not in df.columns:
        return pd.DataFrame()

    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    for col in avail:
        groups = [g[col].dropna().values for _, g in df.groupby("sector") if len(g[col].dropna()) > 1]
        if len(groups) < 2:
            continue
        f_stat, f_p = f_oneway(*groups)
        k_stat, k_p = kruskal(*groups)

        # Welch's ANOVA (robust to unequal variances and sample sizes)
        try:
            welch_result = alexandergovern(*groups)
            welch_stat, welch_p = welch_result.statistic, welch_result.pvalue
        except Exception:
            welch_stat, welch_p = np.nan, np.nan

        # Eta-squared effect size for ANOVA
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
        eta_sq = ss_between / (ss_total + 1e-10)

        rows.append({
            "variable": col,
            "n_groups": len(groups),
            "anova_F": f_stat, "anova_p": f_p,
            "welch_anova_stat": welch_stat, "welch_anova_p": welch_p,
            "kruskal_H": k_stat, "kruskal_p": k_p,
            "eta_squared": eta_sq,
            "effect_size": "Large" if eta_sq > 0.14 else ("Medium" if eta_sq > 0.06 else "Small"),
            "significant_anova": f_p < 0.05,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "sector_anova.csv", index=False, encoding="utf-8")

    # Sector means
    sector_stats = df.groupby("sector")[avail].agg(["mean", "std", "count"])
    sector_stats.to_csv(TABLES / "sector_means.csv", encoding="utf-8")
    print(f"  [OK] Saved sector_anova.csv, sector_means.csv")
    return result


# 6. Country Differences (with Cohen's d)
def test_country_differences(df):
    print("\n--- Test 6: Country Differences (with Effect Sizes) ---")
    if "country" not in df.columns:
        return pd.DataFrame()

    countries = df["country"].dropna().unique()
    if len(countries) < 2:
        return pd.DataFrame()

    avail = [c for c in EXTENDED_SCORES if c in df.columns]
    rows = []
    c1, c2 = countries[0], countries[1]
    g1 = df[df["country"] == c1]
    g2 = df[df["country"] == c2]

    for col in avail:
        d1 = g1[col].dropna()
        d2 = g2[col].dropna()
        if len(d1) < 3 or len(d2) < 3:
            continue
        t_stat, t_p = ttest_ind(d1, d2)
        u_stat, u_p = mannwhitneyu(d1, d2, alternative="two-sided")
        d_effect = _cohens_d(d1, d2)
        cliff_d, cliff_size = cliffs_delta(d1.values, d2.values)

        # Confidence interval for mean difference (from t-distribution)
        n1, n2 = len(d1), len(d2)
        mean_diff = d1.mean() - d2.mean()
        pooled_se = np.sqrt(d1.var() / n1 + d2.var() / n2)
        df_welch = (d1.var() / n1 + d2.var() / n2) ** 2 / (
            (d1.var() / n1) ** 2 / (n1 - 1) + (d2.var() / n2) ** 2 / (n2 - 1)
        )
        t_crit = stats.t.ppf(0.975, df_welch)
        ci_lo = mean_diff - t_crit * pooled_se
        ci_hi = mean_diff + t_crit * pooled_se

        rows.append({
            "variable": col,
            f"mean_{c1}": d1.mean(), f"std_{c1}": d1.std(), f"n_{c1}": n1,
            f"mean_{c2}": d2.mean(), f"std_{c2}": d2.std(), f"n_{c2}": n2,
            "mean_diff": mean_diff,
            "mean_diff_ci_lo": ci_lo, "mean_diff_ci_hi": ci_hi,
            "ttest_stat": t_stat, "ttest_p": t_p,
            "mannwhitney_U": u_stat, "mannwhitney_p": u_p,
            "cohens_d": d_effect,
            "cohens_d_power": _ttest_power(d_effect, n1, n2),
            "effect_size": "Large" if abs(d_effect) > 0.8 else ("Medium" if abs(d_effect) > 0.5 else "Small"),
            "cliffs_delta": cliff_d,
            "cliffs_delta_size": cliff_size,
            "significant_t": t_p < 0.05,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "country_differences.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved country_differences.csv ({len(result)} variables)")
    return result


# 7. ESG Pillar Inter-Correlations
def test_pillar_correlations(df):
    print("\n--- Test 7: ESG Pillar Inter-Correlations ---")
    pillars = ["E_score", "S_score", "G_score"]
    avail = [c for c in pillars if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    rows = []
    for i, p1 in enumerate(avail):
        for p2 in avail[i + 1:]:
            d = df[[p1, p2]].dropna()
            if len(d) < 5:
                continue
            pr, pp = pearsonr(d[p1], d[p2])
            sr, sp = spearmanr(d[p1], d[p2])
            rows.append({
                "pillar1": p1, "pillar2": p2,
                "pearson_r": pr, "pearson_p": pp,
                "spearman_r": sr, "spearman_p": sp,
            })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "pillar_correlations.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved pillar_correlations.csv")
    return result


# 8. Quintile Analysis
def test_quintile_analysis(df):
    print("\n--- Test 8: Quintile Analysis ---")
    if "pref_balanced" not in df.columns:
        return pd.DataFrame()

    df_q = df.copy()
    df_q["quintile"] = pd.qcut(df_q["pref_balanced"], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"])

    metrics = ["roa", "roe", "net_margin", "ESG_composite", "financial_score",
               "market_score", "operational_score", "revenue_growth",
               "current_ratio", "dividend_yield"]
    avail = [c for c in metrics if c in df_q.columns]

    quintile_stats = df_q.groupby("quintile")[avail].agg(["mean", "std", "count"])
    quintile_stats.to_csv(TABLES / "quintile_analysis.csv", encoding="utf-8")

    # T-test: Q5 vs Q1
    q5 = df_q[df_q["quintile"] == "Q5(High)"]
    q1 = df_q[df_q["quintile"] == "Q1(Low)"]
    rows = []
    for col in avail:
        d5 = q5[col].dropna()
        d1 = q1[col].dropna()
        if len(d5) > 1 and len(d1) > 1:
            t, p = ttest_ind(d5, d1)
            d_eff = _cohens_d(d5, d1)
            rows.append({
                "variable": col,
                "Q5_mean": d5.mean(), "Q1_mean": d1.mean(),
                "difference": d5.mean() - d1.mean(),
                "ttest_stat": t, "ttest_p": p,
                "cohens_d": d_eff,
                "significant": p < 0.05,
            })
    q_test = pd.DataFrame(rows)
    q_test.to_csv(TABLES / "quintile_q5_vs_q1.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved quintile_analysis.csv, quintile_q5_vs_q1.csv")
    return quintile_stats


# 9. Factor Contribution Analysis
def test_factor_contributions(df):
    print("\n--- Test 9: Factor Contribution Analysis ---")
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score",
               "risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    avail = [c for c in factors if c in df.columns]
    if "pref_balanced" not in df.columns or len(avail) < 2:
        return pd.DataFrame()

    rows = []
    for factor in avail:
        mask = df[[factor, "pref_balanced"]].dropna().index
        x = df.loc[mask, factor].values
        y = df.loc[mask, "pref_balanced"].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        rows.append({
            "factor": factor,
            "correlation_with_pref": r,
            "r_squared": r**2,
            "slope": slope,
            "p_value": p,
        })

    result = pd.DataFrame(rows).sort_values("r_squared", ascending=False)
    result.to_csv(TABLES / "factor_contributions.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved factor_contributions.csv ({len(result)} factors)")
    return result


# 10. Multicollinearity (VIF)
def test_multicollinearity(df):
    print("\n--- Test 10: Multicollinearity (VIF) ---")
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score",
               "risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    avail = [c for c in factors if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    from numpy.linalg import inv
    X = df[avail].dropna()
    if len(X) < len(avail) + 2:
        return pd.DataFrame()

    corr = X.corr().values
    try:
        inv_corr = inv(corr)
        vif = pd.DataFrame({
            "factor": avail,
            "VIF": [inv_corr[i, i] for i in range(len(avail))],
        })
        vif["multicollinearity"] = vif["VIF"].apply(
            lambda x: "High" if x > 10 else ("Moderate" if x > 5 else "Low")
        )
        vif.to_csv(TABLES / "vif_multicollinearity.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved vif_multicollinearity.csv")
        return vif
    except Exception:
        print("  [SKIP] Could not compute VIF (singular matrix)")
        return pd.DataFrame()


# 11. Profile Rank Correlation
def test_profile_rank_correlation(df):
    print("\n--- Test 11: Profile Rank Correlation ---")
    profiles = ["pref_esg_first", "pref_balanced", "pref_financial_first"]
    avail = [c for c in profiles if c in df.columns]
    if len(avail) < 2:
        return pd.DataFrame()

    rows = []
    for i, p1 in enumerate(avail):
        for p2 in avail[i + 1:]:
            d = df[[p1, p2]].dropna()
            r1 = d[p1].rank()
            r2 = d[p2].rank()
            sr, sp = spearmanr(r1, r2)
            kt, kp = kendalltau(r1, r2)
            rows.append({
                "profile1": p1, "profile2": p2,
                "spearman_rank_r": sr, "spearman_p": sp,
                "kendall_tau": kt, "kendall_p": kp,
                "top10_overlap": len(set(d.nlargest(10, p1).index) & set(d.nlargest(10, p2).index)),
                "top20_overlap": len(set(d.nlargest(20, p1).index) & set(d.nlargest(20, p2).index)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "profile_rank_correlation.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved profile_rank_correlation.csv")
    return result


# 12. Top/Bottom Analysis
def test_top_bottom(df):
    print("\n--- Test 12: Top/Bottom Company Analysis ---")
    for col, label in [("pref_balanced", "balanced"), ("ESG_composite", "esg"),
                       ("financial_score", "financial"), ("growth_score", "growth"),
                       ("value_score", "value")]:
        if col not in df.columns:
            continue
        display_cols = ["ticker", "company_name", "sector", "country",
                        "ESG_composite", "financial_score", "market_score",
                        "operational_score", "risk_adjusted_score",
                        "value_score", "growth_score", col]
        avail_display = [c for c in display_cols if c in df.columns]
        n_top = min(20, len(df))
        top = df.nlargest(n_top, col)[avail_display]
        top.to_csv(TABLES / f"top20_{label}.csv", index=False, encoding="utf-8")
        bottom = df.nsmallest(n_top, col)[avail_display]
        bottom.to_csv(TABLES / f"bottom20_{label}.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved top20_*.csv, bottom20_*.csv")


# 13. Multiple Regression with Controls
def test_multiple_regression(df):
    print("\n--- Test 13: Multiple Regression (ESG -> Financial with Controls) ---")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │ CAVEAT: ESG data is predominantly synthetic with sector-       │")
    print("  │ specific profiles (corr_weight removed — no built-in blend).   │")
    print("  │ ESG-financial coefficients reflect sector patterns, not        │")
    print("  │ empirical discovery. See 01_download_data.py for details.      │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [SKIP] statsmodels not installed")
        return

    dep_vars = ["roa", "roe", "financial_score"]
    indep = ["ESG_composite"]
    controls = ["market_cap", "debt_to_equity", "beta"]

    rows = []
    for dv in dep_vars:
        if dv not in df.columns:
            continue
        all_vars = [dv] + indep + [c for c in controls if c in df.columns]
        subset = df[all_vars].dropna()
        if len(subset) < 20:
            continue

        X = subset[indep + [c for c in controls if c in subset.columns]]
        y = subset[dv]

        # Standardize
        X_std = (X - X.mean()) / (X.std() + 1e-10)
        X_std = sm.add_constant(X_std)

        try:
            model = sm.OLS(y, X_std).fit()
            for var in indep + [c for c in controls if c in X.columns]:
                if var in model.params.index:
                    rows.append({
                        "dependent": dv,
                        "independent": var,
                        "coefficient": model.params[var],
                        "std_error": model.bse[var],
                        "t_stat": model.tvalues[var],
                        "p_value": model.pvalues[var],
                        "r_squared": model.rsquared,
                        "adj_r_squared": model.rsquared_adj,
                        "n_obs": int(model.nobs),
                        "f_stat": model.fvalue,
                        "f_pvalue": model.f_pvalue,
                    })
        except Exception as e:
            import logging
            logging.warning(f"Multiple regression failed for {dv}: {e}")

    if rows:
        result = pd.DataFrame(rows)
        # Prepend synthetic data caveat row
        caveat_row = pd.DataFrame([{
            "dependent": "# SYNTHETIC DATA CAVEAT",
            "independent": "ESG data is predominantly synthetic with sector-specific profiles. "
                          "corr_weight was removed — ESG-financial coefficients reflect "
                          "emergent sector patterns, not a built-in parameter.",
            "coefficient": None, "std_error": None, "t_stat": None,
            "p_value": None, "r_squared": None, "adj_r_squared": None,
            "n_obs": None, "f_stat": None, "f_pvalue": None,
        }])
        result_with_caveat = pd.concat([caveat_row, result], ignore_index=True)
        result_with_caveat.to_csv(TABLES / "multiple_regression.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved multiple_regression.csv ({len(result)} coefficients + caveat row)")


# 14. Heteroscedasticity Test
def test_heteroscedasticity(df):
    print("\n--- Test 14: Heteroscedasticity (Breusch-Pagan) ---")
    print("  NOTE: ESG->financial residual patterns reflect synthetic data structure.")
    try:
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError:
        print("  [SKIP] statsmodels not installed")
        return

    if "ESG_composite" not in df.columns:
        return

    rows = []
    for dv in ["roa", "roe", "financial_score"]:
        if dv not in df.columns:
            continue
        subset = df[["ESG_composite", dv]].dropna()
        if len(subset) < 20:
            continue
        X = sm.add_constant(subset["ESG_composite"])
        y = subset[dv]
        try:
            model = sm.OLS(y, X).fit()
            bp_stat, bp_p, f_stat, f_p = het_breuschpagan(model.resid, X)
            rows.append({
                "dependent": dv,
                "bp_stat": bp_stat, "bp_p": bp_p,
                "f_stat": f_stat, "f_p": f_p,
                "heteroscedastic": bp_p < 0.05,
            })
        except Exception:
            pass

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "heteroscedasticity.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved heteroscedasticity.csv")


# 15. Decile Analysis
def test_decile_analysis(df):
    print("\n--- Test 15: Decile Analysis ---")
    if "pref_balanced" not in df.columns:
        return

    df_d = df.copy()
    df_d["decile"] = pd.qcut(df_d["pref_balanced"], 10, labels=[f"D{i}" for i in range(1, 11)])

    metrics = ["roa", "roe", "ESG_composite", "financial_score", "market_score",
               "revenue_growth", "price_momentum_3m", "sharpe_ratio_1y"]
    avail = [c for c in metrics if c in df_d.columns]

    if avail:
        decile_stats = df_d.groupby("decile")[avail].agg(["mean", "std", "count"])
        decile_stats.to_csv(TABLES / "decile_analysis.csv", encoding="utf-8")
        print(f"  [OK] Saved decile_analysis.csv")


# 16. Subgroup Analysis (by size)
def test_subgroup_analysis(df):
    print("\n--- Test 16: Subgroup Analysis (Size Terciles) ---")
    if "market_cap" not in df.columns:
        return

    df_s = df.copy()
    try:
        df_s["size_group"] = pd.qcut(df_s["market_cap"], 3, labels=["Small", "Medium", "Large"])
    except ValueError:
        return

    score_cols = [c for c in CORE_SCORES if c in df_s.columns]
    if not score_cols:
        return

    # ANOVA for size groups
    rows = []
    for col in score_cols:
        groups = [g[col].dropna().values for _, g in df_s.groupby("size_group") if len(g[col].dropna()) > 1]
        if len(groups) < 2:
            continue
        f_stat, f_p = f_oneway(*groups)
        rows.append({
            "variable": col,
            "anova_F": f_stat, "anova_p": f_p,
            "significant": f_p < 0.05,
        })

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "size_group_anova.csv", index=False, encoding="utf-8")

    # Size group means
    size_means = df_s.groupby("size_group")[score_cols].mean()
    size_means.to_csv(TABLES / "size_group_means.csv", encoding="utf-8")
    print(f"  [OK] Saved size_group_anova.csv, size_group_means.csv")


# 17. Comprehensive Sector-Score Interaction
def test_sector_score_interaction(df):
    print("\n--- Test 17: Sector-Score Interaction ---")
    print("  NOTE: ESG-financial correlations within sectors reflect synthetic")
    print("        ESG data with sector-specific profiles (corr_weight removed —")
    print("        no built-in financial quality blending). Interpret with caution.")
    if "sector" not in df.columns:
        return

    # Within each sector, correlate ESG with financial
    rows = []
    for sector in df["sector"].dropna().unique():
        sub = df[df["sector"] == sector]
        if len(sub) < 5:
            continue
        for iv, dv in [("ESG_composite", "financial_score"), ("ESG_composite", "roa"),
                       ("financial_score", "market_score")]:
            if iv not in sub.columns or dv not in sub.columns:
                continue
            d = sub[[iv, dv]].dropna()
            if len(d) < 5:
                continue
            r, p = pearsonr(d[iv], d[dv])
            rows.append({
                "sector": sector,
                "x_var": iv, "y_var": dv,
                "n": len(d),
                "pearson_r": r, "pearson_p": p,
                "significant": p < 0.05,
            })

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(TABLES / "sector_score_interaction.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved sector_score_interaction.csv ({len(result)} tests)")


# 18. Binary Variable Analysis (Chi-square, Point-Biserial)
def test_binary_variables(df):
    """Analyze binary ESG policy variables: proportions, chi-square, point-biserial correlations.

    Binary variables require different statistical tests than continuous variables:
    - Chi-square for association with sector/country (categorical x categorical)
    - Point-biserial correlation for association with continuous scores
    - Phi coefficient for association between two binary variables

    References:
    - Agresti (2002) "Categorical Data Analysis"
    - Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
    """
    print("\n--- Test 18: Binary Variable Analysis ---")
    binary_cols = [c for c in BINARY_VARS if c in df.columns]
    if not binary_cols:
        return pd.DataFrame()

    rows = []

    # Proportions and basic stats
    for col in binary_cols:
        vals = df[col].dropna()
        if len(vals) < 5:
            continue
        row = {
            "variable": col,
            "n": len(vals),
            "proportion_yes": vals.mean(),
            "proportion_no": 1 - vals.mean(),
        }

        # Point-biserial correlation with key scores
        for score_col in ["ESG_composite", "financial_score", "pref_balanced"]:
            if score_col in df.columns:
                d = df[[col, score_col]].dropna()
                if len(d) > 5:
                    from scipy.stats import pointbiserialr
                    r, p = pointbiserialr(d[col], d[score_col])
                    row[f"pb_r_{score_col}"] = r
                    row[f"pb_p_{score_col}"] = p
                    row[f"pb_sig_{score_col}"] = p < 0.05

        # Sector-wise proportions (Chi-square test)
        if "sector" in df.columns:
            contingency = pd.crosstab(df["sector"], df[col].fillna(0).astype(int))
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                from scipy.stats import chi2_contingency
                chi2, chi_p, dof, _ = chi2_contingency(contingency)
                row["chi2_sector"] = chi2
                row["chi2_p_sector"] = chi_p
                row["chi2_sig_sector"] = chi_p < 0.05

        rows.append(row)

    # Phi coefficient between binary variables (pairwise)
    phi_rows = []
    for i, c1 in enumerate(binary_cols):
        for c2 in binary_cols[i+1:]:
            d = df[[c1, c2]].dropna()
            if len(d) < 5:
                continue
            contingency = pd.crosstab(d[c1].astype(int), d[c2].astype(int))
            if contingency.shape == (2, 2):
                from scipy.stats import chi2_contingency
                chi2, chi_p, _, _ = chi2_contingency(contingency)
                phi = np.sqrt(chi2 / len(d))
                phi_rows.append({
                    "var1": c1, "var2": c2,
                    "phi_coefficient": phi, "chi2": chi2, "p_value": chi_p,
                })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "binary_variable_analysis.csv", index=False, encoding="utf-8")

    if phi_rows:
        phi_df = pd.DataFrame(phi_rows)
        phi_df.to_csv(TABLES / "binary_phi_coefficients.csv", index=False, encoding="utf-8")

    print(f"  [OK] Saved binary_variable_analysis.csv ({len(result)} variables)")
    return result


# 19. Ordinal Variable Analysis (Spearman, Kruskal-Wallis)
def test_ordinal_variables(df):
    """Analyze ordinal variables using non-parametric methods.

    Ordinal variables (like board_size) should not use Pearson correlation
    or standard ANOVA. Instead we use:
    - Spearman rank correlation
    - Kruskal-Wallis H test (non-parametric ANOVA equivalent)
    - Jonckheere-Terpstra trend test (for monotonic relationships)

    Reference: Siegel & Castellan (1988) "Nonparametric Statistics for the Behavioral Sciences"
    """
    print("\n--- Test 19: Ordinal Variable Analysis ---")
    ordinal_cols = [c for c in ORDINAL_VARS if c in df.columns]
    if not ordinal_cols:
        return pd.DataFrame()

    rows = []
    for col in ordinal_cols:
        vals = df[col].dropna()
        if len(vals) < 5:
            continue

        row = {
            "variable": col,
            "n": len(vals),
            "median": vals.median(),
            "mode": vals.mode().iloc[0] if len(vals.mode()) > 0 else None,
            "min": vals.min(),
            "max": vals.max(),
            "n_unique": vals.nunique(),
        }

        # Spearman rank correlation with key scores
        for score_col in ["ESG_composite", "G_score", "financial_score", "pref_balanced"]:
            if score_col in df.columns:
                d = df[[col, score_col]].dropna()
                if len(d) > 5:
                    sr, sp = spearmanr(d[col], d[score_col])
                    row[f"spearman_r_{score_col}"] = sr
                    row[f"spearman_p_{score_col}"] = sp

        # Kruskal-Wallis: does the ordinal variable differ by sector?
        if "sector" in df.columns:
            groups = [g[col].dropna().values for _, g in df.groupby("sector") if len(g[col].dropna()) > 1]
            if len(groups) >= 2:
                h_stat, h_p = kruskal(*groups)
                row["kruskal_H_sector"] = h_stat
                row["kruskal_p_sector"] = h_p

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "ordinal_variable_analysis.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved ordinal_variable_analysis.csv ({len(result)} variables)")
    return result


# 20. Non-Parametric Robustness Tests (Friedman, Wilcoxon)
def test_nonparametric_robustness(df):
    """Run non-parametric tests that don't assume normality.

    Adds Friedman test (non-parametric repeated-measures ANOVA equivalent)
    comparing scores across factors, and pairwise Wilcoxon signed-rank tests.
    """
    print("\n--- Test 20: Non-Parametric Robustness Tests ---")
    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail = [c for c in score_cols if c in df.columns]
    if len(avail) < 3:
        return

    # Friedman test: are scores significantly different across factors?
    data = df[avail].dropna()
    if len(data) < 10:
        return

    try:
        f_stat, f_p = friedmanchisquare(*[data[c].values for c in avail])
        print(f"  Friedman chi-sq = {f_stat:.2f}, p = {f_p:.4f}")
    except Exception:
        f_stat, f_p = 0, 1

    # Pairwise Wilcoxon signed-rank tests
    rows = []
    for i, c1 in enumerate(avail):
        for c2 in avail[i+1:]:
            d = df[[c1, c2]].dropna()
            if len(d) < 10:
                continue
            try:
                w_stat, w_p = wilcoxon(d[c1], d[c2])
            except Exception:
                w_stat, w_p = 0, 1

            # Effect size r = Z / sqrt(N) for Wilcoxon signed-rank test
            n_pairs = len(d)
            # Approximate Z from p-value (two-sided) using inverse normal
            if w_p < 1.0 and w_p > 0.0:
                z_val = abs(_norm.ppf(w_p / 2))
            else:
                z_val = 0.0
            wilcoxon_r = z_val / np.sqrt(n_pairs) if n_pairs > 0 else 0.0
            # Interpret: |r| < 0.1 negligible, < 0.3 small, < 0.5 medium, else large
            abs_r = abs(wilcoxon_r)
            if abs_r < 0.1:
                wilcoxon_r_size = "negligible"
            elif abs_r < 0.3:
                wilcoxon_r_size = "small"
            elif abs_r < 0.5:
                wilcoxon_r_size = "medium"
            else:
                wilcoxon_r_size = "large"

            rows.append({
                "var1": c1, "var2": c2,
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "wilcoxon_r": wilcoxon_r,
                "wilcoxon_r_size": wilcoxon_r_size,
                "significant": w_p < 0.05,
                "mean_diff": (d[c1] - d[c2]).mean(),
            })

    result = pd.DataFrame(rows)
    friedman_row = pd.DataFrame([{"test": "Friedman", "statistic": f_stat, "p_value": f_p,
                                   "n_factors": len(avail), "n_companies": len(data)}])
    friedman_row.to_csv(TABLES / "friedman_test.csv", index=False, encoding="utf-8")
    result.to_csv(TABLES / "wilcoxon_pairwise.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved friedman_test.csv, wilcoxon_pairwise.csv ({len(result)} pairs)")


# 21. Factor Score Pairwise Correlations (H2 overlap diagnostic)
def test_factor_score_correlations(df):
    """Compute and report the full pairwise correlation matrix between all 10
    factor scores.  This directly diagnoses Issue H2 (indicator overlap):
    high inter-factor correlation (|r| > 0.5) signals that shared indicators
    are inflating apparent factor diversity.

    Outputs
    -------
    reports/tables/factor_score_correlations.csv
        Full Pearson correlation matrix between all 10 factor scores.

    Interpretation thresholds (Cohen, 1988; Hair et al., 2019):
      |r| > 0.7  : very high — factors are near-redundant, likely heavy overlap
      |r| > 0.5  : high — substantial shared variance, probable indicator overlap
      |r| > 0.3  : moderate — some shared information but factors are distinct
      |r| <= 0.3 : low — factors capture different constructs
    """
    print("\n--- Test 21: Factor Score Pairwise Correlations (H2 overlap) ---")

    all_factor_scores = [
        "ESG_composite", "financial_score", "market_score", "operational_score",
        "risk_adjusted_score", "value_score", "growth_score", "stability_score",
        "similarity_rank", "sector_position",
    ]
    avail = [c for c in all_factor_scores if c in df.columns]
    if len(avail) < 2:
        print("  [SKIP] Fewer than 2 factor scores available")
        return pd.DataFrame()

    # Pearson correlation matrix
    corr_matrix = df[avail].corr(method="pearson")
    corr_matrix.to_csv(TABLES / "factor_score_correlations.csv", encoding="utf-8")

    # Identify and report high-overlap pairs (|r| > 0.5)
    high_overlap_pairs = []
    for i, c1 in enumerate(avail):
        for c2 in avail[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if abs(r) > 0.5:
                high_overlap_pairs.append((c1, c2, r))

    print(f"  Factor scores analysed: {len(avail)}")
    print(f"  Pairs with |r| > 0.5 (high overlap): {len(high_overlap_pairs)}")
    if high_overlap_pairs:
        print("  HIGH-OVERLAP PAIRS (probable indicator double-counting):")
        for c1, c2, r in sorted(high_overlap_pairs, key=lambda x: -abs(x[2])):
            print(f"    {c1} <-> {c2}: r = {r:.3f}")
    else:
        print("  No factor pairs exceed the |r| > 0.5 threshold.")

    # Also report moderate pairs for completeness
    moderate_pairs = []
    for i, c1 in enumerate(avail):
        for c2 in avail[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if 0.3 < abs(r) <= 0.5:
                moderate_pairs.append((c1, c2, r))
    if moderate_pairs:
        print(f"  Moderate overlap (0.3 < |r| <= 0.5): {len(moderate_pairs)} pairs")
        for c1, c2, r in sorted(moderate_pairs, key=lambda x: -abs(x[2])):
            print(f"    {c1} <-> {c2}: r = {r:.3f}")

    print(f"  [OK] Saved factor_score_correlations.csv ({len(avail)}x{len(avail)})")
    return corr_matrix


# 22. Factor Correlation Matrix & VIF Diagnostics
def compute_factor_diagnostics(df):
    """Compute 10x10 factor correlation matrix and VIF for multicollinearity assessment.

    Outputs
    -------
    reports/tables/factor_score_correlations.csv   (overwritten with Pearson matrix)
    reports/tables/factor_score_spearman.csv        (Spearman matrix)
    reports/tables/factor_vif.csv                   (VIF per factor with concern flags)

    Interpretation (Hair et al., 2019; O'Brien, 2007):
      VIF > 10 : HIGH multicollinearity — factor is near-linearly dependent on others
      VIF > 5  : MODERATE — worth investigating; may inflate standard errors
      VIF <= 5 : LOW — acceptable for regression and composite scoring

    Condition number (Belsley et al., 1980):
      > 30     : HIGH — ill-conditioned design matrix
      <= 30    : ACCEPTABLE
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler

    print("\n--- Test 22: Factor Correlation Matrix & VIF Diagnostics ---")

    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score", "stability_score",
                  "similarity_rank", "sector_position"]
    avail = [c for c in score_cols if c in df.columns]

    if len(avail) < 2:
        print("  [SKIP] Fewer than 2 factor scores available")
        return pd.DataFrame()

    # 1. Correlation matrices (Pearson and Spearman)
    pearson_corr = df[avail].corr(method='pearson')
    spearman_corr = df[avail].corr(method='spearman')

    pearson_corr.to_csv(TABLES / "factor_score_correlations.csv")
    spearman_corr.to_csv(TABLES / "factor_score_spearman.csv")

    # 2. VIF for each factor
    X = df[avail].dropna()
    if len(X) < len(avail) + 2:
        print(f"  [SKIP] Too few complete observations ({len(X)}) for VIF with {len(avail)} factors")
        return pd.DataFrame()

    X_scaled = StandardScaler().fit_transform(X)

    vif_data = []
    for i, col in enumerate(avail):
        vif = variance_inflation_factor(X_scaled, i)
        vif_data.append({"factor": col, "VIF": round(vif, 2),
                         "concern": "HIGH" if vif > 10 else ("MODERATE" if vif > 5 else "LOW")})

    vif_df = pd.DataFrame(vif_data)
    vif_df.to_csv(TABLES / "factor_vif.csv", index=False)

    # 3. Condition number
    cond_num = np.linalg.cond(X_scaled)

    # 4. Print summary
    print(f"\n   Factor Score Diagnostics")
    print(f"   Condition number: {cond_num:.1f} ({'HIGH' if cond_num > 30 else 'ACCEPTABLE'})")
    for _, row in vif_df.iterrows():
        print(f"   {row['factor']:25s}: VIF={row['VIF']:.2f} ({row['concern']})")

    # Flag highly correlated pairs (Spearman |r| > 0.7)
    high_corr_found = False
    for i, c1 in enumerate(avail):
        for c2 in avail[i+1:]:
            r = spearman_corr.loc[c1, c2]
            if abs(r) > 0.7:
                print(f"   WARNING: {c1} <-> {c2}: Spearman r={r:.3f} (high correlation)")
                high_corr_found = True
    if not high_corr_found:
        print("   No factor pairs exceed the Spearman |r| > 0.7 threshold.")

    print(f"  [OK] Saved factor_score_correlations.csv, factor_score_spearman.csv, factor_vif.csv")
    return vif_df


# ---------- Multiple Testing Correction ----------
def apply_multiple_testing_correction(alpha: float = 0.05):
    """Collect ALL p-values from every result CSV, apply Benjamini-Hochberg
    FDR correction, and save a unified summary table.

    This addresses the multiple-comparisons problem: with 200+ tests at
    α = 0.05, we expect ~10 false positives by chance.  BH-FDR controls
    the *expected proportion* of false discoveries.

    Output: reports/tables/multiple_testing_correction.csv
    """
    print("\n--- Multiple Testing Correction (Benjamini-Hochberg FDR) ---")

    # Map of (csv_filename, p-value column) pairs to scan.
    # Each entry may contribute one or many p-values.
    P_COL_MAP = {
        # file stem → list of p-value columns
        "normality_tests":          ["shapiro_p", "jarque_bera_p", "ks_p"],
        "correlation_significance": ["pearson_p", "spearman_p", "kendall_p"],
        "esg_financial_regression": ["p_value"],
        "sector_anova":             ["anova_p", "welch_anova_p", "kruskal_p"],
        "country_differences":      ["ttest_p", "mannwhitney_p"],
        "pillar_correlations":      ["pearson_p", "spearman_p"],
        "quintile_q5_vs_q1":       ["ttest_p"],
        "factor_contributions":     ["p_value"],
        "profile_rank_correlation": ["spearman_p", "kendall_p"],
        "multiple_regression":      ["p_value", "f_pvalue"],
        "heteroscedasticity":       ["bp_p", "f_p"],
        "size_group_anova":         ["anova_p"],
        "sector_score_interaction": ["pearson_p"],
        "binary_variable_analysis": [
            "pb_p_ESG_composite", "pb_p_financial_score", "pb_p_pref_balanced",
            "chi2_p_sector",
        ],
        "binary_phi_coefficients":  ["p_value"],
        "ordinal_variable_analysis": [
            "spearman_p_ESG_composite", "spearman_p_G_score",
            "spearman_p_financial_score", "spearman_p_pref_balanced",
            "kruskal_p_sector",
        ],
        "friedman_test":            ["p_value"],
        "wilcoxon_pairwise":        ["wilcoxon_p"],
    }

    records = []  # (source_file, p_col, row_label, p_value)

    for stem, p_cols in P_COL_MAP.items():
        path = TABLES / f"{stem}.csv"
        if not path.exists():
            continue
        try:
            tbl = pd.read_csv(path, encoding="utf-8")
        except Exception:
            continue

        # Build a human-readable row label from the first few non-p-value columns
        label_candidates = [c for c in tbl.columns
                            if not c.startswith("p_") and "p_value" not in c.lower()
                            and "_p" not in c and tbl[c].dtype == object]
        if not label_candidates:
            label_candidates = [tbl.columns[0]] if len(tbl.columns) > 0 else []

        for p_col in p_cols:
            if p_col not in tbl.columns:
                continue
            for idx, row in tbl.iterrows():
                pval = row[p_col]
                if pd.isna(pval):
                    continue
                try:
                    pval = float(pval)
                except (ValueError, TypeError):
                    continue
                # Build label
                parts = []
                for lc in label_candidates[:3]:
                    v = row.get(lc)
                    if pd.notna(v):
                        parts.append(str(v))
                label = " | ".join(parts) if parts else f"row_{idx}"
                records.append({
                    "source_file": f"{stem}.csv",
                    "p_column": p_col,
                    "test_label": label,
                    "p_value": pval,
                })

    if not records:
        print("  [SKIP] No p-values found in result tables.")
        return

    summary = pd.DataFrame(records)
    n_tests = len(summary)
    print(f"  Collected {n_tests} p-values from {summary['source_file'].nunique()} tables")

    # --- Benjamini-Hochberg correction ---
    reject_bh, pvals_bh, _, _ = multipletests(
        summary["p_value"].values, alpha=alpha, method="fdr_bh",
    )
    summary["p_adjusted_bh"] = pvals_bh
    summary["significant_bh"] = reject_bh

    # --- Bonferroni correction (controls FWER) ---
    reject_bonf, pvals_bonf, _, _ = multipletests(
        summary["p_value"].values, alpha=alpha, method="bonferroni",
    )
    summary["p_adjusted_bonferroni"] = pvals_bonf
    summary["significant_bonferroni"] = reject_bonf

    # Keep the original (uncorrected) significance for comparison
    summary["significant_uncorrected"] = summary["p_value"] < alpha

    # Sort by adjusted p-value for readability
    summary = summary.sort_values("p_adjusted_bh").reset_index(drop=True)

    # Save
    summary.to_csv(TABLES / "multiple_testing_correction.csv", index=False, encoding="utf-8")

    # Print summary statistics
    n_sig_raw = summary["significant_uncorrected"].sum()
    n_sig_bh = summary["significant_bh"].sum()
    n_sig_bonf = summary["significant_bonferroni"].sum()
    n_lost = n_sig_raw - n_sig_bh
    n_lost_bonf = n_sig_raw - n_sig_bonf
    print(f"  Significant at α={alpha} (uncorrected) : {n_sig_raw} / {n_tests}")
    print(f"  Significant at α={alpha} (BH-corrected) : {n_sig_bh} / {n_tests}")
    print(f"  Significant at α={alpha} (Bonferroni)   : {n_sig_bonf} / {n_tests}")
    print(f"  Tests losing significance after BH correction: {n_lost}")
    print(f"  Tests losing significance after Bonferroni:    {n_lost_bonf}")
    print(f"  [OK] Saved multiple_testing_correction.csv")


# ---------- Post-Hoc Power Analysis ----------
def power_analysis(df):
    """Formal post-hoc power analysis for all major statistical tests.

    Reports:
      - Minimum detectable Spearman r at alpha=0.05, power=0.80 for N=56
      - Power achieved for observed key correlations
      - Minimum detectable Cohen's d for US vs India comparison
      - Sample sizes needed for observed effects to reach 80% power
      - Power limitations acknowledgment

    Output: reports/tables/power_analysis.csv
    """
    print("\n--- Post-Hoc Power Analysis ---")
    N = len(df)
    alpha = 0.05
    power_target = 0.80
    rows = []

    # ── 1. Correlation power (full sample) ─────────────────────────────
    r_min = _min_detectable_r(N, alpha, power_target)
    rows.append({
        "test_type": "correlation",
        "description": f"Minimum detectable |r| (N={N}, alpha={alpha}, power={power_target})",
        "n_total": N, "n1": N, "n2": None,
        "observed_effect": None,
        "min_detectable_effect": round(r_min, 4),
        "achieved_power": None,
        "n_needed_for_80pct": N,
        "interpretation": f"Correlations weaker than |r|={r_min:.3f} are undetectable",
    })

    # ── 2. Power for key observed correlations ─────────────────────────
    key_pairs = [
        ("ESG_composite", "financial_score", "ESG-Financial (design parameter ~0.35)"),
        ("ESG_composite", "roa", "ESG-ROA"),
        ("pref_balanced", "financial_score", "Balanced preference-Financial"),
        ("E_score", "S_score", "E pillar-S pillar"),
    ]
    for c1, c2, label in key_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        d = df[[c1, c2]].dropna()
        n_pair = len(d)
        if n_pair < 5:
            continue
        sr, _ = spearmanr(d[c1], d[c2])
        pwr = _correlation_power(sr, n_pair, alpha)
        n_need = _n_for_correlation(sr, alpha, power_target)
        ci_lo, ci_hi = _spearman_ci(sr, n_pair)
        rows.append({
            "test_type": "correlation",
            "description": f"Spearman r: {label}",
            "n_total": n_pair, "n1": n_pair, "n2": None,
            "observed_effect": round(sr, 4),
            "min_detectable_effect": round(r_min, 4),
            "achieved_power": round(pwr, 4),
            "n_needed_for_80pct": n_need,
            "interpretation": (
                f"r={sr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], power={pwr:.2f}"
                + (", UNDERPOWERED" if pwr < power_target else ", adequately powered")
            ),
        })

    # ── 3. Two-group comparison power (country: US vs India) ───────────
    if "country" in df.columns:
        countries = df["country"].dropna().unique()
        if len(countries) >= 2:
            c1_name, c2_name = countries[0], countries[1]
            n1 = len(df[df["country"] == c1_name])
            n2 = len(df[df["country"] == c2_name])
            d_min = _min_detectable_d(n1, n2, alpha, power_target)
            rows.append({
                "test_type": "two_group_ttest",
                "description": (
                    f"Min detectable Cohen's d ({c1_name} N={n1} vs "
                    f"{c2_name} N={n2})"
                ),
                "n_total": n1 + n2, "n1": n1, "n2": n2,
                "observed_effect": None,
                "min_detectable_effect": round(d_min, 4),
                "achieved_power": None,
                "n_needed_for_80pct": n1 + n2,
                "interpretation": (
                    f"Effects smaller than d={d_min:.3f} are undetectable; "
                    f"unequal groups ({n1} vs {n2}) reduce power"
                ),
            })

            # Power for observed Cohen's d values on key variables
            g1 = df[df["country"] == c1_name]
            g2 = df[df["country"] == c2_name]
            for col in ["ESG_composite", "financial_score", "pref_balanced"]:
                if col not in df.columns:
                    continue
                d1_vals = g1[col].dropna()
                d2_vals = g2[col].dropna()
                if len(d1_vals) < 3 or len(d2_vals) < 3:
                    continue
                d_obs = abs(_cohens_d(d1_vals, d2_vals))
                pwr = _ttest_power(d_obs, len(d1_vals), len(d2_vals), alpha)
                # Sample size needed (equal groups) for this effect
                from scipy.optimize import brentq
                try:
                    n_each = brentq(
                        lambda n: _ttest_power(d_obs, int(n), int(n), alpha) - power_target,
                        5, 5000, xtol=1
                    )
                    n_total_needed = int(np.ceil(n_each)) * 2
                except (ValueError, ZeroDivisionError):
                    n_total_needed = None
                rows.append({
                    "test_type": "two_group_ttest",
                    "description": f"Cohen's d for {col}: {c1_name} vs {c2_name}",
                    "n_total": len(d1_vals) + len(d2_vals),
                    "n1": len(d1_vals), "n2": len(d2_vals),
                    "observed_effect": round(d_obs, 4),
                    "min_detectable_effect": round(d_min, 4),
                    "achieved_power": round(pwr, 4),
                    "n_needed_for_80pct": n_total_needed,
                    "interpretation": (
                        f"d={d_obs:.3f}, power={pwr:.2f}"
                        + (", UNDERPOWERED" if pwr < power_target else ", adequately powered")
                    ),
                })

    # ── 4. ANOVA power note ────────────────────────────────────────────
    if "sector" in df.columns:
        n_sectors = df["sector"].nunique()
        rows.append({
            "test_type": "anova",
            "description": (
                f"Sector ANOVA ({n_sectors} groups, N={N}): "
                f"avg ~{N // n_sectors} per group"
            ),
            "n_total": N, "n1": None, "n2": None,
            "observed_effect": None,
            "min_detectable_effect": None,
            "achieved_power": None,
            "n_needed_for_80pct": None,
            "interpretation": (
                f"Small sector groups ({N // n_sectors}/group) limit power "
                f"for detecting medium effects (eta^2~0.06); "
                f"only large effects (eta^2>0.14) reliably detectable"
            ),
        })

    # ── 5. Summary power limitation statement ──────────────────────────
    rows.append({
        "test_type": "LIMITATION_SUMMARY",
        "description": "Overall power limitations for N=" + str(N),
        "n_total": N, "n1": None, "n2": None,
        "observed_effect": None,
        "min_detectable_effect": None,
        "achieved_power": None,
        "n_needed_for_80pct": None,
        "interpretation": (
            f"With N={N} mid-cap companies: (a) correlations below |r|={r_min:.3f} "
            f"are undetectable at 80% power; (b) subgroup analyses (country, sector) "
            f"have further reduced power due to small cell sizes; (c) multiple testing "
            f"(BH-FDR) applied to control false discovery rate. Results should be "
            f"interpreted as exploratory for effects near the detection threshold."
        ),
    })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "power_analysis.csv", index=False, encoding="utf-8")

    # Print key findings
    print(f"  N = {N} companies")
    print(f"  Min detectable |r| (alpha=0.05, power=0.80): {r_min:.4f}")
    if "country" in df.columns and len(df["country"].dropna().unique()) >= 2:
        countries = df["country"].dropna().unique()
        n1 = len(df[df["country"] == countries[0]])
        n2 = len(df[df["country"] == countries[1]])
        d_min = _min_detectable_d(n1, n2)
        print(f"  Min detectable Cohen's d (N1={n1}, N2={n2}): {d_min:.4f}")
    print(f"  [OK] Saved power_analysis.csv ({len(result)} rows)")
    return result


def main():
    print("=" * 70)
    print("STEP 04: COMPREHENSIVE STATISTICAL TESTS")
    print("=" * 70)

    df = load_data()
    test_descriptive(df)
    test_normality(df)
    test_correlations(df)
    test_esg_financial_regression(df)
    test_sector_differences(df)
    test_country_differences(df)
    test_pillar_correlations(df)
    test_quintile_analysis(df)
    test_factor_contributions(df)
    test_multicollinearity(df)
    test_profile_rank_correlation(df)
    test_top_bottom(df)
    test_multiple_regression(df)
    test_heteroscedasticity(df)
    test_decile_analysis(df)
    test_subgroup_analysis(df)
    test_sector_score_interaction(df)
    test_binary_variables(df)
    test_ordinal_variables(df)
    test_nonparametric_robustness(df)
    test_factor_score_correlations(df)
    compute_factor_diagnostics(df)

    # --- Multiple testing correction (must run after all tests) ---
    apply_multiple_testing_correction(alpha=0.05)

    # --- Post-hoc power analysis ---
    power_analysis(df)

    n_tables = len(list(TABLES.glob("*.csv")))
    print(f"\n[DONE] {n_tables} tables in {TABLES}/")
    print("Next: python scripts/05_weight_sensitivity.py")


if __name__ == "__main__":
    main()
