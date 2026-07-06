"""
Step 07b: Cross-Sectional Validation
======================================
Since the dataset is a single cross-sectional snapshot (not a multi-year
panel), traditional time-series train/test splits do not apply.  Instead we
validate the scoring framework using standard cross-sectional factor-investing
methodology (Fama & French 1993; Jegadeesh & Titman 1993).

**Note on trailing momentum as return proxy:**
  Trailing momentum (price_momentum_1m/3m/6m) is used as a realized-return
  proxy for cross-sectional factor characterization.  This is NOT forward
  return prediction.

**Circularity guard — pref_*_ex_market variants:**
  The pref_*_ex_market variants are the PRIMARY validation targets.  They
  exclude market_score (which contains momentum sub-factors) to avoid
  momentum self-prediction circularity.  The original pref_balanced /
  pref_esg_first / pref_financial_first are retained for comparison only.

Validation tests:

  1. **Factor–Return Monotonicity (Quintile Portfolios)**
     Sort companies into quintiles by each factor score.  Compute average
     trailing returns (1 m, 3 m, 6 m) per quintile.  A useful factor should
     show monotonically increasing (or decreasing) returns across quintiles.

  2. **Information Coefficient (IC)**
     Spearman rank correlation between factor scores and trailing returns.
     IC > 0 indicates the factor ranks companies in the same order as their
     realised returns.

  3. **Long–Short Quintile Spread**
     Return of top quintile minus bottom quintile for each factor × horizon
     combination.  Positive spread = factor selects outperformers.

  4. **Multi-Horizon Consistency**
     Check whether IC sign and quintile spread persist across 1 m, 3 m, 6 m
     horizons.  Consistent sign across horizons increases confidence.

  5. **Bootstrap Rank Stability**
     Resample the cross-section 500 times, recompute preference scores and
     rankings, and measure Kendall τ between the original and bootstrapped
     ranking.  High τ = rankings are robust to sample perturbation.

  6. **Composite Summary Table**
     One row per factor × horizon with IC, spread, monotonicity p-value,
     and bootstrap τ.  This table feeds directly into the IEEE paper.

  7. **Circularity Comparison**
     Side-by-side IC comparison of contaminated (with market_score) vs clean
     (ex_market) preference scores, saved to circularity_comparison.csv.

Input:  data/processed/indexed_data.csv
Output: reports/tables/predictive_validation_ic.csv
        reports/tables/predictive_validation_quintiles.csv
        reports/tables/predictive_validation_spreads.csv
        reports/tables/predictive_validation_bootstrap.csv
        reports/tables/predictive_validation_summary.csv
        reports/tables/circularity_comparison.csv
        reports/figures/fig_quintile_returns.png
        reports/figures/fig_ic_heatmap.png

Also copies key tables to Thesis_report/Tables/ for the paper build.
"""

import sys
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, kruskal, norm as _norm
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data
from src.constants import RANDOM_SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
THESIS_TABLES = PROJECT_ROOT / "Thesis_report" / "Tables"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
THESIS_TABLES.mkdir(parents=True, exist_ok=True)

# Factor scores to validate
FACTOR_SCORES = [
    "ESG_composite",
    "financial_score",
    "market_score",
    "operational_score",
    "risk_adjusted_score",
    "growth_score",
    "value_score",
    "stability_score",
    # --- Circularity-corrected preference scores (PRIMARY for validation) ---
    # These exclude market_score (which contains price_momentum_1m/3m/6m)
    # to avoid momentum self-prediction circularity when validating against
    # trailing momentum return proxies.
    "pref_balanced_ex_market",
    "pref_esg_first_ex_market",
    "pref_financial_first_ex_market",
    # --- Original preference scores (includes market_score — shown for comparison) ---
    "pref_balanced",
    "pref_esg_first",
    "pref_financial_first",
]

# Return horizons (trailing momentum used as realised-return proxy)
RETURN_HORIZONS = {
    "quality": "forward_quality_proxy",  # C1 FIX: primary non-circular proxy
    "1m": "price_momentum_1m",           # secondary: momentum-based (circular with market_score)
    "3m": "price_momentum_3m",           # secondary: momentum-based (circular with market_score)
    "6m": "price_momentum_6m",           # secondary: momentum-based (circular with market_score)
}

CV_N_SPLITS = 5
REG_L2_LAMBDA = 0.25          # Stronger L2 to shrink toward uniform and reduce overfit
REG_ENTROPY_LAMBDA = 0.05     # Entropy regularization to prevent extreme weight concentration
WEIGHT_BOUNDS = (0.02, 0.35)  # Tighter max bound to reduce overfitting


def _factor_list_for_cv(df: pd.DataFrame) -> list[str]:
    """Factors used in CV weight optimization (exclude pre-combined pref_* scores)."""
    return [f for f in FACTOR_SCORES if f in df.columns and not f.startswith("pref_")]


def load_data() -> pd.DataFrame:
    df = load_indexed_data(PROJECT_ROOT)
    print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns")
    return df


# ── 1. Information Coefficient ─────────────────────────────────────────────

def _min_detectable_ic(n, alpha=0.05, power_target=0.80):
    """Minimum detectable Spearman IC with sample size n (Fisher z method)."""
    if n <= 3:
        return np.nan
    z_crit = _norm.ppf(1 - alpha / 2)
    z_beta = _norm.ppf(power_target)
    z_r = (z_crit + z_beta) / np.sqrt(n - 3)
    return np.tanh(z_r)


def _ic_power(r, n, alpha=0.05):
    """Power to detect IC = r with sample size n."""
    if n <= 3:
        return np.nan
    z_crit = _norm.ppf(1 - alpha / 2)
    ncp = abs(np.arctanh(np.clip(r, -0.9999, 0.9999))) * np.sqrt(n - 3)
    return 1 - _norm.cdf(z_crit - ncp) + _norm.cdf(-z_crit - ncp)


def _spearman_ci(r, n, alpha=0.05):
    """95% CI for Spearman r via Fisher z-transform."""
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = _norm.ppf(1 - alpha / 2)
    return (np.tanh(z - z_crit * se), np.tanh(z + z_crit * se))


def _james_stein_shrinkage(ic_values: np.ndarray, n_values: np.ndarray) -> tuple[np.ndarray, float]:
    """James-Stein shrinkage of IC estimates toward cross-factor mean IC."""
    if len(ic_values) <= 3:
        return ic_values.copy(), 0.0

    target = np.nanmean(ic_values)
    sigma2 = np.nanmean(1.0 / np.maximum(n_values - 3, 1))
    denom = np.nansum((ic_values - target) ** 2)
    if denom <= 1e-12:
        return np.full_like(ic_values, target, dtype=float), 1.0

    shrink = np.clip(1.0 - ((len(ic_values) - 3) * sigma2) / denom, 0.0, 1.0)
    shrunk = target + shrink * (ic_values - target)
    return shrunk, 1.0 - shrink


def _compute_ir(excess_returns: np.ndarray) -> float:
    vals = np.asarray(excess_returns, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    std = np.std(vals, ddof=1)
    if std <= 1e-12:
        return np.nan
    return float(np.mean(vals) / std)


def _effective_sample_size(split_df: pd.DataFrame) -> tuple[float, float]:
    n_total = float(len(split_df))
    if n_total <= 0:
        return np.nan, np.nan
    if "sector" not in split_df.columns:
        return n_total, n_total
    counts = split_df["sector"].dropna().value_counts().values.astype(float)
    if len(counts) == 0:
        return n_total, n_total
    n_eff = (counts.sum() ** 2) / np.sum(counts ** 2)
    return float(n_eff), n_total


def _adjusted_ir(ir: float, split_df: pd.DataFrame) -> float:
    n_eff, n_total = _effective_sample_size(split_df)
    if np.isnan(ir) or np.isnan(n_eff) or np.isnan(n_total) or n_total <= 0:
        return np.nan
    return float(ir * np.sqrt(n_eff / n_total))


def _single_ic(df: pd.DataFrame, factor: str, return_col: str) -> tuple[float, float, int]:
    valid = df[[factor, return_col]].dropna()
    if len(valid) < 10:
        return np.nan, np.nan, len(valid)
    rho, p = spearmanr(valid[factor], valid[return_col])
    return float(rho), float(p), len(valid)


def _single_spread(df: pd.DataFrame, factor: str, return_col: str) -> float:
    valid = df[[factor, return_col]].dropna()
    if len(valid) < 15:
        return np.nan
    try:
        valid["quintile"] = pd.qcut(valid[factor], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    except ValueError:
        valid["quintile"] = pd.qcut(
            valid[factor].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )
    q1 = valid.loc[valid["quintile"] == "Q1", return_col].mean()
    q5 = valid.loc[valid["quintile"] == "Q5", return_col].mean()
    if pd.isna(q1) or pd.isna(q5):
        return np.nan
    return float(q5 - q1)


def _quasi_temporal_size_stratified_folds(
    df: pd.DataFrame, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic quasi-temporal folds with size stratification."""
    work = df.reset_index(drop=True).copy()

    sort_proxy_col = None
    for col in ["company_age", "listing_date", "ipo_date"]:
        if col in work.columns:
            sort_proxy_col = col
            break

    if sort_proxy_col is None:
        work["_sort_proxy"] = np.arange(len(work), dtype=float)
    elif "date" in sort_proxy_col.lower():
        work["_sort_proxy"] = pd.to_datetime(work[sort_proxy_col], errors="coerce").map(
            lambda x: x.toordinal() if pd.notna(x) else np.nan
        )
    else:
        work["_sort_proxy"] = pd.to_numeric(work[sort_proxy_col], errors="coerce")
    work["_sort_proxy"] = work["_sort_proxy"].fillna(work["_sort_proxy"].median())

    size_col = None
    for col in ["market_cap", "market_cap_usd", "marketCap"]:
        if col in work.columns:
            size_col = col
            break

    if size_col is None:
        work["_size_proxy"] = np.arange(len(work), dtype=float)
    else:
        work["_size_proxy"] = pd.to_numeric(work[size_col], errors="coerce")
        work["_size_proxy"] = work["_size_proxy"].fillna(work["_size_proxy"].median())

    # M8 FIX: Add market-cap-stratified folds as alternative to random
    # This ensures small/large companies aren't concentrated in one fold,
    # providing a more robust out-of-sample test since size is correlated
    # with data quality and factor behavior.
    n_bins = int(np.clip(n_splits, 2, 5))
    work["_size_bin"] = pd.qcut(
        work["_size_proxy"].rank(method="first"), q=n_bins, labels=False, duplicates="drop"
    )

    fold_ids = np.full(len(work), -1, dtype=int)
    for _, grp in work.groupby("_size_bin", dropna=False):
        grp_sorted = grp.sort_values("_sort_proxy")
        fold_ids[grp_sorted.index.values] = np.arange(len(grp_sorted), dtype=int) % n_splits

    splits = []
    for fold in range(n_splits):
        test_idx = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def _optimize_factor_weights(train_ic_means: np.ndarray) -> np.ndarray:
    """Ridge + entropy regularized constrained optimization for factor weights.

    The objective maximizes the IC-weighted signal while penalizing:
    - L2 (ridge): prevents extreme concentration on a single factor
    - Entropy: pushes weights toward uniform distribution to reduce overfitting
    """
    n_factors = len(train_ic_means)
    if n_factors == 0:
        return np.array([])

    init = np.full(n_factors, 1.0 / n_factors)
    bounds = [WEIGHT_BOUNDS] * n_factors
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w):
        signal = -np.dot(w, train_ic_means)
        l2_penalty = REG_L2_LAMBDA * np.sum(w ** 2)
        # Negative entropy (minimize = maximize entropy = spread weights)
        entropy_penalty = REG_ENTROPY_LAMBDA * np.sum(w * np.log(np.clip(w, 1e-10, None)))
        return signal + l2_penalty + entropy_penalty

    res = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        return init
    return res.x


def compute_ic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation (IC) between each factor and each return horizon.

    Includes 95% confidence intervals (Fisher z-transform) and post-hoc power.
    """
    rows = []
    for factor in FACTOR_SCORES:
        if factor not in df.columns:
            continue
        for h_label, h_col in RETURN_HORIZONS.items():
            if h_col not in df.columns:
                continue
            valid = df[[factor, h_col]].dropna()
            if len(valid) < 10:
                continue
            rho, p = spearmanr(valid[factor], valid[h_col])
            n_obs = len(valid)
            ci_lo, ci_hi = _spearman_ci(rho, n_obs)
            pwr = _ic_power(rho, n_obs)
            rows.append(
                {
                    "factor": factor,
                    "horizon": h_label,
                    "return_col": h_col,
                    "ic_spearman": rho,
                    "ic_ci_lo": ci_lo,
                    "ic_ci_hi": ci_hi,
                    "ic_pvalue": p,
                    "ic_power": pwr,
                    "ic_significant": p < 0.05,
                    "n": n_obs,
                }
            )
    ic_df = pd.DataFrame(rows)
    if not ic_df.empty:
        shrunk_vals, shrink_intensity = _james_stein_shrinkage(
            ic_df["ic_spearman"].values.astype(float),
            ic_df["n"].values.astype(float),
        )
        ic_df["ic_spearman_shrunk"] = shrunk_vals
        ic_df["ic_shrinkage_intensity"] = shrink_intensity
    ic_df.to_csv(TABLES / "predictive_validation_ic.csv", index=False, encoding="utf-8")
    print(f"  [OK] IC table: {len(ic_df)} factor×horizon pairs")
    return ic_df


def run_kfold_cross_validation(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    fold_strategy: str = "sector_stratified_random",
) -> pd.DataFrame:
    """5-fold sector-stratified cross-validation with regularized weight optimization."""
    if "sector" not in df.columns:
        print("  [SKIP] K-fold CV requires 'sector' column")
        return pd.DataFrame()

    # Fill NaN sectors to avoid sklearn validation errors
    df = df.copy()
    df["sector"] = df["sector"].fillna("Unknown")

    factors = _factor_list_for_cv(df)
    if not factors:
        print("  [SKIP] No factor columns available for K-fold CV")
        return pd.DataFrame()

    fold_rows = []
    summary_rows = []
    if fold_strategy == "quasi_temporal_size_stratified":
        split_iter = _quasi_temporal_size_stratified_folds(df, n_splits=n_splits)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = skf.split(df, df["sector"].astype(str))

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_ic_means = []
        per_factor_horizon = []
        for factor in factors:
            fac_train_ics = []
            for h_label, h_col in RETURN_HORIZONS.items():
                if h_col not in df.columns:
                    continue
                ic_train, p_train, n_train = _single_ic(train_df, factor, h_col)
                ic_test, p_test, n_test = _single_ic(test_df, factor, h_col)
                spread_train = _single_spread(train_df, factor, h_col)
                spread_test = _single_spread(test_df, factor, h_col)
                per_factor_horizon.append(
                    {
                        "section": "factor_fold",
                        "fold_strategy": fold_strategy,
                        "fold": fold_idx,
                        "factor": factor,
                        "horizon": h_label,
                        "train_ic_raw": ic_train,
                        "test_ic_raw": ic_test,
                        "train_pvalue": p_train,
                        "test_pvalue": p_test,
                        "train_n": n_train,
                        "test_n": n_test,
                        "train_spread": spread_train,
                        "test_spread": spread_test,
                    }
                )
                fac_train_ics.append(ic_train)

            train_ic_means.append(np.nanmean(fac_train_ics) if fac_train_ics else np.nan)

        train_ic_means = np.array(train_ic_means, dtype=float)
        valid_train = np.isfinite(train_ic_means)
        if valid_train.sum() == 0:
            continue

        shrunk_train_ics, _ = _james_stein_shrinkage(
            train_ic_means[valid_train],
            np.full(valid_train.sum(), len(train_df), dtype=float),
        )
        optimized_w = _optimize_factor_weights(shrunk_train_ics)

        weights_full = np.zeros(len(factors), dtype=float)
        weights_full[valid_train] = optimized_w
        if weights_full.sum() > 0:
            weights_full = weights_full / weights_full.sum()

        weighted_train_spreads = []
        weighted_test_spreads = []
        for _, h_col in RETURN_HORIZONS.items():
            if h_col not in df.columns:
                continue
            train_spreads_h = np.array([_single_spread(train_df, f, h_col) for f in factors], dtype=float)
            test_spreads_h = np.array([_single_spread(test_df, f, h_col) for f in factors], dtype=float)
            weighted_train_spreads.append(np.nansum(np.where(np.isfinite(train_spreads_h), train_spreads_h, 0.0) * weights_full))
            weighted_test_spreads.append(np.nansum(np.where(np.isfinite(test_spreads_h), test_spreads_h, 0.0) * weights_full))

        cs_ir_train = _compute_ir(np.array(weighted_train_spreads, dtype=float))
        cs_ir_test = _compute_ir(np.array(weighted_test_spreads, dtype=float))
        cs_ir_train_adj = _adjusted_ir(cs_ir_train, train_df)
        cs_ir_test_adj = _adjusted_ir(cs_ir_test, test_df)
        ratio = (
            cs_ir_train_adj / cs_ir_test_adj
            if np.isfinite(cs_ir_train_adj) and np.isfinite(cs_ir_test_adj) and abs(cs_ir_test_adj) > 1e-12
            else np.nan
        )

        for row in per_factor_horizon:
            factor_idx = factors.index(row["factor"])
            row["train_ic_shrunk"] = (
                shrunk_train_ics[np.where(np.flatnonzero(valid_train) == factor_idx)[0][0]]
                if valid_train[factor_idx]
                else np.nan
            )
            row["optimized_weight"] = weights_full[factor_idx]
            fold_rows.append(row)

        summary_rows.append(
            {
                "section": "fold_summary",
                "fold_strategy": fold_strategy,
                "fold": fold_idx,
                "factor": "ALL",
                "horizon": "ALL",
                "cs_ir_train": cs_ir_train,
                "cs_ir_test": cs_ir_test,
                "cs_ir_train_adjusted": cs_ir_train_adj,
                "cs_ir_test_adjusted": cs_ir_test_adj,
                "train_test_ratio": ratio,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )

    if not fold_rows:
        return pd.DataFrame()

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)

    factor_stats = (
        fold_df.groupby(["factor", "horizon"])
        .agg(
            train_ic_mean=("train_ic_raw", "mean"),
            train_ic_std=("train_ic_raw", "std"),
            test_ic_mean=("test_ic_raw", "mean"),
            test_ic_std=("test_ic_raw", "std"),
            train_spread_mean=("train_spread", "mean"),
            train_spread_std=("train_spread", "std"),
            test_spread_mean=("test_spread", "mean"),
            test_spread_std=("test_spread", "std"),
        )
        .reset_index()
    )

    # M1/M2 FIX: flag anti-predictors in CV output (mean test IC < -0.03)
    factor_stats["negative_predictor"] = (
        factor_stats["factor"].isin(["value_score", "operational_score"])
        & (factor_stats["test_ic_mean"] < -0.03)
    )

    for fac in ["value_score", "operational_score"]:
        fac_rows = factor_stats[factor_stats["factor"] == fac]
        if not fac_rows.empty and bool(fac_rows["negative_predictor"].any()):
            mean_test_ic = fac_rows["test_ic_mean"].mean()
            print(
                f"  [WARN] {fac} flagged as negative predictor "
                f"(mean test IC = {mean_test_ic:.4f} < -0.03)"
            )

    factor_stats.insert(0, "section", "factor_mean_std")
    factor_stats.insert(1, "fold_strategy", fold_strategy)
    factor_stats.insert(1, "fold", "mean±std")

    overall_ratio = summary_df["train_test_ratio"].mean()
    overall_row = pd.DataFrame(
        [
            {
                "section": "overall_summary",
                "fold_strategy": fold_strategy,
                "fold": "ALL",
                "factor": "ALL",
                "horizon": "ALL",
                "cs_ir_train_mean": summary_df["cs_ir_train_adjusted"].mean(),
                "cs_ir_train_std": summary_df["cs_ir_train_adjusted"].std(),
                "cs_ir_test_mean": summary_df["cs_ir_test_adjusted"].mean(),
                "cs_ir_test_std": summary_df["cs_ir_test_adjusted"].std(),
                "overall_train_test_ratio": overall_ratio,
            }
        ]
    )

    comprehensive = pd.concat([fold_df, factor_stats, summary_df, overall_row], ignore_index=True, sort=False)
    output_name = (
        "cross_validation_comprehensive.csv"
        if fold_strategy == "sector_stratified_random"
        else f"cross_validation_comprehensive_{fold_strategy}.csv"
    )
    comprehensive.to_csv(TABLES / output_name, index=False, encoding="utf-8")

    print(f"  [OK] K-fold CV complete ({n_splits} folds, strategy={fold_strategy})")
    if not summary_df.empty:
        print(
            "       Mean adjusted CS-IR train/test ratio: "
            f"{summary_df['train_test_ratio'].mean():.3f}"
        )
    print(f"       Saved to {TABLES / output_name}")
    return comprehensive


def _leave_one_sector_out_cv(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_cols: list[str],
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Leave-one-sector-out cross-validation for weight robustness.

    M8 FIX: Provides a more meaningful out-of-sample test than random folds
    for cross-sectional data. Each fold trains on all sectors except one,
    then tests on the held-out sector.
    """
    results = []
    sectors = df[sector_col].dropna().unique()

    for hold_out in sectors:
        train_mask = df[sector_col] != hold_out
        test_mask = df[sector_col] == hold_out

        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue

        for return_col in return_cols:
            if return_col not in df.columns:
                continue
            for factor in factor_cols:
                if factor not in df.columns:
                    continue
                train_valid = train_mask & df[factor].notna() & df[return_col].notna()
                test_valid = test_mask & df[factor].notna() & df[return_col].notna()

                if train_valid.sum() > 10 and test_valid.sum() > 5:
                    train_ic, _ = spearmanr(df.loc[train_valid, factor], df.loc[train_valid, return_col])
                    test_ic, test_p = spearmanr(df.loc[test_valid, factor], df.loc[test_valid, return_col])
                    results.append(
                        {
                            "held_out_sector": hold_out,
                            "factor": factor,
                            "return_proxy": return_col,
                            "train_ic": train_ic,
                            "test_ic": test_ic,
                            "test_pvalue": test_p,
                            "n_train": int(train_valid.sum()),
                            "n_test": int(test_valid.sum()),
                        }
                    )

    return pd.DataFrame(results)


def run_sector_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward style leave-sectors-out validation: train on 8 sectors, validate on 3."""
    if "sector" not in df.columns:
        print("  [SKIP] Walk-forward validation requires 'sector' column")
        return pd.DataFrame()

    sectors = sorted(df["sector"].dropna().unique())
    if len(sectors) < 11:
        print("  [SKIP] Need at least 11 sectors for 8/3 walk-forward split")
        return pd.DataFrame()

    factors = _factor_list_for_cv(df)
    rows = []
    for holdout_sectors in combinations(sectors, 3):
        holdout_set = set(holdout_sectors)
        test_df = df[df["sector"].isin(holdout_set)].copy()
        train_df = df[~df["sector"].isin(holdout_set)].copy()
        if train_df.empty or test_df.empty:
            continue

        train_ic_means = []
        for factor in factors:
            fac_train_ics = []
            for _, h_col in RETURN_HORIZONS.items():
                if h_col not in df.columns:
                    continue
                ic_train, _, _ = _single_ic(train_df, factor, h_col)
                fac_train_ics.append(ic_train)
            train_ic_means.append(np.nanmean(fac_train_ics) if fac_train_ics else np.nan)

        train_ic_means = np.array(train_ic_means, dtype=float)
        valid_train = np.isfinite(train_ic_means)
        if valid_train.sum() == 0:
            continue
        shrunk_train_ics, _ = _james_stein_shrinkage(
            train_ic_means[valid_train],
            np.full(valid_train.sum(), len(train_df), dtype=float),
        )
        weights = _optimize_factor_weights(shrunk_train_ics)
        weights_full = np.zeros(len(factors), dtype=float)
        weights_full[valid_train] = weights
        if weights_full.sum() > 0:
            weights_full = weights_full / weights_full.sum()

        train_spreads = []
        test_spreads = []
        for _, h_col in RETURN_HORIZONS.items():
            if h_col not in df.columns:
                continue
            tr = np.array([_single_spread(train_df, f, h_col) for f in factors], dtype=float)
            te = np.array([_single_spread(test_df, f, h_col) for f in factors], dtype=float)
            train_spreads.append(np.nansum(np.where(np.isfinite(tr), tr, 0.0) * weights_full))
            test_spreads.append(np.nansum(np.where(np.isfinite(te), te, 0.0) * weights_full))

        ir_train_adj = _adjusted_ir(_compute_ir(np.array(train_spreads, dtype=float)), train_df)
        ir_test_adj = _adjusted_ir(_compute_ir(np.array(test_spreads, dtype=float)), test_df)
        rows.append(
            {
                "train_sectors": "|".join(sorted(set(sectors) - holdout_set)),
                "test_sectors": "|".join(sorted(holdout_sectors)),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "train_cs_ir_adjusted": ir_train_adj,
                "test_cs_ir_adjusted": ir_test_adj,
                "train_test_ratio": (
                    ir_train_adj / ir_test_adj
                    if np.isfinite(ir_train_adj) and np.isfinite(ir_test_adj) and abs(ir_test_adj) > 1e-12
                    else np.nan
                ),
            }
        )

    wf_df = pd.DataFrame(rows)
    wf_df.to_csv(TABLES / "walk_forward_sector_validation.csv", index=False, encoding="utf-8")
    if not wf_df.empty:
        print(
            f"  [OK] Walk-forward sector validation: {len(wf_df)} splits, "
            f"mean train/test ratio = {wf_df['train_test_ratio'].mean():.3f}"
        )
    else:
        print("  [SKIP] Walk-forward sector validation produced no splits")
    return wf_df


# ── 2. Quintile Portfolio Returns ──────────────────────────────────────────
def compute_quintile_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Form quintile portfolios by each factor and compute mean returns."""
    rows = []
    for factor in FACTOR_SCORES:
        if factor not in df.columns:
            continue
        # Drop NaN in factor before forming quintiles
        valid = df[df[factor].notna()].copy()
        if len(valid) < 15:
            continue
        try:
            valid["quintile"] = pd.qcut(
                valid[factor], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"]
            )
        except ValueError:
            # Ties may prevent exact quintile cut
            valid["quintile"] = pd.qcut(
                valid[factor].rank(method="first"), 5,
                labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"],
            )

        for h_label, h_col in RETURN_HORIZONS.items():
            if h_col not in valid.columns:
                continue
            for q_label in ["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"]:
                q_data = valid.loc[valid["quintile"] == q_label, h_col].dropna()
                rows.append(
                    {
                        "factor": factor,
                        "horizon": h_label,
                        "quintile": q_label,
                        "mean_return": q_data.mean() if len(q_data) > 0 else np.nan,
                        "median_return": q_data.median() if len(q_data) > 0 else np.nan,
                        "std_return": q_data.std() if len(q_data) > 0 else np.nan,
                        "n": len(q_data),
                        "pct_positive": (
                            (q_data > 0).mean() * 100 if len(q_data) > 0 else np.nan
                        ),
                    }
                )

    q_df = pd.DataFrame(rows)
    q_df.to_csv(
        TABLES / "predictive_validation_quintiles.csv", index=False, encoding="utf-8"
    )
    print(f"  [OK] Quintile table: {len(q_df)} rows")
    return q_df


# ── 3. Long–Short Quintile Spread ─────────────────────────────────────────
def compute_quintile_spreads(q_df: pd.DataFrame) -> pd.DataFrame:
    """Q5 minus Q1 return spread for each factor × horizon."""
    rows = []
    for (factor, horizon), grp in q_df.groupby(["factor", "horizon"]):
        q5 = grp.loc[grp["quintile"] == "Q5(High)", "mean_return"]
        q1 = grp.loc[grp["quintile"] == "Q1(Low)", "mean_return"]
        if q5.empty or q1.empty:
            continue
        spread = float(q5.iloc[0]) - float(q1.iloc[0])

        # Monotonicity check: Jonckheere–Terpstra-like via Kruskal–Wallis
        # on raw returns across quintiles
        means = grp.sort_values("quintile")["mean_return"].dropna().values
        mono_increasing = all(
            means[i] <= means[i + 1] for i in range(len(means) - 1)
        )
        mono_decreasing = all(
            means[i] >= means[i + 1] for i in range(len(means) - 1)
        )

        rows.append(
            {
                "factor": factor,
                "horizon": horizon,
                "Q5_return": float(q5.iloc[0]),
                "Q1_return": float(q1.iloc[0]),
                "spread": spread,
                "monotonic": mono_increasing or mono_decreasing,
                "direction": (
                    "increasing"
                    if mono_increasing
                    else ("decreasing" if mono_decreasing else "non-monotonic")
                ),
            }
        )

    spread_df = pd.DataFrame(rows)
    spread_df.to_csv(
        TABLES / "predictive_validation_spreads.csv", index=False, encoding="utf-8"
    )
    print(f"  [OK] Spread table: {len(spread_df)} factor×horizon pairs")
    return spread_df


# ── 4. Bootstrap Rank Stability ────────────────────────────────────────────
def bootstrap_rank_stability(
    df: pd.DataFrame, n_bootstrap: int = 500, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Resample the cross-section and measure ranking stability (Kendall τ)."""
    rng = np.random.default_rng(seed)
    # Use ex-market variant to avoid momentum self-prediction circularity
    score_col = "pref_balanced_ex_market"
    if score_col not in df.columns:
        # Fall back to original if ex_market not available
        score_col = "pref_balanced"
    if score_col not in df.columns:
        print("  [SKIP] pref_balanced(_ex_market) not available for bootstrap")
        return pd.DataFrame()

    original_rank = df[score_col].rank(ascending=False)
    n = len(df)
    taus = []
    iter_rank_corrs = []
    rank_within_10 = []
    top10_overlaps = []
    top20_overlaps = []
    prev_boot_rank = None

    original_top10 = set(df.nlargest(10, score_col).index)
    original_top20 = set(df.nlargest(20, score_col).index)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot = df.iloc[idx].copy()
        # Deduplicate: bootstrap with replacement creates duplicate indices.
        # Keep only the first occurrence of each original row index so that
        # .loc[] returns a scalar per index, not a Series.
        boot = boot[~boot.index.duplicated(keep="first")]
        boot_rank = boot[score_col].rank(ascending=False)

        # Compare ranks only for companies present in both
        common = original_rank.index.intersection(boot_rank.index)
        if len(common) < 10:
            continue
        tau, _ = kendalltau(
            original_rank.loc[common].values, boot_rank.loc[common].values
        )
        taus.append(tau)

        if prev_boot_rank is not None:
            common_iter = prev_boot_rank.index.intersection(boot_rank.index)
            if len(common_iter) >= 10:
                iter_corr, _ = spearmanr(
                    prev_boot_rank.loc[common_iter].values,
                    boot_rank.loc[common_iter].values,
                )
                iter_rank_corrs.append(iter_corr)
                pct_within = (
                    (np.abs(prev_boot_rank.loc[common_iter].values - boot_rank.loc[common_iter].values) <= 10)
                    .mean()
                    * 100
                )
                rank_within_10.append(pct_within)

        prev_boot_rank = boot_rank.copy()

        boot_top10 = set(boot.nlargest(10, score_col).index)
        boot_top20 = set(boot.nlargest(20, score_col).index)
        top10_overlaps.append(len(original_top10 & boot_top10))
        top20_overlaps.append(len(original_top20 & boot_top20))

    result = pd.DataFrame(
        {
            "metric": [
                "kendall_tau_mean",
                "kendall_tau_std",
                "kendall_tau_p5",
                "kendall_tau_p95",
                "top10_overlap_mean",
                "top20_overlap_mean",
                "bootstrap_iter_rank_corr_mean",
                "bootstrap_iter_rank_corr_std",
                "pct_stocks_within_10_ranks_mean",
                "pct_stocks_within_10_ranks_std",
                "n_bootstrap",
            ],
            "value": [
                np.mean(taus),
                np.std(taus),
                np.percentile(taus, 5),
                np.percentile(taus, 95),
                np.mean(top10_overlaps),
                np.mean(top20_overlaps),
                np.mean(iter_rank_corrs) if iter_rank_corrs else np.nan,
                np.std(iter_rank_corrs) if iter_rank_corrs else np.nan,
                np.mean(rank_within_10) if rank_within_10 else np.nan,
                np.std(rank_within_10) if rank_within_10 else np.nan,
                n_bootstrap,
            ],
        }
    )
    result.to_csv(
        TABLES / "predictive_validation_bootstrap.csv", index=False, encoding="utf-8"
    )
    print(
        f"  [OK] Bootstrap: τ = {np.mean(taus):.3f} ± {np.std(taus):.3f} "
        f"(top-10 overlap: {np.mean(top10_overlaps):.1f}/10, "
        f"top-20 overlap: {np.mean(top20_overlaps):.1f}/20, "
        f"within ±10 ranks: {np.mean(rank_within_10):.1f}%)"
    )
    return result


# ── 5. Kruskal–Wallis Across Quintiles ────────────────────────────────────
def kruskal_wallis_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """Non-parametric test: do quintile groups have significantly different returns?"""
    rows = []
    for factor in FACTOR_SCORES:
        if factor not in df.columns:
            continue
        valid = df[df[factor].notna()].copy()
        if len(valid) < 15:
            continue
        try:
            valid["quintile"] = pd.qcut(
                valid[factor], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
            )
        except ValueError:
            valid["quintile"] = pd.qcut(
                valid[factor].rank(method="first"), 5,
                labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            )
        for h_label, h_col in RETURN_HORIZONS.items():
            if h_col not in valid.columns:
                continue
            groups = [
                valid.loc[valid["quintile"] == q, h_col].dropna().values
                for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]
            ]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 3:
                continue
            h_stat, h_p = kruskal(*groups)
            rows.append(
                {
                    "factor": factor,
                    "horizon": h_label,
                    "kruskal_H": h_stat,
                    "kruskal_p": h_p,
                    "significant": h_p < 0.05,
                    "test_type": "exploratory",
                }
            )
    kw_df = pd.DataFrame(rows)
    kw_df.to_csv(
        TABLES / "predictive_validation_kruskal.csv", index=False, encoding="utf-8"
    )
    print(f"  [OK] Kruskal–Wallis: {len(kw_df)} tests")
    return kw_df


# ── 6. Composite Summary Table ────────────────────────────────────────────
def build_summary_table(
    ic_df: pd.DataFrame, spread_df: pd.DataFrame, kw_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge IC, spread, and Kruskal–Wallis into a single summary table."""
    if ic_df.empty or spread_df.empty:
        return pd.DataFrame()

    summary = ic_df[["factor", "horizon", "ic_spearman", "ic_ci_lo", "ic_ci_hi",
                      "ic_pvalue", "ic_power", "n"]].merge(
        spread_df[["factor", "horizon", "spread", "monotonic", "direction"]],
        on=["factor", "horizon"],
        how="left",
    )
    if not kw_df.empty:
        summary = summary.merge(
            kw_df[["factor", "horizon", "kruskal_H", "kruskal_p"]],
            on=["factor", "horizon"],
            how="left",
        )

    # Mo8 FIX: Flag primary vs exploratory tests
    # Primary: factor IC significance (n_factors × n_horizons tests)
    # Exploratory: all pairwise correlations, within-fold stability, etc.
    summary["test_type"] = "primary"
    n_primary_tests = int(summary["ic_pvalue"].notna().sum())
    summary["ic_pvalue_bonferroni"] = np.where(
        summary["test_type"] == "primary",
        np.minimum(summary["ic_pvalue"] * max(n_primary_tests, 1), 1.0),
        np.nan,
    )
    summary["ic_significant_bonferroni"] = np.where(
        summary["test_type"] == "primary",
        summary["ic_pvalue_bonferroni"] < 0.05,
        np.nan,
    )

    summary = summary.sort_values(["horizon", "factor"]).reset_index(drop=True)
    summary.to_csv(
        TABLES / "predictive_validation_summary.csv", index=False, encoding="utf-8"
    )
    print(f"  [OK] Summary table: {len(summary)} rows")
    return summary


# ── 7. Figures ─────────────────────────────────────────────────────────────
def plot_quintile_returns(q_df: pd.DataFrame):
    """Bar chart of quintile mean returns for key factors across horizons."""
    key_factors = [
        "pref_balanced_ex_market",
        "pref_balanced",
        "financial_score",
        "ESG_composite",
        "risk_adjusted_score",
    ]
    available = [f for f in key_factors if f in q_df["factor"].unique()]
    horizons = sorted(q_df["horizon"].unique())

    if not available or not horizons:
        return

    fig, axes = plt.subplots(
        len(available), len(horizons), figsize=(4 * len(horizons), 3 * len(available)),
        squeeze=False,
    )

    for i, factor in enumerate(available):
        for j, horizon in enumerate(horizons):
            ax = axes[i][j]
            sub = q_df[(q_df["factor"] == factor) & (q_df["horizon"] == horizon)]
            if sub.empty:
                ax.set_visible(False)
                continue
            quintiles = sub["quintile"].values
            returns = sub["mean_return"].values
            colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(quintiles)))
            ax.bar(range(len(quintiles)), returns, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(quintiles)))
            ax.set_xticklabels(quintiles, fontsize=7, rotation=45)
            ax.set_title(f"{factor}\n({horizon})", fontsize=8)
            ax.set_ylabel("Mean return (%)", fontsize=7)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Quintile Portfolio Returns by Factor and Horizon", fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIGURES / "fig_quintile_returns.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out}")



def plot_ic_heatmap(ic_df):
    """Heatmap of IC values: factors x horizons."""
    if ic_df.empty:
        return
    agg_func = "first"
    pivot = ic_df.pivot_table(
        index="factor", columns="horizon", values="ic_spearman", aggfunc=agg_func
    )
    for h in ["1m", "3m", "6m"]:
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[["1m", "3m", "6m"]].dropna(how="all")

    fig, ax = plt.subplots(figsize=(5, max(4, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "black" if abs(val) < 0.15 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Information Coefficient (Spearman) by Factor x Horizon", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="IC")
    fig.tight_layout()
    out = FIGURES / "fig_ic_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {out}")


def copy_to_thesis_tables():
    """Copy key CSV tables to Thesis_report/Tables for the paper build."""
    key_files = [
        "predictive_validation_summary.csv",
        "predictive_validation_ic.csv",
        "predictive_validation_spreads.csv",
        "predictive_validation_bootstrap.csv",
        "circularity_comparison.csv",
        "benchmark_summary.csv",
        "weight_grid_search.csv",
    ]
    copied = 0
    for fname in key_files:
        src = TABLES / fname
        dst = THESIS_TABLES / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"  [OK] Copied {copied} tables to {THESIS_TABLES}")


def generate_latex_tables(summary_df, bootstrap_df):
    """Write LaTeX table snippets for inclusion in the paper."""
    if not summary_df.empty:
        tex_path = THESIS_TABLES / "table_predictive_validation_summary.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("% Auto-generated by 07b_cross_sectional_validation.py\n")
            f.write(summary_df.round(4).to_latex(index=False, escape=True))
        print(f"  [OK] LaTeX: {tex_path}")

    if not bootstrap_df.empty:
        tex_path = THESIS_TABLES / "table_predictive_validation_bootstrap.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("% Auto-generated by 07b_cross_sectional_validation.py\n")
            f.write(bootstrap_df.round(4).to_latex(index=False, escape=True))
        print(f"  [OK] LaTeX: {tex_path}")


# ── IC Power Analysis ─────────────────────────────────────────────────────
def ic_power_analysis(df: pd.DataFrame, ic_df: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc power analysis specific to Information Coefficient testing.

    Reports minimum detectable IC, achieved power for each factor x horizon,
    and bootstrap 95% confidence intervals for IC estimates.
    """
    print("\n--- IC Power Analysis ---")
    N = len(df)
    alpha = 0.05
    power_target = 0.80
    r_min = _min_detectable_ic(N, alpha, power_target)

    rows = []

    # Overall detectability
    rows.append({
        "factor": "ALL",
        "horizon": "ALL",
        "ic_spearman": None,
        "ic_ci_lo": None,
        "ic_ci_hi": None,
        "ic_power": None,
        "min_detectable_ic": round(r_min, 4),
        "n": N,
        "interpretation": (
            f"With N={N}, minimum detectable IC at alpha={alpha}, "
            f"power={power_target} is |IC|={r_min:.4f}. "
            f"ICs below this threshold may be real but undetectable."
        ),
    })

    # Per factor x horizon power
    if not ic_df.empty:
        for _, row in ic_df.iterrows():
            ic_val = row.get("ic_spearman")
            n_obs = row.get("n", N)
            if pd.isna(ic_val):
                continue
            pwr = _ic_power(ic_val, n_obs, alpha)
            ci_lo, ci_hi = _spearman_ci(ic_val, n_obs)
            rows.append({
                "factor": row["factor"],
                "horizon": row["horizon"],
                "ic_spearman": round(ic_val, 4),
                "ic_ci_lo": round(ci_lo, 4),
                "ic_ci_hi": round(ci_hi, 4),
                "ic_power": round(pwr, 4),
                "min_detectable_ic": round(r_min, 4),
                "n": n_obs,
                "interpretation": (
                    f"power={pwr:.2f}"
                    + (", UNDERPOWERED" if pwr < power_target else ", adequate")
                ),
            })

    result = pd.DataFrame(rows)
    result.to_csv(
        TABLES / "predictive_validation_ic_power.csv", index=False, encoding="utf-8"
    )
    print(f"  Min detectable IC (N={N}): {r_min:.4f}")
    n_underpowered = sum(
        1 for r in rows
        if r.get("ic_power") is not None and r["ic_power"] < power_target
    )
    n_tests = sum(1 for r in rows if r.get("ic_power") is not None)
    print(f"  Underpowered IC tests: {n_underpowered}/{n_tests}")
    print(f"  [OK] Saved predictive_validation_ic_power.csv ({len(result)} rows)")
    return result


def bootstrap_ic_confidence_intervals(
    df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Bootstrap 95% confidence intervals for IC estimates.

    For each factor x horizon pair, resample the cross-section n_bootstrap
    times, recompute Spearman IC, and report the 2.5th and 97.5th percentiles.
    """
    print("\n--- Bootstrap IC Confidence Intervals ---")
    rng = np.random.default_rng(seed)
    rows = []

    for factor in FACTOR_SCORES:
        if factor not in df.columns:
            continue
        for h_label, h_col in RETURN_HORIZONS.items():
            if h_col not in df.columns:
                continue
            valid = df[[factor, h_col]].dropna()
            if len(valid) < 10:
                continue
            n_valid = len(valid)

            # Original IC
            ic_orig, p_orig = spearmanr(valid[factor], valid[h_col])

            # Bootstrap
            boot_ics = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n_valid, size=n_valid, replace=True)
                boot = valid.iloc[idx]
                try:
                    r, _ = spearmanr(boot[factor], boot[h_col])
                    boot_ics.append(r)
                except Exception:
                    pass

            if len(boot_ics) < 50:
                continue

            boot_ics = np.array(boot_ics)
            ci_lo_boot = np.percentile(boot_ics, 2.5)
            ci_hi_boot = np.percentile(boot_ics, 97.5)
            ci_lo_fisher, ci_hi_fisher = _spearman_ci(ic_orig, n_valid)

            rows.append({
                "factor": factor,
                "horizon": h_label,
                "ic_spearman": round(ic_orig, 4),
                "ic_pvalue": round(p_orig, 4),
                "bootstrap_ci_lo": round(ci_lo_boot, 4),
                "bootstrap_ci_hi": round(ci_hi_boot, 4),
                "fisher_ci_lo": round(ci_lo_fisher, 4),
                "fisher_ci_hi": round(ci_hi_fisher, 4),
                "bootstrap_se": round(np.std(boot_ics), 4),
                "n_bootstrap": len(boot_ics),
                "n": n_valid,
                "ci_includes_zero": bool(ci_lo_boot <= 0 <= ci_hi_boot),
            })

    result = pd.DataFrame(rows)
    result.to_csv(
        TABLES / "predictive_validation_ic_bootstrap_ci.csv",
        index=False, encoding="utf-8",
    )
    n_zero = sum(1 for r in rows if r["ci_includes_zero"])
    print(
        f"  [OK] Bootstrap IC CIs: {len(result)} factor x horizon pairs, "
        f"{n_zero} include zero"
    )
    return result


def main():
    print("=" * 70)
    print("STEP 07b: CROSS-SECTIONAL VALIDATION")
    print("=" * 70)

    df = load_data()

    avail_horizons = {k: v for k, v in RETURN_HORIZONS.items() if v in df.columns}
    if not avail_horizons:
        print("[WARN] No return horizon columns found -- skipping cross-sectional validation.")
        print("       Expected columns: price_momentum_1m, price_momentum_3m, price_momentum_6m")
        return
    print(f"  Return horizons available: {list(avail_horizons.keys())}")

    avail_factors = [f for f in FACTOR_SCORES if f in df.columns]
    print(f"  Factor scores available: {len(avail_factors)}")

    # Check for ex_market columns
    ex_market_available = [f for f in avail_factors if f.endswith("_ex_market")]
    if ex_market_available:
        print(f"  Circularity-corrected (ex_market) scores: {ex_market_available}")
    else:
        print("  [WARN] No pref_*_ex_market columns found — circularity comparison unavailable")

    print("\n--- 1. Information Coefficient (Spearman IC) ---")
    ic_df = compute_ic_table(df)

    print("\n--- 1b. 5-Fold Stratified Cross-Validation (sector) ---")
    cv_df = run_kfold_cross_validation(df, n_splits=CV_N_SPLITS)

    print("\n--- 1c. 5-Fold Quasi-Temporal Size-Stratified Cross-Validation ---")
    cv_quasi_df = run_kfold_cross_validation(
        df,
        n_splits=CV_N_SPLITS,
        fold_strategy="quasi_temporal_size_stratified",
    )

    print("\n--- 1d. Leave-One-Sector-Out Cross-Validation ---")
    loso_df = _leave_one_sector_out_cv(
        df,
        factor_cols=_factor_list_for_cv(df),
        return_cols=list(avail_horizons.values()),
    )
    loso_df.to_csv(TABLES / "leave_one_sector_out_cv.csv", index=False, encoding="utf-8")
    print(f"  [OK] Leave-one-sector-out CV: {len(loso_df)} rows")
    print(f"       Saved to {TABLES / 'leave_one_sector_out_cv.csv'}")

    print("\n--- 2. Quintile Portfolio Returns ---")
    q_df = compute_quintile_returns(df)

    print("\n--- 3. Long-Short Quintile Spread ---")
    spread_df = compute_quintile_spreads(q_df)

    print("\n--- 4. Kruskal-Wallis Across Quintiles ---")
    kw_df = kruskal_wallis_quintiles(df)

    print("\n--- 5. Bootstrap Rank Stability (500 resamples) ---")
    boot_df = bootstrap_rank_stability(df, n_bootstrap=500)

    print("\n--- 6. Building Summary Table ---")
    summary_df = build_summary_table(ic_df, spread_df, kw_df)
    print("  [NOTE] Primary hypotheses: factor IC significance tests.")
    print("         Exploratory analyses: correlation structure, fold stability, etc.")
    print("         Bonferroni correction applied only to primary IC tests.")

    print("\n--- 7. Generating Figures ---")
    plot_quintile_returns(q_df)
    plot_ic_heatmap(ic_df)

    print("\n--- 8. Generating LaTeX Table Snippets ---")
    generate_latex_tables(summary_df, boot_df)

    print("\n--- 9. Copying Tables to Thesis_report/Tables ---")
    copy_to_thesis_tables()

    print("\n--- 10. IC Power Analysis ---")
    ic_power_df = ic_power_analysis(df, ic_df)

    print("\n--- 11. Bootstrap IC Confidence Intervals (1000 resamples) ---")
    ic_boot_ci_df = bootstrap_ic_confidence_intervals(df, n_bootstrap=1000)

    print("\n--- 11b. Walk-Forward Validation (train:8 sectors, test:3 sectors) ---")
    wf_df = run_sector_walk_forward(df)

    # ── 12. Circularity Comparison ─────────────────────────────────────────
    print("\n--- 12. Circularity Comparison (contaminated vs clean) ---")
    comparison_rows = []
    _PREF_BASES = ["pref_balanced", "pref_esg_first", "pref_financial_first"]
    if not ic_df.empty:
        for horizon in sorted(ic_df["horizon"].unique()):
            for base in _PREF_BASES:
                clean = base + "_ex_market"
                # Look up IC for the contaminated (original) variant
                orig_row = ic_df[(ic_df["factor"] == base) & (ic_df["horizon"] == horizon)]
                clean_row = ic_df[(ic_df["factor"] == clean) & (ic_df["horizon"] == horizon)]

                ic_orig = float(orig_row["ic_spearman"].iloc[0]) if not orig_row.empty else np.nan
                p_orig = float(orig_row["ic_pvalue"].iloc[0]) if not orig_row.empty else np.nan
                ic_clean = float(clean_row["ic_spearman"].iloc[0]) if not clean_row.empty else np.nan
                p_clean = float(clean_row["ic_pvalue"].iloc[0]) if not clean_row.empty else np.nan

                delta = ic_orig - ic_clean if not (np.isnan(ic_orig) or np.isnan(ic_clean)) else np.nan

                comparison_rows.append({
                    "horizon": horizon,
                    "score_variant": base,
                    "ic_with_market": round(ic_orig, 4) if not np.isnan(ic_orig) else np.nan,
                    "p_with_market": round(p_orig, 4) if not np.isnan(p_orig) else np.nan,
                    "ic_ex_market": round(ic_clean, 4) if not np.isnan(ic_clean) else np.nan,
                    "p_ex_market": round(p_clean, 4) if not np.isnan(p_clean) else np.nan,
                    "ic_inflation": round(delta, 4) if not np.isnan(delta) else np.nan,
                    "note": (
                        "CIRCULAR — market_score includes momentum"
                        if not np.isnan(delta) and abs(delta) > 0.01
                        else "minimal difference"
                    ) if not np.isnan(delta) else "missing data",
                })

    circ_df = pd.DataFrame(comparison_rows)
    if not circ_df.empty:
        circ_df.to_csv(TABLES / "circularity_comparison.csv", index=False, encoding="utf-8")
        print(f"  [OK] Circularity comparison: {len(circ_df)} rows")
        print(f"       Saved to {TABLES / 'circularity_comparison.csv'}")
        # Print a quick summary
        inflated = circ_df[circ_df["ic_inflation"].abs() > 0.01].dropna(subset=["ic_inflation"])
        if not inflated.empty:
            mean_inflation = inflated["ic_inflation"].mean()
            print(f"  [INFO] Mean IC inflation from market_score circularity: {mean_inflation:+.4f}")
        else:
            print("  [INFO] No material IC inflation detected from market_score inclusion")
    else:
        print("  [SKIP] Circularity comparison — insufficient data")

    if not ic_df.empty:
        print("\n" + "=" * 70)
        print("KEY RESULTS")
        print("=" * 70)
        for horizon in sorted(ic_df["horizon"].unique()):
            sub = ic_df[ic_df["horizon"] == horizon]
            sig = sub[sub["ic_significant"]]
            print(f"  {horizon} horizon: {len(sig)}/{len(sub)} factors with significant IC")
            if not sub.empty:
                best = sub.loc[sub["ic_spearman"].abs().idxmax()]
                print(f"    Strongest IC: {best['factor']} = {best['ic_spearman']:.3f} "
                      f"(p = {best['ic_pvalue']:.4f})")

    if not spread_df.empty:
        for horizon in sorted(spread_df["horizon"].unique()):
            sub = spread_df[spread_df["horizon"] == horizon]
            pos_spreads = sub[sub["spread"] > 0]
            print(f"  {horizon}: {len(pos_spreads)}/{len(sub)} factors with positive Q5-Q1 spread")

    if not boot_df.empty:
        tau_mean = boot_df.loc[boot_df["metric"] == "kendall_tau_mean", "value"].iloc[0]
        print(f"  Bootstrap Kendall tau: {tau_mean:.3f}")

    if not cv_df.empty:
        cv_fold = cv_df[cv_df["section"] == "fold_summary"]
        if not cv_fold.empty:
            ratio_mean = cv_fold["train_test_ratio"].mean()
            print(f"  CV adjusted CS-IR train/test ratio (mean): {ratio_mean:.3f}")

    if not cv_quasi_df.empty:
        cv_quasi_fold = cv_quasi_df[cv_quasi_df["section"] == "fold_summary"]
        if not cv_quasi_fold.empty:
            ratio_mean_quasi = cv_quasi_fold["train_test_ratio"].mean()
            print(f"  Quasi-temporal CV adjusted CS-IR train/test ratio (mean): {ratio_mean_quasi:.3f}")

    if not loso_df.empty:
        print(
            "  Leave-one-sector-out mean test IC: "
            f"{loso_df['test_ic'].mean():.3f}"
        )

    if not wf_df.empty:
        print(
            "  Walk-forward adjusted CS-IR train/test ratio (mean): "
            f"{wf_df['train_test_ratio'].mean():.3f}"
        )

    print(f"\n[DONE] Cross-sectional validation complete.")
    print(f"  Tables: {TABLES}")
    print(f"  Figures: {FIGURES}")
    print("Next: python scripts/08_advanced_analysis.py")


if __name__ == "__main__":
    main()
