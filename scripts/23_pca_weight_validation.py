"""
Script 23: PCA Weight Validation
==================================
Formally compares configured investor-profile weights against PCA-derived
optimal weights to assess whether the weighting scheme is empirically
grounded in the cross-sectional variance structure of the 8 main factors.

The existing PCA analysis in 03_build_index.py (derive_pca_weight_rationale)
is post-hoc — it computes PCA after scoring and *suggests* weight ranges,
but those ranges are never used to constrain or validate the actual
configured weights.  This script makes the PCA analysis *prescriptive*.

Analyses:

  1. **PCA-suggested weights** — proportional to variance explained by each
     factor across the first K components explaining >= 80% of total variance.

  2. **Configured vs PCA weight comparison** — chi-squared goodness-of-fit,
     L1 (Manhattan) distance, and maximum absolute deviation.

  3. **PCA-constrained grid search** — search weights within PCA range +/-0.05,
     maximise cross-sectional Sharpe proxy (using ex_market momentum as
     return proxy, with appropriate caveats about circularity).

  4. **Factor independence (VIF)** — Variance Inflation Factor for all 8
     factors to detect multicollinearity.

  5. **Effective dimensionality** — number of PCA components needed to
     explain 80%, 90%, 95% of total variance.

Input:  data/processed/indexed_data.csv
        config/index_config.yaml (investor profiles)
Outputs:
  reports/tables/pca_weight_validation.csv
  reports/tables/pca_factor_vif.csv
  reports/tables/pca_effective_dimensionality.csv
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
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import product
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, ensure_dir, load_profile_weights
from src.constants import DEFAULT_WEIGHTS, load_profiles_from_config

TABLES = PROJECT_ROOT / "reports" / "tables"
ensure_dir(TABLES)

# The 8 main factor scores (excluding similarity_rank and sector_position
# which are secondary/derived and not primary investment factors).
MAIN_FACTORS = [
    "ESG_composite",
    "financial_score",
    "market_score",
    "operational_score",
    "risk_adjusted_score",
    "growth_score",
    "value_score",
    "stability_score",
]


# ============================================================================
# Helpers
# ============================================================================

def _standardise_factors(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Standardise the 8 main factor scores, returning (X_scaled, col_names).

    Drops rows with any NaN so PCA receives a complete matrix.
    """
    available = [c for c in MAIN_FACTORS if c in df.columns]
    sub = df[available].dropna()
    if len(sub) < 10:
        raise ValueError(
            f"Too few complete observations ({len(sub)}) for PCA; need >= 10."
        )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)
    return X_scaled, available


def _compute_preference(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Rank-based preference score (mirrors PreferenceScorer pipeline)."""
    score = pd.Series(0.0, index=df.index)
    total = sum(weights.values())
    if total < 1e-12:
        return score
    for comp, w in weights.items():
        if comp in df.columns:
            vals = df[comp].fillna(
                df[comp].median() if df[comp].notna().any() else 50
            )
            arr = vals.to_numpy(dtype=float)
            ranked = rankdata(arr, method="average")
            vals = pd.Series(ranked / len(ranked) * 100, index=df.index)
            score += (w / total) * vals
    return score.clip(0, 100)


def _cross_sectional_sharpe(
    df: pd.DataFrame,
    weights: dict[str, float],
    return_col: str,
    top_n: int = 20,
) -> float:
    """Cross-sectional information ratio: mean/std of top-N returns."""
    score = _compute_preference(df, weights)
    top_idx = score.nlargest(top_n).index
    rets = df.loc[top_idx, return_col].dropna()
    if len(rets) == 0 or rets.std() < 1e-10:
        return 0.0
    return rets.mean() / rets.std()


# ============================================================================
# 1. PCA-Suggested Weights
# ============================================================================

def pca_suggested_weights(
    X_scaled: np.ndarray,
    factor_names: list[str],
    variance_threshold: float = 0.80,
) -> dict[str, float]:
    """Derive PCA-proportional weights from the first K components.

    For each factor, its weight is proportional to its total contribution
    across the first K principal components (where K is the minimum number
    of components explaining >= `variance_threshold` of total variance).

    Contribution of factor j = sum_k |loading_kj| * var_ratio_k, for k=1..K.
    Weights are then normalised to sum to 1.
    """
    pca = PCA()
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.searchsorted(cumvar, variance_threshold) + 1)
    K = min(K, len(factor_names))

    loadings = np.abs(pca.components_[:K])          # (K, n_factors)
    var_ratios = pca.explained_variance_ratio_[:K]   # (K,)

    contributions = (loadings.T * var_ratios).sum(axis=1)  # (n_factors,)
    contributions = contributions / contributions.sum()

    return {name: float(contributions[i]) for i, name in enumerate(factor_names)}


# ============================================================================
# 2. Configured vs PCA Weight Comparison
# ============================================================================

def compare_weights(
    configured: dict[str, float],
    pca_weights: dict[str, float],
    factor_names: list[str],
) -> pd.DataFrame:
    """Compare configured vs PCA weights using multiple distance metrics.

    Returns a per-factor comparison DataFrame with chi-squared contribution,
    and appends summary rows with aggregate statistics.
    """
    rows = []
    for f in factor_names:
        cw = configured.get(f, 0.0)
        pw = pca_weights.get(f, 0.0)
        rows.append({
            "factor": f,
            "configured_weight": round(cw, 4),
            "pca_weight": round(pw, 4),
            "abs_deviation": round(abs(cw - pw), 4),
            "direction": "over" if cw > pw else ("under" if cw < pw else "match"),
        })

    result = pd.DataFrame(rows)

    # --- Aggregate metrics ---
    cw_arr = np.array([configured.get(f, 0.0) for f in factor_names])
    pw_arr = np.array([pca_weights.get(f, 0.0) for f in factor_names])

    # Normalise so both sum to 1 (they should already, but be safe)
    cw_arr = cw_arr / cw_arr.sum() if cw_arr.sum() > 0 else cw_arr
    pw_arr = pw_arr / pw_arr.sum() if pw_arr.sum() > 0 else pw_arr

    # L1 (Manhattan) distance
    l1 = float(np.abs(cw_arr - pw_arr).sum())

    # L2 (Euclidean) distance
    l2 = float(np.sqrt(((cw_arr - pw_arr) ** 2).sum()))

    # Maximum absolute deviation
    max_dev = float(np.abs(cw_arr - pw_arr).max())

    # Chi-squared goodness-of-fit: H0 = configured weights follow PCA distribution
    # Use PCA weights as expected frequencies (multiply by N = 1000 pseudo-obs)
    N_pseudo = 1000
    observed = cw_arr * N_pseudo
    expected = pw_arr * N_pseudo
    # Guard against zero expected values
    mask = expected > 0
    if mask.sum() >= 2:
        chi2, p_chi2 = stats.chisquare(observed[mask], f_exp=expected[mask])
    else:
        chi2, p_chi2 = np.nan, np.nan

    # Cosine similarity between weight vectors
    dot = np.dot(cw_arr, pw_arr)
    norm_c = np.linalg.norm(cw_arr)
    norm_p = np.linalg.norm(pw_arr)
    cosine_sim = dot / (norm_c * norm_p) if norm_c > 0 and norm_p > 0 else np.nan

    # Append summary rows
    summary_rows = pd.DataFrame([
        {"factor": "METRIC:L1_distance", "configured_weight": l1,
         "pca_weight": np.nan, "abs_deviation": np.nan, "direction": ""},
        {"factor": "METRIC:L2_distance", "configured_weight": l2,
         "pca_weight": np.nan, "abs_deviation": np.nan, "direction": ""},
        {"factor": "METRIC:max_abs_deviation", "configured_weight": max_dev,
         "pca_weight": np.nan, "abs_deviation": np.nan, "direction": ""},
        {"factor": "METRIC:chi2_statistic", "configured_weight": chi2,
         "pca_weight": np.nan, "abs_deviation": np.nan, "direction": ""},
        {"factor": "METRIC:chi2_p_value", "configured_weight": p_chi2,
         "pca_weight": np.nan, "abs_deviation": np.nan, "direction": ""},
        {"factor": "METRIC:cosine_similarity", "configured_weight": cosine_sim,
         "pca_weight": np.nan, "abs_deviation": np.nan, "direction": ""},
    ])
    result = pd.concat([result, summary_rows], ignore_index=True)
    return result


# ============================================================================
# 3. PCA-Constrained Grid Search
# ============================================================================

def pca_constrained_grid_search(
    df: pd.DataFrame,
    pca_weights: dict[str, float],
    factor_names: list[str],
    delta: float = 0.05,
    step: float = 0.025,
    top_n: int = 20,
) -> pd.DataFrame:
    """Search over weights within PCA range +/- delta.

    For each factor, the search range is [pca_w - delta, pca_w + delta],
    discretised in steps of `step`.  Because a full 8-dimensional grid is
    computationally prohibitive, we use a block-coordinate (cyclic) approach:
    optimise one factor at a time while holding the others fixed, iterating
    until convergence.

    Maximises cross-sectional Sharpe using ex-market momentum as the return
    proxy (price_momentum_3m or fallback).

    CAVEAT: momentum is a trailing measure, not a true forward return.  This
    optimisation should be interpreted as "which weights best separate past
    winners from losers", not as a backtest.
    """
    # Resolve return column
    return_col = None
    for rc in ["price_momentum_3m", "price_momentum_6m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break
    if return_col is None:
        print("  [WARN] No momentum column available; skipping grid search.")
        return pd.DataFrame()

    print(f"  Using return proxy: {return_col}")
    print(f"  CAVEAT: Trailing momentum is a proxy, NOT true forward returns.")
    print(f"          Results indicate factor separation power, not forecast accuracy.")

    # Build per-factor candidate grids
    grids: dict[str, np.ndarray] = {}
    for f in factor_names:
        pw = pca_weights.get(f, 0.125)
        lo = max(0.02, pw - delta)
        hi = min(0.40, pw + delta)
        grids[f] = np.arange(lo, hi + step / 2, step)

    # Initialise with PCA weights
    current = {f: pca_weights.get(f, 0.125) for f in factor_names}
    # Add secondary factors with small fixed weights
    secondary = {
        "similarity_rank": 0.03,
        "sector_position": 0.02,
    }
    best_sharpe = _cross_sectional_sharpe(
        df, {**current, **secondary}, return_col, top_n
    )

    # Block-coordinate optimisation (3 passes)
    n_evals = 0
    for _pass in range(3):
        improved = False
        for f in factor_names:
            best_val = current[f]
            for candidate in grids[f]:
                trial = current.copy()
                trial[f] = float(candidate)
                # Re-normalise main weights to sum to ~1.0 minus secondary
                main_sum = sum(trial.values())
                target_sum = 1.0 - sum(secondary.values())
                trial = {k: v * target_sum / main_sum for k, v in trial.items()}
                full_weights = {**trial, **secondary}
                cs = _cross_sectional_sharpe(df, full_weights, return_col, top_n)
                n_evals += 1
                if cs > best_sharpe:
                    best_sharpe = cs
                    best_val = float(candidate)
                    improved = True
            current[f] = best_val
        if not improved:
            break

    # Normalise final weights
    main_sum = sum(current.values())
    target_sum = 1.0 - sum(secondary.values())
    current = {k: v * target_sum / main_sum for k, v in current.items()}
    optimal = {**current, **secondary}

    print(f"  Grid search: {n_evals} evaluations, best CS-Sharpe = {best_sharpe:.4f}")

    # Build result table
    rows = []
    for f in factor_names:
        rows.append({
            "factor": f,
            "pca_weight": round(pca_weights.get(f, 0.0), 4),
            "pca_lower": round(max(0.02, pca_weights.get(f, 0.125) - delta), 4),
            "pca_upper": round(min(0.40, pca_weights.get(f, 0.125) + delta), 4),
            "grid_optimal_weight": round(optimal.get(f, 0.0), 4),
        })
    for f in secondary:
        rows.append({
            "factor": f,
            "pca_weight": np.nan,
            "pca_lower": np.nan,
            "pca_upper": np.nan,
            "grid_optimal_weight": round(secondary[f], 4),
        })
    rows.append({
        "factor": "METRIC:best_cs_sharpe",
        "pca_weight": np.nan,
        "pca_lower": np.nan,
        "pca_upper": np.nan,
        "grid_optimal_weight": round(best_sharpe, 4),
    })
    return pd.DataFrame(rows)


# ============================================================================
# 4. Variance Inflation Factor (VIF)
# ============================================================================

def compute_vif(X_scaled: np.ndarray, factor_names: list[str]) -> pd.DataFrame:
    """Compute VIF for each factor to detect multicollinearity.

    VIF_j = 1 / (1 - R²_j), where R²_j is from regressing factor j on all
    other factors.  VIF > 5 suggests moderate collinearity; VIF > 10 is severe.

    Uses manual OLS (via normal equation) to avoid a statsmodels dependency.
    """
    n_factors = X_scaled.shape[1]
    vifs = []

    for j in range(n_factors):
        y = X_scaled[:, j]
        X_other = np.delete(X_scaled, j, axis=1)
        # Add intercept
        X_design = np.column_stack([np.ones(len(y)), X_other])
        # OLS via normal equation: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
            y_hat = X_design @ beta
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        except np.linalg.LinAlgError:
            vif = np.nan

        interpretation = (
            "low" if vif < 5
            else "moderate" if vif < 10
            else "severe"
        )
        vifs.append({
            "factor": factor_names[j],
            "VIF": round(vif, 3),
            "R2_vs_others": round(r2, 4) if not np.isnan(vif) else np.nan,
            "collinearity": interpretation,
        })

    return pd.DataFrame(vifs)


# ============================================================================
# 5. Effective Dimensionality
# ============================================================================

def effective_dimensionality(X_scaled: np.ndarray, factor_names: list[str]) -> pd.DataFrame:
    """Report PCA explained variance and cumulative thresholds."""
    pca = PCA()
    pca.fit(X_scaled)

    rows = []
    cumvar = 0.0
    for i in range(len(factor_names)):
        var_i = pca.explained_variance_ratio_[i]
        cumvar += var_i
        rows.append({
            "component": f"PC{i + 1}",
            "eigenvalue": round(pca.explained_variance_[i], 4),
            "variance_explained": round(var_i, 4),
            "cumulative_variance": round(cumvar, 4),
        })

    result = pd.DataFrame(rows)

    # Append threshold summaries
    cumvar_arr = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.80, 0.90, 0.95]:
        k = int(np.searchsorted(cumvar_arr, threshold) + 1)
        k = min(k, len(factor_names))
        result = pd.concat([result, pd.DataFrame([{
            "component": f"THRESHOLD:{int(threshold * 100)}%",
            "eigenvalue": np.nan,
            "variance_explained": np.nan,
            "cumulative_variance": k,
        }])], ignore_index=True)

    return result


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("SCRIPT 23: PCA WEIGHT VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load data (mid-cap only, excludes large-cap benchmarks)
    # ------------------------------------------------------------------
    df = load_indexed_data(PROJECT_ROOT)
    print(f"[OK] Loaded {len(df)} companies (mid-cap universe)")

    # Scale similarity_rank / sector_position to 0-100 if needed
    for col in ["similarity_rank", "sector_position"]:
        if col in df.columns and df[col].max() <= 1.0:
            df[col] = df[col] * 100

    # ------------------------------------------------------------------
    # Standardise the 8 main factors for PCA
    # ------------------------------------------------------------------
    X_scaled, factor_names = _standardise_factors(df)
    print(f"[OK] Standardised {len(factor_names)} factors, "
          f"{X_scaled.shape[0]} complete observations")

    # ------------------------------------------------------------------
    # 1. PCA-suggested weights
    # ------------------------------------------------------------------
    print("\n--- 1. PCA-Suggested Weights ---")
    pca_wts = pca_suggested_weights(X_scaled, factor_names, variance_threshold=0.80)
    for f, w in sorted(pca_wts.items(), key=lambda x: -x[1]):
        print(f"  {f:25s}: {w:.4f}")

    # ------------------------------------------------------------------
    # 2. Compare vs configured weights (all profiles)
    # ------------------------------------------------------------------
    print("\n--- 2. Configured vs PCA Weight Comparison ---")

    profiles = load_profiles_from_config()
    all_comparisons = []

    for profile_name, profile_weights in profiles.items():
        print(f"\n  Profile: {profile_name}")
        # Extract only the 8 main factors from the profile and re-normalise
        main_wts = {
            f: profile_weights.get(f, 0.0) for f in factor_names
        }
        total_main = sum(main_wts.values())
        if total_main > 0:
            main_wts = {k: v / total_main for k, v in main_wts.items()}

        comp = compare_weights(main_wts, pca_wts, factor_names)
        comp.insert(0, "profile", profile_name)
        all_comparisons.append(comp)

        # Print per-factor comparison for this profile
        factor_rows = comp[~comp["factor"].str.startswith("METRIC:")]
        for _, row in factor_rows.iterrows():
            print(f"    {row['factor']:25s}: "
                  f"cfg={row['configured_weight']:.3f}  "
                  f"pca={row['pca_weight']:.3f}  "
                  f"dev={row['abs_deviation']:+.3f}  "
                  f"({row['direction']})")

        # Print aggregate metrics
        metric_rows = comp[comp["factor"].str.startswith("METRIC:")]
        for _, row in metric_rows.iterrows():
            metric_name = row["factor"].replace("METRIC:", "")
            val = row["configured_weight"]
            if pd.notna(val):
                print(f"    {metric_name:25s}: {val:.4f}")

    comparison_df = pd.concat(all_comparisons, ignore_index=True)
    out_path = TABLES / "pca_weight_validation.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"\n  [OK] Saved {out_path.name} ({len(comparison_df)} rows)")

    # ------------------------------------------------------------------
    # 3. PCA-constrained grid search
    # ------------------------------------------------------------------
    print("\n--- 3. PCA-Constrained Grid Search ---")
    grid_result = pca_constrained_grid_search(
        df, pca_wts, factor_names, delta=0.05, step=0.025, top_n=20,
    )
    if len(grid_result) > 0:
        grid_path = TABLES / "pca_grid_search_optimal.csv"
        grid_result.to_csv(grid_path, index=False)
        print(f"  [OK] Saved {grid_path.name}")

        # Print optimal vs PCA vs configured
        opt_rows = grid_result[~grid_result["factor"].str.startswith("METRIC:")]
        print("\n  Optimal weights (PCA-constrained):")
        for _, row in opt_rows.iterrows():
            cfg_w = DEFAULT_WEIGHTS.get(row["factor"], 0.0)
            pca_str = f"{row['pca_weight']:.3f}" if pd.notna(row["pca_weight"]) else "  n/a"
            print(f"    {row['factor']:25s}: "
                  f"optimal={row['grid_optimal_weight']:.3f}  "
                  f"pca={pca_str}  "
                  f"configured={cfg_w:.3f}")

    # ------------------------------------------------------------------
    # 4. Factor independence (VIF)
    # ------------------------------------------------------------------
    print("\n--- 4. Variance Inflation Factors ---")
    vif_df = compute_vif(X_scaled, factor_names)
    vif_path = TABLES / "pca_factor_vif.csv"
    vif_df.to_csv(vif_path, index=False)
    print(f"  [OK] Saved {vif_path.name}")

    for _, row in vif_df.iterrows():
        flag = ""
        if row["VIF"] >= 10:
            flag = " ** SEVERE **"
        elif row["VIF"] >= 5:
            flag = " * MODERATE *"
        print(f"    {row['factor']:25s}: VIF={row['VIF']:6.2f}  "
              f"R²={row['R2_vs_others']:.3f}  "
              f"[{row['collinearity']}]{flag}")

    max_vif = vif_df["VIF"].max()
    if max_vif < 5:
        print("  CONCLUSION: All VIF < 5 — no concerning multicollinearity.")
    elif max_vif < 10:
        print("  CONCLUSION: Some moderate collinearity detected (VIF 5-10). "
              "Factors are partially redundant but usable.")
    else:
        print("  CONCLUSION: Severe collinearity detected (VIF >= 10). "
              "Consider dropping or combining highly correlated factors.")

    # ------------------------------------------------------------------
    # 5. Effective dimensionality
    # ------------------------------------------------------------------
    print("\n--- 5. Effective Dimensionality ---")
    dim_df = effective_dimensionality(X_scaled, factor_names)
    dim_path = TABLES / "pca_effective_dimensionality.csv"
    dim_df.to_csv(dim_path, index=False)
    print(f"  [OK] Saved {dim_path.name}")

    # Print component table
    comp_rows = dim_df[~dim_df["component"].str.startswith("THRESHOLD:")]
    for _, row in comp_rows.iterrows():
        bar = "#" * int(row["variance_explained"] * 100)
        print(f"    {row['component']:5s}: eigenvalue={row['eigenvalue']:6.3f}  "
              f"var={row['variance_explained']:.3f}  "
              f"cumvar={row['cumulative_variance']:.3f}  {bar}")

    # Print thresholds
    thresh_rows = dim_df[dim_df["component"].str.startswith("THRESHOLD:")]
    for _, row in thresh_rows.iterrows():
        pct = row["component"].replace("THRESHOLD:", "")
        k = int(row["cumulative_variance"])
        print(f"    {pct} variance explained by {k} / {len(factor_names)} components")

    n_80 = dim_df[dim_df["component"] == "THRESHOLD:80%"]["cumulative_variance"].iloc[0]
    print(f"\n  Effective dimensionality (80%): {int(n_80)} of {len(factor_names)} factors")
    if int(n_80) <= 3:
        print("  NOTE: Only 3 or fewer components capture 80% of variance. "
              "Several factors may be near-redundant.")
    elif int(n_80) >= 6:
        print("  NOTE: 6+ components needed for 80% — factors are largely independent, "
              "supporting a multi-factor approach.")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # For the balanced profile specifically
    balanced_comp = comparison_df[comparison_df["profile"] == "balanced"]
    if len(balanced_comp) > 0:
        chi2_row = balanced_comp[balanced_comp["factor"] == "METRIC:chi2_p_value"]
        cosine_row = balanced_comp[balanced_comp["factor"] == "METRIC:cosine_similarity"]
        l1_row = balanced_comp[balanced_comp["factor"] == "METRIC:L1_distance"]

        if len(chi2_row) > 0:
            p_val = chi2_row.iloc[0]["configured_weight"]
            if pd.notna(p_val):
                if p_val > 0.05:
                    print(f"  Chi-squared test (balanced): p={p_val:.4f} — "
                          "configured weights are NOT significantly different from PCA-optimal (GOOD)")
                else:
                    print(f"  Chi-squared test (balanced): p={p_val:.4f} — "
                          "configured weights differ significantly from PCA-optimal")

        if len(cosine_row) > 0:
            cos_val = cosine_row.iloc[0]["configured_weight"]
            if pd.notna(cos_val):
                print(f"  Cosine similarity (balanced vs PCA): {cos_val:.4f}")

        if len(l1_row) > 0:
            l1_val = l1_row.iloc[0]["configured_weight"]
            if pd.notna(l1_val):
                print(f"  L1 distance (balanced vs PCA): {l1_val:.4f}")

    print(f"  Max VIF across factors: {max_vif:.2f}")
    print(f"  Effective dimensionality (80%): {int(n_80)} / {len(factor_names)}")

    print(f"\n[DONE] PCA weight validation complete. Results in {TABLES}/")
    print("       - pca_weight_validation.csv")
    print("       - pca_factor_vif.csv")
    print("       - pca_effective_dimensionality.csv")
    if len(grid_result) > 0:
        print("       - pca_grid_search_optimal.csv")


if __name__ == "__main__":
    main()
