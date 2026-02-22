"""
Step 05: Weight Sensitivity & Optimization Analysis
=====================================================
Tests how index rankings change under different weight configurations:
  1. Sensitivity of each weight parameter (+/- 10%)
  2. Grid search over preference score weights (10-factor)
  3. Rank stability analysis (Kendall tau across weight changes)
  4. Return-based optimal weight selection (maximize actual portfolio Sharpe)
  5. Profile comparison under different weight regimes
  6. PCA-informed weight validation

Input:  data/processed/indexed_data.csv, config/index_config.yaml
Output: reports/tables/weight_*.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from itertools import product
import warnings
warnings.filterwarnings("ignore")

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# All 10 score components
WEIGHT_NAMES = [
    "ESG_composite", "financial_score", "market_score", "operational_score",
    "risk_adjusted_score", "growth_score", "value_score", "stability_score",
    "similarity_rank", "sector_position",
]

# Default weights — empirically calibrated from PCA contribution analysis
# and grid search over actual portfolio Sharpe ratios (see Section 8 of report).
# Rationale: financial_score and growth_score carry strongest return-predictive
# signal; ESG maintains integration objective; remaining factors provide
# diversification and risk management.
DEFAULT_WEIGHTS = {
    "ESG_composite": 0.15, "financial_score": 0.25,
    "market_score": 0.10, "operational_score": 0.10,
    "risk_adjusted_score": 0.08, "growth_score": 0.12,
    "value_score": 0.08, "stability_score": 0.05,
    "similarity_rank": 0.04, "sector_position": 0.03,
}


def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} companies")
    return df


def compute_preference(df, weights):
    """Compute preference score with given weights dict."""
    score = pd.Series(0.0, index=df.index)
    total = sum(weights.values())
    for comp, w in weights.items():
        if comp in df.columns:
            vals = df[comp].fillna(df[comp].median()) / 100.0
            score += (w / total) * vals * 100
    return score.clip(0, 100)


def compute_portfolio_sharpe(df, weights, return_col="price_momentum_3m", top_n=20):
    """Compute actual portfolio Sharpe ratio: select top_n by score, average their returns."""
    score = compute_preference(df, weights)
    top_idx = score.nlargest(top_n).index
    rets = df.loc[top_idx, return_col].dropna()
    if len(rets) < 5 or rets.std() < 1e-10:
        return 0.0
    return rets.mean() / rets.std()


def _prepare_df(df):
    """Return a copy with similarity_rank/sector_position scaled to 0-100."""
    df = df.copy()
    for col in ["similarity_rank", "sector_position"]:
        if col in df.columns and df[col].max() <= 1.0:
            df[col] = df[col] * 100
    return df


# ---------------------------------------------------------------------------
# 1. Single-Parameter Sensitivity
# ---------------------------------------------------------------------------
def sensitivity_single_param(df):
    """Change each weight +/-10% and measure rank correlation."""
    print("\n--- Weight Sensitivity: Single Parameter ---")

    base_score = compute_preference(df, DEFAULT_WEIGHTS)
    base_rank = base_score.rank(ascending=False)

    rows = []
    for param in DEFAULT_WEIGHTS:
        if param not in df.columns:
            continue
        for delta in [-0.10, -0.05, +0.05, +0.10]:
            new_weights = DEFAULT_WEIGHTS.copy()
            old_val = new_weights[param]
            new_val = max(0.01, old_val + delta)
            new_weights[param] = new_val
            total = sum(new_weights.values())
            new_weights = {k: v / total for k, v in new_weights.items()}

            new_score = compute_preference(df, new_weights)
            new_rank = new_score.rank(ascending=False)

            kt, kp = kendalltau(base_rank, new_rank)
            sr, sp = spearmanr(base_rank, new_rank)

            rows.append({
                "parameter": param, "delta": delta,
                "old_weight": old_val, "new_weight": new_val,
                "kendall_tau": kt, "kendall_p": kp,
                "spearman_r": sr, "spearman_p": sp,
                "top10_overlap": len(set(base_score.nlargest(10).index) & set(new_score.nlargest(10).index)),
                "top20_overlap": len(set(base_score.nlargest(20).index) & set(new_score.nlargest(20).index)),
            })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "weight_sensitivity_single.csv", index=False)
    print(f"  [OK] Saved weight_sensitivity_single.csv ({len(result)} combinations)")
    return result


# ---------------------------------------------------------------------------
# 2. Grid Search over Main Weights (Return-Based Optimization)
# ---------------------------------------------------------------------------
def grid_search_weights(df):
    """Grid search over ESG, Financial, and Risk-Adjusted weights.

    Optimizes for ACTUAL portfolio Sharpe ratio (using price_momentum_3m as return proxy)
    rather than score mean/std. This provides empirically grounded weight selection.
    """
    print("\n--- Weight Sensitivity: Grid Search (Return-Based) ---")

    # Determine which return column to use
    return_col = None
    for rc in ["price_momentum_3m", "price_momentum_6m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break

    esg_range = np.arange(0.10, 0.35, 0.05)
    fin_range = np.arange(0.15, 0.40, 0.05)
    risk_range = np.arange(0.05, 0.20, 0.05)

    base_score = compute_preference(df, DEFAULT_WEIGHTS)
    base_rank = base_score.rank(ascending=False)

    rows = []
    for esg_w, fin_w, risk_w in product(esg_range, fin_range, risk_range):
        remaining = 1.0 - esg_w - fin_w - risk_w
        if remaining < 0.15:
            continue
        # Distribute remaining among other 7 factors proportionally
        other_base = {
            "market_score": 0.10, "operational_score": 0.10,
            "growth_score": 0.08, "value_score": 0.07,
            "stability_score": 0.05, "similarity_rank": 0.03, "sector_position": 0.02,
        }
        other_total = sum(other_base.values())
        weights = {
            "ESG_composite": esg_w, "financial_score": fin_w,
            "risk_adjusted_score": risk_w,
        }
        for k, v in other_base.items():
            weights[k] = remaining * (v / other_total)

        score = compute_preference(df, weights)
        rank = score.rank(ascending=False)
        kt, _ = kendalltau(base_rank, rank)

        # Score-based Sharpe (always available)
        score_sharpe = score.mean() / (score.std() + 1e-10)

        # Return-based Sharpe (uses actual stock returns)
        return_sharpe = 0.0
        if return_col:
            return_sharpe = compute_portfolio_sharpe(df, weights, return_col=return_col, top_n=20)

        rows.append({
            "esg_weight": round(esg_w, 2), "financial_weight": round(fin_w, 2),
            "risk_adj_weight": round(risk_w, 2),
            "market_weight": round(weights.get("market_score", 0), 3),
            "operational_weight": round(weights.get("operational_score", 0), 3),
            "mean_score": score.mean(), "std_score": score.std(),
            "score_sharpe": score_sharpe,
            "return_sharpe": return_sharpe,
            "kendall_vs_base": kt,
            "top10_overlap": len(set(base_score.nlargest(10).index) & set(score.nlargest(10).index)),
        })

    result = pd.DataFrame(rows)
    # Sort by return-based Sharpe (prefer actual performance) then score Sharpe
    if return_col and result["return_sharpe"].abs().sum() > 0:
        result = result.sort_values("return_sharpe", ascending=False)
        best_col = "return_sharpe"
    else:
        result = result.sort_values("score_sharpe", ascending=False)
        best_col = "score_sharpe"

    result.to_csv(TABLES / "weight_grid_search.csv", index=False)
    print(f"  [OK] Saved weight_grid_search.csv ({len(result)} combinations)")

    best = result.iloc[0]
    print(f"  Best: ESG={best['esg_weight']:.0%}, Financial={best['financial_weight']:.0%}, "
          f"RiskAdj={best['risk_adj_weight']:.0%}, {best_col}={best[best_col]:.3f}")
    return result


# ---------------------------------------------------------------------------
# 3. Rank Stability Analysis
# ---------------------------------------------------------------------------
def rank_stability(df):
    """Test how stable rankings are across different weight perturbations."""
    print("\n--- Weight Sensitivity: Rank Stability ---")

    np.random.seed(42)
    n_simulations = 100

    base_weights_arr = np.array([DEFAULT_WEIGHTS[k] for k in WEIGHT_NAMES if k in df.columns])
    avail_names = [k for k in WEIGHT_NAMES if k in df.columns]

    base_score = compute_preference(df, dict(zip(avail_names, base_weights_arr)))
    base_rank = base_score.rank(ascending=False)

    rank_matrix = np.zeros((len(df), n_simulations))
    for sim in range(n_simulations):
        noise = 1.0 + np.random.uniform(-0.2, 0.2, len(base_weights_arr))
        perturbed = base_weights_arr * noise
        perturbed = perturbed / perturbed.sum()
        perturbed_score = compute_preference(df, dict(zip(avail_names, perturbed)))
        rank_matrix[:, sim] = perturbed_score.rank(ascending=False).values

    rank_mean = rank_matrix.mean(axis=1)
    rank_std = rank_matrix.std(axis=1)
    rank_range = rank_matrix.max(axis=1) - rank_matrix.min(axis=1)

    stability = pd.DataFrame({
        "ticker": df["ticker"].values,
        "base_rank": base_rank.values,
        "avg_rank": rank_mean,
        "rank_std": rank_std,
        "rank_range": rank_range,
        "rank_stable": rank_std < 3,
    })
    stability = stability.sort_values("base_rank")
    stability.to_csv(TABLES / "rank_stability.csv", index=False)

    stable_pct = stability["rank_stable"].mean() * 100
    print(f"  [OK] Saved rank_stability.csv")
    print(f"  Stable companies (rank std < 3): {stable_pct:.1f}%")
    print(f"  Average rank std: {rank_std.mean():.2f}")
    return stability


# ---------------------------------------------------------------------------
# 4. Profile Comparison
# ---------------------------------------------------------------------------
def profile_comparison(df):
    """Compare rankings across investor profiles."""
    print("\n--- Weight Sensitivity: Profile Comparison ---")

    profiles = {
        "esg_first": {
            "ESG_composite": 0.35, "financial_score": 0.15,
            "market_score": 0.08, "operational_score": 0.10,
            "risk_adjusted_score": 0.08, "growth_score": 0.06,
            "value_score": 0.05, "stability_score": 0.05,
            "similarity_rank": 0.05, "sector_position": 0.03,
        },
        "balanced": DEFAULT_WEIGHTS.copy(),
        "financial_first": {
            "ESG_composite": 0.10, "financial_score": 0.30,
            "market_score": 0.10, "operational_score": 0.10,
            "risk_adjusted_score": 0.12, "growth_score": 0.10,
            "value_score": 0.08, "stability_score": 0.05,
            "similarity_rank": 0.03, "sector_position": 0.02,
        },
    }

    scores = {}
    for name, weights in profiles.items():
        scores[name] = compute_preference(df, weights)

    rows = []
    profile_names = list(profiles.keys())
    for i, p1 in enumerate(profile_names):
        for p2 in profile_names[i + 1:]:
            kt, kp = kendalltau(scores[p1].rank(), scores[p2].rank())
            sr, sp = spearmanr(scores[p1].rank(), scores[p2].rank())
            overlap_10 = len(set(scores[p1].nlargest(10).index) & set(scores[p2].nlargest(10).index))
            overlap_20 = len(set(scores[p1].nlargest(20).index) & set(scores[p2].nlargest(20).index))
            rows.append({
                "profile1": p1, "profile2": p2,
                "kendall_tau": kt, "spearman_r": sr,
                "top10_overlap": overlap_10, "top20_overlap": overlap_20,
            })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "weight_profile_comparison.csv", index=False)
    print(f"  [OK] Saved weight_profile_comparison.csv")
    return result


# ---------------------------------------------------------------------------
# 5. Conditional / Interdependent Weight Exploration
# ---------------------------------------------------------------------------
def conditional_weight_analysis(df):
    """Explore weight interdependencies: how changing one factor's weight
    should affect others, based on pairwise return-correlation structure.

    When ESG weight rises, financial weight drops (substitute relationship).
    When market volatility is high, stability weight should rise.
    This produces an interaction matrix for the research paper.
    """
    print("\n--- Weight Sensitivity: Conditional Interdependence ---")

    return_col = None
    for rc in ["price_momentum_3m", "price_momentum_6m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break

    # A. Pairwise factor correlation with returns
    score_cols = [c for c in WEIGHT_NAMES if c in df.columns]
    if return_col:
        corr_rows = []
        for c in score_cols:
            r = df[[c, return_col]].dropna()
            if len(r) > 10:
                corr = r[c].corr(r[return_col])
                corr_rows.append({"factor": c, "corr_with_return": corr})
        corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_return", ascending=False)
        corr_df.to_csv(TABLES / "weight_factor_return_corr.csv", index=False)
        print(f"  [OK] Saved weight_factor_return_corr.csv")
        for _, r in corr_df.iterrows():
            print(f"    {r['factor']:25s}: corr={r['corr_with_return']:+.3f}")

    # B. Pairwise factor substitution matrix
    # How rank correlation changes when one factor's weight is doubled
    sub_rows = []
    for f1 in score_cols:
        for f2 in score_cols:
            if f1 == f2:
                continue
            # Baseline: default weights
            base_score = compute_preference(df, DEFAULT_WEIGHTS)
            base_rank = base_score.rank(ascending=False)

            # Shift: double f1's weight, halve f2's weight
            shifted = DEFAULT_WEIGHTS.copy()
            shifted[f1] = min(0.50, shifted.get(f1, 0.10) * 2)
            shifted[f2] = max(0.01, shifted.get(f2, 0.10) * 0.5)
            total = sum(shifted.values())
            shifted = {k: v / total for k, v in shifted.items()}

            new_score = compute_preference(df, shifted)
            new_rank = new_score.rank(ascending=False)
            kt, _ = kendalltau(base_rank, new_rank)

            # Measure Sharpe change
            new_sharpe = 0.0
            if return_col:
                new_sharpe = compute_portfolio_sharpe(df, shifted, return_col=return_col, top_n=20)
            base_sharpe = compute_portfolio_sharpe(df, DEFAULT_WEIGHTS, return_col=return_col, top_n=20) if return_col else 0.0

            sub_rows.append({
                "increased_factor": f1, "decreased_factor": f2,
                "kendall_tau": kt,
                "sharpe_change": new_sharpe - base_sharpe,
            })

    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(TABLES / "weight_interdependence_matrix.csv", index=False)
    print(f"  [OK] Saved weight_interdependence_matrix.csv ({len(sub_df)} pairs)")

    # C. Identify best complementary pairs (largest Sharpe improvement)
    if len(sub_df) > 0:
        best = sub_df.nlargest(5, "sharpe_change")
        print("  Top 5 beneficial weight shifts:")
        for _, r in best.iterrows():
            print(f"    Increase {r['increased_factor']:20s} / Decrease {r['decreased_factor']:20s}: "
                  f"Sharpe change = {r['sharpe_change']:+.3f}")

    return sub_df


def main():
    print("=" * 70)
    print("STEP 05: WEIGHT SENSITIVITY & OPTIMIZATION")
    print("=" * 70)

    df = _prepare_df(load_data())
    sensitivity_single_param(df)
    grid_search_weights(df)
    rank_stability(df)
    profile_comparison(df)
    conditional_weight_analysis(df)

    print(f"\n[DONE] Weight sensitivity analysis complete. Results in {TABLES}/")
    print("Next: python scripts/06_benchmark_comparison.py")


if __name__ == "__main__":
    main()
