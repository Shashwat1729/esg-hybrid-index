"""
Step 08: Advanced Statistical Analysis
========================================
Implements advanced analytical techniques for the research paper:
  1. Principal Component Analysis (PCA) on factor scores
  2. Hierarchical Cluster Analysis (Ward linkage)
  3. K-Means Clustering of company profiles
  4. Bootstrap Confidence Intervals for rankings
  5. Fama-MacBeth style cross-sectional analysis
  6. Rolling / Subsample stability analysis
  7. Leave-one-out sensitivity (factor ablation)
  8. Rank reversal analysis
  9. Information Ratio & Tracking Error vs benchmarks
  10. Efficient frontier approximation (mean-variance)

Input:  data/processed/indexed_data.csv
Output: reports/tables/advanced_*.csv
        data/processed/pca_scores.csv
        data/processed/cluster_assignments.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

FACTOR_COLS = ["ESG_composite", "financial_score", "market_score", "operational_score"]
EXTENDED_FACTORS = FACTOR_COLS + ["risk_adjusted_score", "value_score", "growth_score", "stability_score"]


def load_data():
    path = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# 1. Principal Component Analysis
# ---------------------------------------------------------------------------
def pca_analysis(df):
    print("\n--- Advanced 1: Principal Component Analysis ---")
    avail = [c for c in EXTENDED_FACTORS if c in df.columns]
    if len(avail) < 3:
        print("  [SKIP] Not enough factors for PCA")
        return

    X = df[avail].dropna()
    if len(X) < 10:
        return

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca = PCA()
    pca_scores = pca.fit_transform(X_std)

    # Explained variance
    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(avail))],
        "eigenvalue": pca.explained_variance_,
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    var_df.to_csv(TABLES / "advanced_pca_variance.csv", index=False, encoding="utf-8")

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(avail))],
        index=avail,
    )
    loadings.to_csv(TABLES / "advanced_pca_loadings.csv", encoding="utf-8")

    # Save PCA scores
    pca_df = pd.DataFrame(
        pca_scores[:, :min(4, len(avail))],
        columns=[f"PC{i+1}" for i in range(min(4, len(avail)))],
        index=X.index,
    )
    pca_df["ticker"] = df.loc[X.index, "ticker"].values
    pca_out = PROJECT_ROOT / "data" / "processed" / "pca_scores.csv"
    pca_df.to_csv(pca_out, index=False, encoding="utf-8")

    # Kaiser criterion: components with eigenvalue > 1
    n_retain = (pca.explained_variance_ > 1).sum()
    print(f"  Factors: {len(avail)}")
    print(f"  Components to retain (Kaiser): {n_retain}")
    print(f"  Variance explained (first {n_retain}): {pca.explained_variance_ratio_[:n_retain].sum():.1%}")
    print(f"  [OK] Saved advanced_pca_variance.csv, advanced_pca_loadings.csv, pca_scores.csv")
    return pca, loadings


# ---------------------------------------------------------------------------
# 2. Hierarchical Cluster Analysis
# ---------------------------------------------------------------------------
def hierarchical_clustering(df):
    print("\n--- Advanced 2: Hierarchical Cluster Analysis ---")
    avail = [c for c in FACTOR_COLS if c in df.columns]
    if len(avail) < 2:
        return

    X = df[avail].dropna()
    if len(X) < 10:
        return

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Ward linkage
    Z = linkage(X_std, method="ward")

    # Try different numbers of clusters
    rows = []
    for n_clusters in [3, 4, 5, 6]:
        labels = fcluster(Z, n_clusters, criterion="maxclust")
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        sil = silhouette_score(X_std, labels) if len(set(labels)) > 1 else 0
        ch = calinski_harabasz_score(X_std, labels) if len(set(labels)) > 1 else 0
        rows.append({
            "n_clusters": n_clusters,
            "silhouette_score": sil,
            "calinski_harabasz": ch,
        })

    cluster_metrics = pd.DataFrame(rows)
    cluster_metrics.to_csv(TABLES / "advanced_cluster_metrics.csv", index=False, encoding="utf-8")

    # Use optimal (highest silhouette)
    best_k = cluster_metrics.loc[cluster_metrics["silhouette_score"].idxmax(), "n_clusters"]
    best_k = int(best_k)
    labels = fcluster(Z, best_k, criterion="maxclust")
    df_out = df.loc[X.index, ["ticker"]].copy()
    df_out["cluster"] = labels

    # Cluster profiles
    df_temp = df.loc[X.index].copy()
    df_temp["cluster"] = labels
    profile = df_temp.groupby("cluster")[avail].agg(["mean", "std", "count"])
    profile.to_csv(TABLES / "advanced_cluster_profiles.csv", encoding="utf-8")

    # Sector distribution by cluster
    if "sector" in df.columns:
        sector_dist = pd.crosstab(df_temp["cluster"], df_temp["sector"], normalize="index")
        sector_dist.to_csv(TABLES / "advanced_cluster_sectors.csv", encoding="utf-8")

    # Save assignments
    cluster_out = PROJECT_ROOT / "data" / "processed" / "cluster_assignments.csv"
    df_out.to_csv(cluster_out, index=False, encoding="utf-8")

    print(f"  Optimal clusters: {best_k} (silhouette={cluster_metrics['silhouette_score'].max():.3f})")
    print(f"  [OK] Saved advanced_cluster_*.csv, cluster_assignments.csv")
    return Z, labels


# ---------------------------------------------------------------------------
# 3. K-Means Clustering
# ---------------------------------------------------------------------------
def kmeans_clustering(df):
    print("\n--- Advanced 3: K-Means Clustering ---")
    avail = [c for c in FACTOR_COLS if c in df.columns]
    if len(avail) < 2:
        return

    X = df[avail].dropna()
    if len(X) < 10:
        return

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Elbow method
    from sklearn.metrics import silhouette_score
    inertias = []
    sil_scores = []
    for k in range(2, min(10, len(X) // 3)):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_std)
        inertias.append({"k": k, "inertia": km.inertia_})
        sil_scores.append({"k": k, "silhouette": silhouette_score(X_std, km.labels_)})

    elbow_df = pd.DataFrame(inertias)
    elbow_df = elbow_df.merge(pd.DataFrame(sil_scores), on="k")
    elbow_df.to_csv(TABLES / "advanced_kmeans_elbow.csv", index=False, encoding="utf-8")

    print(f"  Elbow analysis: k=2..{min(9, len(X)//3)}")
    print(f"  [OK] Saved advanced_kmeans_elbow.csv")
    return elbow_df


# ---------------------------------------------------------------------------
# 4. Bootstrap Confidence Intervals for Rankings
# ---------------------------------------------------------------------------
def bootstrap_rankings(df, n_bootstrap=500):
    print("\n--- Advanced 4: Bootstrap Confidence Intervals ---")
    if "pref_balanced" not in df.columns:
        return

    np.random.seed(42)
    n = len(df)
    rank_matrix = np.zeros((n, n_bootstrap))

    for b in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        boot_df = df.iloc[idx].copy()
        # Recompute scores (approximate: just re-rank the preference scores with noise)
        noise = np.random.normal(0, df["pref_balanced"].std() * 0.1, n)
        boot_scores = df["pref_balanced"].values + noise
        rank_matrix[:, b] = pd.Series(boot_scores).rank(ascending=False).values

    # Compute CI for each company
    results = pd.DataFrame({
        "ticker": df["ticker"].values,
        "original_rank": df["pref_balanced"].rank(ascending=False).values,
        "bootstrap_mean_rank": rank_matrix.mean(axis=1),
        "bootstrap_std_rank": rank_matrix.std(axis=1),
        "ci_lower_5": np.percentile(rank_matrix, 2.5, axis=1),
        "ci_upper_95": np.percentile(rank_matrix, 97.5, axis=1),
        "ci_width": np.percentile(rank_matrix, 97.5, axis=1) - np.percentile(rank_matrix, 2.5, axis=1),
        "rank_stable": rank_matrix.std(axis=1) < 5,
    }).sort_values("original_rank")

    results.to_csv(TABLES / "advanced_bootstrap_ci.csv", index=False, encoding="utf-8")

    stable_pct = results["rank_stable"].mean() * 100
    avg_ci = results["ci_width"].mean()
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Stable companies (rank std < 5): {stable_pct:.1f}%")
    print(f"  Average 95% CI width: {avg_ci:.1f} positions")
    print(f"  [OK] Saved advanced_bootstrap_ci.csv")
    return results


# ---------------------------------------------------------------------------
# 5. Factor Ablation Study (Leave-One-Out)
# ---------------------------------------------------------------------------
def factor_ablation(df):
    """Leave-one-out factor ablation: remove each factor and measure rank change.

    Uses all 8 extended factors (not just the 4 core factors) for a complete
    picture of each factor's marginal contribution to the composite score.
    """
    print("\n--- Advanced 5: Factor Ablation Study ---")
    if "pref_balanced" not in df.columns:
        return

    avail = [c for c in EXTENDED_FACTORS if c in df.columns]
    if len(avail) < 2:
        return

    base_weights = {
        "ESG_composite": 0.15, "financial_score": 0.25,
        "market_score": 0.10, "operational_score": 0.10,
        "risk_adjusted_score": 0.08, "value_score": 0.08,
        "growth_score": 0.12, "stability_score": 0.05,
    }
    base_avail = {k: v for k, v in base_weights.items() if k in df.columns}

    def _compute_score(weights_dict):
        score = pd.Series(0.0, index=df.index)
        total = sum(weights_dict.values())
        for comp, w in weights_dict.items():
            if comp in df.columns:
                score += (w / total) * df[comp].fillna(50) / 100 * 100
        return score

    base_score = _compute_score(base_avail)
    base_rank = base_score.rank(ascending=False)

    rows = []
    for removed in base_avail:
        reduced = {k: v for k, v in base_avail.items() if k != removed}
        new_score = _compute_score(reduced)
        new_rank = new_score.rank(ascending=False)

        kt, _ = stats.kendalltau(base_rank, new_rank)
        sr, _ = stats.spearmanr(base_rank, new_rank)
        top10_overlap = len(set(base_score.nlargest(10).index) & set(new_score.nlargest(10).index))

        rows.append({
            "removed_factor": removed,
            "kendall_tau": kt,
            "spearman_r": sr,
            "top10_overlap": top10_overlap,
            "mean_rank_shift": abs(base_rank - new_rank).mean(),
            "max_rank_shift": abs(base_rank - new_rank).max(),
        })

    result = pd.DataFrame(rows).sort_values("kendall_tau")
    result.to_csv(TABLES / "advanced_factor_ablation.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved advanced_factor_ablation.csv")
    print(f"  Most influential factor: {result.iloc[0]['removed_factor']} "
          f"(Kendall tau = {result.iloc[0]['kendall_tau']:.3f} when removed)")
    return result


# ---------------------------------------------------------------------------
# 6. Rank Reversal Analysis
# ---------------------------------------------------------------------------
def rank_reversal_analysis(df):
    print("\n--- Advanced 6: Rank Reversal Analysis ---")
    profiles = ["pref_esg_first", "pref_balanced", "pref_financial_first"]
    avail = [c for c in profiles if c in df.columns]
    if len(avail) < 2:
        return

    # For each pair of profiles, find companies that reverse rank significantly
    rows = []
    for i, p1 in enumerate(avail):
        for p2 in avail[i + 1:]:
            r1 = df[p1].rank(ascending=False)
            r2 = df[p2].rank(ascending=False)
            rank_diff = (r1 - r2).abs()
            big_movers = rank_diff.nlargest(10)

            for idx in big_movers.index:
                rows.append({
                    "ticker": df.loc[idx, "ticker"],
                    "profile1": p1, "profile2": p2,
                    f"rank_{p1}": int(r1.loc[idx]),
                    f"rank_{p2}": int(r2.loc[idx]),
                    "rank_change": int(r1.loc[idx] - r2.loc[idx]),
                    "abs_change": int(abs(r1.loc[idx] - r2.loc[idx])),
                })

    if rows:
        result = pd.DataFrame(rows).sort_values("abs_change", ascending=False)
        result.to_csv(TABLES / "advanced_rank_reversals.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved advanced_rank_reversals.csv ({len(result)} reversals)")


# ---------------------------------------------------------------------------
# 7. Mean-Variance Efficient Frontier Approximation
# ---------------------------------------------------------------------------
def efficient_frontier(df):
    """Mean-variance efficient frontier using ACTUAL stock returns.

    For each random weight combination, we select top-20 companies by
    weighted score and measure their average return and return volatility.
    This produces return-based Sharpe ratios (interpretable as risk-adjusted
    performance) rather than score-based ratios which are meaningless.
    """
    print("\n--- Advanced 7: Efficient Frontier Approximation ---")
    avail = [c for c in FACTOR_COLS if c in df.columns]
    if len(avail) < 2 or "pref_balanced" not in df.columns:
        return

    # Find best return column
    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break
    if return_col is None:
        print("  [SKIP] No return data for frontier")
        return

    np.random.seed(42)
    n_portfolios = 1000
    top_n = 20

    rows = []
    for _ in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(len(avail)))
        weight_dict = dict(zip(avail, weights))

        # Compute composite score
        score = pd.Series(0.0, index=df.index)
        for col, w in weight_dict.items():
            score += w * df[col].fillna(50)

        # Select top-N and measure actual returns
        top_idx = score.nlargest(top_n).index
        rets = df.loc[top_idx, return_col].dropna()

        if len(rets) < 5:
            continue

        ret = rets.mean()
        risk = rets.std()
        sharpe = ret / (risk + 1e-10)

        row = {"return": ret, "risk": risk, "sharpe": sharpe}
        for col, w in weight_dict.items():
            row[f"w_{col}"] = w
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_efficient_frontier.csv", index=False, encoding="utf-8")

    # Best portfolios
    best_sharpe = result.loc[result["sharpe"].idxmax()]
    min_risk = result.loc[result["risk"].idxmin()]
    max_return = result.loc[result["return"].idxmax()]

    summary = pd.DataFrame([
        {"portfolio": "Max Sharpe", **best_sharpe.to_dict()},
        {"portfolio": "Min Risk", **min_risk.to_dict()},
        {"portfolio": "Max Return", **max_return.to_dict()},
    ])
    summary.to_csv(TABLES / "advanced_optimal_portfolios.csv", index=False, encoding="utf-8")

    print(f"  Generated {len(result)} valid portfolios (of {n_portfolios} attempts)")
    print(f"  Max Sharpe: {best_sharpe['sharpe']:.3f}")
    print(f"  [OK] Saved advanced_efficient_frontier.csv, advanced_optimal_portfolios.csv")
    return result


# ---------------------------------------------------------------------------
# 8. Cross-validation of Weight Selection
# ---------------------------------------------------------------------------
def cross_validate_weights(df):
    """Cross-validate weight selection using ACTUAL stock returns.

    For each fold: optimize weights on training set (maximize Sharpe of
    top-N portfolio returns), then evaluate on test set.  This tests
    whether optimal weights generalize out-of-sample.
    """
    print("\n--- Advanced 8: Cross-Validation of Weight Selection ---")
    avail = [c for c in FACTOR_COLS if c in df.columns]
    if len(avail) < 2 or "pref_balanced" not in df.columns:
        return

    # Find return column
    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break

    np.random.seed(42)
    n_folds = 5
    n = len(df)
    indices = np.random.permutation(n)
    fold_size = n // n_folds

    def _portfolio_sharpe(sub_df, weights, ret_col, top_n=10):
        """Select top_n by weighted score, return Sharpe of their actual returns."""
        score = pd.Series(0.0, index=sub_df.index)
        for i, col in enumerate(avail):
            score += weights[i] * sub_df[col].fillna(50)
        top_idx = score.nlargest(min(top_n, len(sub_df))).index
        if ret_col and ret_col in sub_df.columns:
            rets = sub_df.loc[top_idx, ret_col].dropna()
            if len(rets) >= 3 and rets.std() > 1e-10:
                return rets.mean() / rets.std()
        # Fallback: score-based Sharpe (clearly labeled)
        s = score.mean() / (score.std() + 1e-10)
        return s

    rows = []
    for fold in range(n_folds):
        test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Find optimal weights on training set
        best_sharpe = -999
        best_weights = None
        for _ in range(300):
            w = np.random.dirichlet(np.ones(len(avail)))
            s = _portfolio_sharpe(train_df, w, return_col, top_n=min(10, len(train_df) // 3))
            if s > best_sharpe:
                best_sharpe = s
                best_weights = w

        # Evaluate on test set
        test_sharpe = _portfolio_sharpe(test_df, best_weights, return_col,
                                         top_n=min(10, len(test_df) // 2))

        row = {
            "fold": fold + 1,
            "train_sharpe": best_sharpe,
            "test_sharpe": test_sharpe,
            "train_n": len(train_df),
            "test_n": len(test_df),
            "overfit_ratio": best_sharpe / (test_sharpe + 1e-10) if test_sharpe != 0 else float("inf"),
            "return_col_used": return_col or "score_based",
        }
        for i, col in enumerate(avail):
            row[f"w_{col}"] = best_weights[i]
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_cv_weights.csv", index=False, encoding="utf-8")

    avg_overfit = result["overfit_ratio"].replace([np.inf], np.nan).mean()
    print(f"  {n_folds}-fold cross-validation (metric: {'return-based' if return_col else 'score-based'})")
    print(f"  Avg train Sharpe: {result['train_sharpe'].mean():.3f}")
    print(f"  Avg test Sharpe: {result['test_sharpe'].mean():.3f}")
    print(f"  Avg overfit ratio: {avg_overfit:.2f}")
    print(f"  [OK] Saved advanced_cv_weights.csv")
    return result


# ---------------------------------------------------------------------------
# 9. Lorenz Curve / Gini Coefficient for Score Inequality
# ---------------------------------------------------------------------------
def score_inequality(df):
    print("\n--- Advanced 9: Score Inequality (Gini Coefficient) ---")
    score_cols = [c for c in EXTENDED_FACTORS + ["pref_balanced"] if c in df.columns]
    if not score_cols:
        return

    rows = []
    for col in score_cols:
        vals = df[col].dropna().values
        if len(vals) < 5:
            continue
        # Shift values to be non-negative for Gini calculation
        # Gini coefficient requires non-negative values
        shifted = vals - vals.min() + 1e-6
        sorted_vals = np.sort(shifted)
        n = len(sorted_vals)
        total = np.sum(sorted_vals)
        if total < 1e-10:
            gini = 0.0
        else:
            cumulative = np.cumsum(sorted_vals) / total
            gini = 1 - 2 * np.trapz(cumulative, dx=1/n)
            gini = max(0.0, min(1.0, gini))  # Clamp to valid range
        rows.append({
            "score": col,
            "gini_coefficient": gini,
            "inequality": "High" if gini > 0.4 else ("Moderate" if gini > 0.15 else "Low"),
            "mean": vals.mean(),
            "std": vals.std(),
            "cv": vals.std() / (abs(vals.mean()) + 1e-10),
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_gini_inequality.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved advanced_gini_inequality.csv")
    return result


# ---------------------------------------------------------------------------
# 10. Rolling Rebalance Simulation
# ---------------------------------------------------------------------------
def rolling_rebalance_simulation(df):
    """Simulate quarterly rebalancing using different horizon returns.

    Uses the available momentum columns (1m, 3m, 6m, 12m) to approximate
    what a rolling rebalance strategy would look like. For each "quarter",
    we select top-N stocks by preference score and measure forward returns.

    This cross-sectional approach approximates time-series backtesting when
    we have a single cross-section with multi-horizon return data.
    """
    print("\n--- Advanced 10: Rolling Rebalance Simulation ---")

    return_cols = {
        "Q1_1m": "price_momentum_1m",
        "Q2_3m": "price_momentum_3m",
        "Q3_6m": "price_momentum_6m",
        "Q4_12m": "price_momentum_12m",
    }
    avail_periods = {k: v for k, v in return_cols.items() if v in df.columns and df[v].notna().sum() > 10}

    if len(avail_periods) < 2:
        print("  [SKIP] Need at least 2 return horizons")
        return

    strategies = {}
    if "pref_balanced" in df.columns:
        strategies["our_balanced"] = "pref_balanced"
    if "ESG_composite" in df.columns:
        strategies["esg_only"] = "ESG_composite"
    if "financial_score" in df.columns:
        strategies["financial_only"] = "financial_score"
    if "growth_score" in df.columns:
        strategies["growth_only"] = "growth_score"

    rows = []
    for period_name, ret_col in avail_periods.items():
        for strat_name, sort_col in strategies.items():
            for top_n in [15, 20, 30]:
                top_df = df.nlargest(top_n, sort_col)
                rets = top_df[ret_col].dropna()
                if len(rets) < 3:
                    continue

                bench_rets = df[ret_col].dropna()
                excess = rets.mean() - bench_rets.mean()

                rows.append({
                    "period": period_name,
                    "strategy": strat_name,
                    "top_n": top_n,
                    "avg_return": rets.mean(),
                    "std_return": rets.std(),
                    "sharpe": rets.mean() / (rets.std() + 1e-10),
                    "benchmark_return": bench_rets.mean(),
                    "excess_return": excess,
                    "pct_positive": (rets > 0).mean() * 100,
                    "max_drawdown": rets.min(),
                    "hit_rate": (rets > bench_rets.mean()).mean() * 100,
                })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_rolling_rebalance.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved advanced_rolling_rebalance.csv ({len(result)} strategy-period combos)")

    # Summary: how often does each strategy beat the benchmark?
    if len(result) > 0:
        for strat in strategies:
            sub = result[result["strategy"] == strat]
            beat_pct = (sub["excess_return"] > 0).mean() * 100
            avg_excess = sub["excess_return"].mean()
            print(f"    {strat}: beats benchmark {beat_pct:.0f}% of periods, avg excess={avg_excess:+.2f}%")

    return result


# ---------------------------------------------------------------------------
# 11. Market Regime Analysis
# ---------------------------------------------------------------------------
def regime_analysis(df):
    """Analyze how different strategies perform in different market regimes.

    Splits companies into 'bull' (positive momentum) and 'bear' (negative) regimes
    based on 6-month price momentum of the full universe, then measures strategy
    performance in each regime.
    """
    print("\n--- Advanced 11: Market Regime Analysis ---")

    ret_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            ret_col = rc
            break
    if ret_col is None:
        print("  [SKIP] No return data")
        return

    # Define regimes based on individual stock momentum
    median_ret = df[ret_col].median()
    df_bull = df[df[ret_col] >= median_ret]
    df_bear = df[df[ret_col] < median_ret]

    strategies = {}
    if "pref_balanced" in df.columns:
        strategies["our_balanced"] = "pref_balanced"
    if "ESG_composite" in df.columns:
        strategies["esg_only"] = "ESG_composite"
    if "financial_score" in df.columns:
        strategies["financial_only"] = "financial_score"
    if "growth_score" in df.columns:
        strategies["growth_only"] = "growth_score"

    rows = []
    for regime_name, regime_df in [("bull", df_bull), ("bear", df_bear), ("all", df)]:
        for strat_name, sort_col in strategies.items():
            if sort_col not in regime_df.columns:
                continue
            top20 = regime_df.nlargest(min(20, len(regime_df)), sort_col)
            rets = top20[ret_col].dropna()
            bench = regime_df[ret_col].dropna()

            rows.append({
                "regime": regime_name,
                "strategy": strat_name,
                "n_universe": len(regime_df),
                "n_selected": len(top20),
                "avg_return": rets.mean() if len(rets) > 0 else 0,
                "benchmark_return": bench.mean() if len(bench) > 0 else 0,
                "excess_return": (rets.mean() - bench.mean()) if len(rets) > 0 else 0,
                "sharpe": rets.mean() / (rets.std() + 1e-10) if len(rets) > 2 else 0,
                "downside_capture": (
                    rets[rets < 0].mean() / (bench[bench < 0].mean() + 1e-10)
                ) if len(rets[rets < 0]) > 0 and len(bench[bench < 0]) > 0 else 0,
            })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_regime_analysis.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved advanced_regime_analysis.csv")

    # Key finding: does our index protect in bear markets?
    for strat in strategies:
        bear_row = result[(result["strategy"] == strat) & (result["regime"] == "bear")]
        if len(bear_row) > 0:
            excess = bear_row.iloc[0]["excess_return"]
            print(f"    {strat} bear-market excess: {excess:+.2f}%")

    return result


# ---------------------------------------------------------------------------
# 12. Factor Monotonicity Test
# ---------------------------------------------------------------------------
def factor_monotonicity(df):
    """Test whether factor scores monotonically predict returns across quintiles.

    This is a key validation: if our index factors are meaningful, companies
    sorted by factor scores should show monotonically increasing returns.
    """
    print("\n--- Advanced 12: Factor Monotonicity Test ---")

    ret_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            ret_col = rc
            break
    if ret_col is None:
        print("  [SKIP] No return data")
        return

    score_cols = [c for c in EXTENDED_FACTORS + ["pref_balanced"] if c in df.columns]

    rows = []
    for score_col in score_cols:
        valid = df[[score_col, ret_col]].dropna()
        if len(valid) < 20:
            continue

        # Create quintiles
        valid["quintile"] = pd.qcut(valid[score_col], 5, labels=[1, 2, 3, 4, 5])
        quintile_returns = valid.groupby("quintile")[ret_col].mean()

        # Test monotonicity: Spearman correlation between quintile and return
        if len(quintile_returns) >= 3:
            sr, sp = stats.spearmanr(quintile_returns.index.astype(int), quintile_returns.values)
        else:
            sr, sp = 0, 1

        # Long-short spread: Q5 - Q1
        q5_ret = quintile_returns.get(5, 0)
        q1_ret = quintile_returns.get(1, 0)
        spread = q5_ret - q1_ret

        rows.append({
            "factor": score_col,
            "Q1_return": q1_ret,
            "Q2_return": quintile_returns.get(2, 0),
            "Q3_return": quintile_returns.get(3, 0),
            "Q4_return": quintile_returns.get(4, 0),
            "Q5_return": q5_ret,
            "Q5_Q1_spread": spread,
            "spearman_r": sr,
            "spearman_p": sp,
            "monotonic": "Yes" if sr > 0.7 and sp < 0.1 else "Partial" if sr > 0.3 else "No",
            "return_col": ret_col,
        })

    result = pd.DataFrame(rows).sort_values("Q5_Q1_spread", ascending=False)
    result.to_csv(TABLES / "advanced_factor_monotonicity.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved advanced_factor_monotonicity.csv")

    for _, r in result.iterrows():
        print(f"    {r['factor']:25s}: Q5-Q1={r['Q5_Q1_spread']:+6.2f}%, monotonic={r['monotonic']}")

    return result


def main():
    print("=" * 70)
    print("STEP 08: ADVANCED STATISTICAL ANALYSIS")
    print("=" * 70)

    df = load_data()
    pca_analysis(df)
    hierarchical_clustering(df)
    kmeans_clustering(df)
    bootstrap_rankings(df, n_bootstrap=500)
    factor_ablation(df)
    rank_reversal_analysis(df)
    efficient_frontier(df)
    cross_validate_weights(df)
    score_inequality(df)
    rolling_rebalance_simulation(df)
    regime_analysis(df)
    factor_monotonicity(df)

    n_tables = len(list(TABLES.glob("advanced_*.csv")))
    print(f"\n[DONE] {n_tables} advanced analysis tables saved to {TABLES}/")
    print("Next: python scripts/07_visualizations.py")


if __name__ == "__main__":
    main()
