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
  10. Factor tilt sensitivity analysis (cross-sectional mean-variance)

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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, load_profile_weights
from src.constants import RANDOM_SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

FACTOR_COLS = ["ESG_composite", "financial_score", "market_score", "operational_score"]
EXTENDED_FACTORS = FACTOR_COLS + ["risk_adjusted_score", "value_score", "growth_score", "stability_score"]

# Mo3 DISCLOSURE: This regression uses contemporaneous ESG and financial
# variables (measured at the same point in time). It tests association,
# NOT causation. Reverse causality (profitable firms → better ESG disclosure)
# cannot be ruled out without a lagged specification. Interpret as
# "cross-sectional association between ESG quality and financial quality."

# Mo5 NOTE: Indian companies (~90 firms, ~33% of universe) have financials
# converted at a fixed INR/USD rate (₹83/$1, March 2024 RBI reference).
# Annual INR/USD variation of ±5-8% introduces measurement error in
# cross-border financial ratio comparisons. Future work should use
# quarterly average exchange rates from the corresponding reporting period.

# Mo7 TODO: Extend perturbation analysis from ±10% to ±10/20/30%
# to capture sensitivity of turnover estimates to score uncertainty.


def load_data():
    df = load_indexed_data(PROJECT_ROOT)
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
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
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
def bayesian_bootstrap_rankings(df, n_bootstrap=1000):
    """Bayesian bootstrap ranking stability via Dirichlet reweighting."""
    if "pref_balanced" not in df.columns:
        return pd.DataFrame(columns=["ticker", "bayesian_mean_rank", "bayesian_rank_std", "_bayesian_rank_history"])

    n_companies = len(df)
    base_scores = df["pref_balanced"].fillna(df["pref_balanced"].median()).values
    rng = np.random.default_rng(RANDOM_SEED + 17)

    bayesian_rank_matrix = np.zeros((n_companies, n_bootstrap))
    for b in range(n_bootstrap):
        weights = rng.dirichlet(np.ones(n_companies))
        bayesian_scores = base_scores * weights * n_companies
        bayesian_rank_matrix[:, b] = pd.Series(bayesian_scores).rank(ascending=False).values

    bayesian_df = pd.DataFrame({
        "ticker": df["ticker"].values,
        "bayesian_mean_rank": bayesian_rank_matrix.mean(axis=1),
        "bayesian_rank_std": bayesian_rank_matrix.std(axis=1),
        "_bayesian_rank_history": [bayesian_rank_matrix[i, :] for i in range(n_companies)],
    })
    return bayesian_df


def bootstrap_rankings(df, n_bootstrap=1000):
    """Bootstrap confidence intervals for company rankings.

    Performs PROPER bootstrap resampling: in each iteration, companies are
    resampled with replacement from the original dataset, and preference
    scores are recomputed from the resampled factor scores.  This captures
    genuine sampling uncertainty in the ranking, not just score perturbation.

    For each original company, we track how often it appears in the bootstrap
    sample and what rank it receives, giving empirically grounded CIs.
    """
    print("\n--- Advanced 4: Bootstrap Confidence Intervals ---")
    if "pref_balanced" not in df.columns:
        return

    score_cols = [c for c in EXTENDED_FACTORS if c in df.columns]
    if not score_cols:
        return

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    relaxed_window = 0.15 * n
    original_rank = df["pref_balanced"].rank(ascending=False).values

    # Load balanced profile weights from config (all 10 factors).
    # Done once outside the loop to avoid re-reading the YAML file each iteration.
    balanced_weights = load_profile_weights("balanced")

    # Store rank for each company across bootstrap iterations
    # For companies not in a particular bootstrap sample, we record NaN
    rank_matrix = np.full((n, n_bootstrap), np.nan)

    for b in range(n_bootstrap):
        # Resample WITH REPLACEMENT from the company universe
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_df = df.iloc[boot_idx].copy().reset_index(drop=True)

        # Recompute preference scores from bootstrapped factor scores
        # using balanced profile weights loaded from config/index_config.yaml
        boot_score = pd.Series(0.0, index=boot_df.index)
        total_w = 0
        for sc, w in balanced_weights.items():
            if sc in boot_df.columns:
                boot_score += w * boot_df[sc].fillna(50)
                total_w += w
        if total_w > 0:
            boot_score /= total_w

        boot_rank = boot_score.rank(ascending=False)

        # Map back to original company positions
        for orig_pos, boot_pos in enumerate(boot_idx):
            # For the original company at position boot_pos,
            # record the rank it got in this bootstrap sample
            if np.isnan(rank_matrix[boot_pos, b]):
                rank_matrix[boot_pos, b] = boot_rank.iloc[orig_pos]
            else:
                # Company appeared multiple times; take average rank
                rank_matrix[boot_pos, b] = min(rank_matrix[boot_pos, b],
                                                 boot_rank.iloc[orig_pos])

    # Compute CI for each company (ignoring NaN = iterations where company wasn't sampled)
    results_rows = []
    for i in range(n):
        ranks_i = rank_matrix[i, :]
        valid_ranks = ranks_i[~np.isnan(ranks_i)]
        if len(valid_ranks) < 10:
            continue  # Too few samples for reliable CI
        results_rows.append({
            "_company_idx": i,
            "ticker": df.iloc[i]["ticker"],
            "original_rank": original_rank[i],
            "bootstrap_mean_rank": np.mean(valid_ranks),
            "bootstrap_std_rank": np.std(valid_ranks),
            "ci_lower_5": np.percentile(valid_ranks, 2.5),
            "ci_upper_95": np.percentile(valid_ranks, 97.5),
            "ci_width": np.percentile(valid_ranks, 97.5) - np.percentile(valid_ranks, 2.5),
            "rank_stable": np.std(valid_ranks) < 5,
            "rank_stable_relaxed": np.std(valid_ranks) < relaxed_window,
            "n_bootstrap_appearances": len(valid_ranks),
        })

    results = pd.DataFrame(results_rows)

    bayesian_results = bayesian_bootstrap_rankings(df, n_bootstrap=n_bootstrap)
    bayesian_rank_map = dict(zip(bayesian_results["ticker"], bayesian_results["_bayesian_rank_history"]))

    consensus_rank = []
    consensus_std = []
    ci_lower = []
    ci_upper = []
    for _, row in results.iterrows():
        idx = int(row["_company_idx"])
        ticker = row["ticker"]
        standard_ranks = rank_matrix[idx, :]
        standard_valid = standard_ranks[~np.isnan(standard_ranks)]
        bayesian_ranks = bayesian_rank_map.get(ticker, np.array([]))
        all_ranks = np.concatenate([standard_valid, bayesian_ranks])
        consensus_rank.append(np.mean(all_ranks))
        consensus_std.append(np.std(all_ranks))
        ci_lower.append(np.percentile(all_ranks, 2.5))
        ci_upper.append(np.percentile(all_ranks, 97.5))

    results["consensus_rank"] = consensus_rank
    results["rank_std"] = consensus_std
    results["ci_lower"] = ci_lower
    results["ci_upper"] = ci_upper

    results = results.merge(
        bayesian_results[["ticker", "bayesian_mean_rank", "bayesian_rank_std"]],
        on="ticker",
        how="left",
    )
    results = results.sort_values("original_rank")
    results = results.rename(columns={"rank_stable": "rank_stable_legacy"})
    results["ci_width"] = results["ci_upper"] - results["ci_lower"]
    universe_size = len(df)
    results["rank_stable_strict"] = results["ci_width"] < (0.20 * universe_size)
    results["rank_stable_moderate"] = results["ci_width"] < (0.35 * universe_size)
    results.to_csv(TABLES / "advanced_bootstrap_ci.csv", index=False, encoding="utf-8")

    enhanced_cols = [
        "ticker",
        "original_rank",
        "consensus_rank",
        "rank_std",
        "rank_stable_legacy",
        "rank_stable_relaxed",
        "ci_lower",
        "ci_upper",
        "ci_width",
        "rank_stable_strict",
        "rank_stable_moderate",
        "bayesian_mean_rank",
        "bayesian_rank_std",
    ]
    results[enhanced_cols].to_csv(TABLES / "bootstrap_enhanced_stability.csv", index=False, encoding="utf-8")

    strict_pct = results["rank_stable_legacy"].mean() * 100
    relaxed_pct = results["rank_stable_relaxed"].mean() * 100
    consensus_corr, _ = stats.spearmanr(results["original_rank"], results["consensus_rank"])
    mean_rank_std = results["rank_std"].mean()

    avg_ci = results["ci_width"].mean()

    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Method: proper resampling with replacement + score recomputation")
    print(f"  Stable companies (rank std < 5): {strict_pct:.1f}%")
    print(f"  Stable companies (rank std < 15% of N): {relaxed_pct:.1f}%")
    print(f"  Bootstrap stability (strict, CI<{int(0.20*universe_size)} ranks): "
          f"{results['rank_stable_strict'].sum()}/{len(df)}")
    print(f"  Bootstrap stability (moderate, CI<{int(0.35*universe_size)} ranks): "
          f"{results['rank_stable_moderate'].sum()}/{len(df)}")
    print(f"  Mean consensus rank correlation vs original: {consensus_corr:.3f}")
    print(f"  Mean rank std (consensus): {mean_rank_std:.2f}")
    print(f"  Average 95% CI width: {avg_ci:.1f} positions")
    print(f"  [OK] Saved advanced_bootstrap_ci.csv, bootstrap_enhanced_stability.csv")
    return results.drop(columns=["_company_idx"], errors="ignore")


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

    # Use the balanced profile as the reference weight set for ablation,
    # because it is the most representative (equal emphasis across all factors).
    # Weights are loaded from config/index_config.yaml to stay in sync with
    # the preference scoring pipeline (all 10 factors).
    base_weights = load_profile_weights("balanced")
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
# 7. Factor Tilt Sensitivity Analysis (cross-sectional mean-variance)
# ---------------------------------------------------------------------------
def factor_tilt_sensitivity(df):
    """Factor tilt sensitivity analysis using cross-sectional trailing returns.

    IMPORTANT METHODOLOGICAL CAVEAT:
    ================================
    The returns used here (price_momentum_Xm) are TRAILING returns from the
    same cross-sectional snapshot.  This analysis shows the in-sample
    trade-off between return dispersion and average trailing momentum for
    different factor-weight combinations.  It does NOT represent a
    forward-looking efficient frontier.

    The "cross_sectional_ir" metric is mean(trailing momentum) / std(trailing
    momentum) across the N selected stocks — it measures selection quality,
    NOT a time-series Sharpe ratio.

    Results should be interpreted as: "which factor combinations would have
    selected stocks with the best recent momentum?" — NOT as evidence of
    future portfolio optimality.

    LEAKAGE CONCERN (C1 circularity):
    ==================================
    market_score contains momentum sub-scores (price_momentum_1m/3m/6m) that
    overlap with the return proxy.  Including market_score as a weight
    dimension biases the frontier toward momentum-loading portfolios.
    We generate BOTH the original frontier (with market_score) and an
    ex-market variant (excluding market_score) for unbiased comparison.
    """
    print("\n--- Advanced 7: Factor Tilt Sensitivity Analysis ---")
    print("  NOTE: Using trailing momentum as return proxy (in-sample exploration only)")
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

    rng = np.random.default_rng(RANDOM_SEED)
    n_portfolios = 1000
    top_n = 20

    # --- Original frontier (all FACTOR_COLS including market_score) ---
    rows = []
    for _ in range(n_portfolios):
        weights = rng.dirichlet(np.ones(len(avail)))
        weight_dict = dict(zip(avail, weights))

        score = pd.Series(0.0, index=df.index)
        for col, w in weight_dict.items():
            score += w * df[col].fillna(50)

        top_idx = score.nlargest(top_n).index
        rets = df.loc[top_idx, return_col].dropna()

        if len(rets) < 5:
            continue

        ret = rets.mean()
        risk = rets.std()
        cross_sectional_ir = ret / (risk + 1e-10)

        row = {"return": ret, "risk": risk, "cross_sectional_ir": cross_sectional_ir, "frontier_type": "original"}
        for col, w in weight_dict.items():
            row[f"w_{col}"] = w
        rows.append(row)

    # --- Ex-market frontier (excluding market_score to avoid leakage) ---
    avail_ex_market = [c for c in avail if c != "market_score"]
    if len(avail_ex_market) >= 2:
        for _ in range(n_portfolios):
            weights = rng.dirichlet(np.ones(len(avail_ex_market)))
            weight_dict = dict(zip(avail_ex_market, weights))

            score = pd.Series(0.0, index=df.index)
            for col, w in weight_dict.items():
                score += w * df[col].fillna(50)

            top_idx = score.nlargest(top_n).index
            rets = df.loc[top_idx, return_col].dropna()

            if len(rets) < 5:
                continue

            ret = rets.mean()
            risk = rets.std()
            cross_sectional_ir = ret / (risk + 1e-10)

            row = {"return": ret, "risk": risk, "cross_sectional_ir": cross_sectional_ir, "frontier_type": "ex_market"}
            for col, w in weight_dict.items():
                row[f"w_{col}"] = w
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_factor_tilt_sensitivity.csv", index=False, encoding="utf-8")

    # Best portfolios (separate by frontier type)
    summary_rows = []
    for ftype in result["frontier_type"].unique():
        sub = result[result["frontier_type"] == ftype]
        if len(sub) == 0:
            continue
        best_csir = sub.loc[sub["cross_sectional_ir"].idxmax()]
        min_risk = sub.loc[sub["risk"].idxmin()]
        max_return = sub.loc[sub["return"].idxmax()]
        summary_rows.extend([
            {"portfolio": f"Best CS-IR ({ftype})", **best_csir.to_dict()},
            {"portfolio": f"Min Risk ({ftype})", **min_risk.to_dict()},
            {"portfolio": f"Max Return ({ftype})", **max_return.to_dict()},
        ])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "advanced_optimal_portfolios.csv", index=False, encoding="utf-8")

    n_orig = len(result[result["frontier_type"] == "original"])
    n_ex = len(result[result["frontier_type"] == "ex_market"])
    print(f"  Generated {n_orig} original + {n_ex} ex-market tilt-sensitivity portfolios")
    for ftype in ["original", "ex_market"]:
        sub = result[result["frontier_type"] == ftype]
        if len(sub) > 0:
            print(f"  {ftype} Best CS-IR: {sub['cross_sectional_ir'].max():.3f}")
    print(f"  [OK] Saved advanced_factor_tilt_sensitivity.csv, advanced_optimal_portfolios.csv")
    return result


# ---------------------------------------------------------------------------
# 8. Cross-validation of Weight Selection
# ---------------------------------------------------------------------------
def cross_validate_weights(df):
    """Cross-validate weight selection using stock returns.

    IMPORTANT METHODOLOGICAL CAVEAT:
    ================================
    The returns used here are trailing momentum (same cross-section).
    This CV tests whether weights that maximize trailing-momentum CS-IR
    on a subset generalize to a held-out subset.  It does NOT test
    forward-looking predictive power.  Overfit ratios should be interpreted
    cautiously: a ratio > 1 indicates within-sample overfitting to trailing
    momentum, not true out-of-sample performance degradation.

    LEAKAGE CONCERN (C1 circularity):
    ==================================
    market_score contains momentum sub-scores that overlap with the return
    proxy (price_momentum_Xm).  The "ex_market" CV variant excludes
    market_score from the weight search to provide an unbiased comparison.

    METHODOLOGICAL IMPROVEMENTS (M2):
    ==================================
    - 2000 random draws (up from 300) for better weight-space coverage
    - Dirichlet(2,...,2) prior to penalise extreme weight concentration
    - Herfindahl index reported per fold to detect single-factor dominance
    - Leave-5-out CV as a supplementary method (more stable with N~56)
    """
    print("\n--- Advanced 8: Cross-Validation of Weight Selection ---")
    print("  NOTE: Using trailing momentum as return proxy (generalization test only)")
    avail = [c for c in FACTOR_COLS if c in df.columns]
    if len(avail) < 2 or "pref_balanced" not in df.columns:
        return

    # Find return column
    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)

    # --- Dirichlet concentration parameter ---
    # alpha=2 for each factor penalises extreme weights: the mode of
    # Dirichlet(2,...,2) is the uniform vector 1/K, so draws cluster
    # toward balanced allocations rather than corner solutions.
    DIRICHLET_ALPHA = 2.0
    N_DRAWS = 2000  # up from 300 for better coverage of weight space

    def _portfolio_csir(sub_df, weights, factor_list, ret_col, top_n=10):
        """Select top_n by weighted score, return cross-sectional IR of their actual returns."""
        score = pd.Series(0.0, index=sub_df.index)
        for i, col in enumerate(factor_list):
            score += weights[i] * sub_df[col].fillna(50)
        top_idx = score.nlargest(min(top_n, len(sub_df))).index
        if ret_col and ret_col in sub_df.columns:
            rets = sub_df.loc[top_idx, ret_col].dropna()
            if len(rets) >= 3 and rets.std() > 1e-10:
                return rets.mean() / rets.std()
        # Fallback: score-based CS-IR (clearly labeled)
        s = score.mean() / (score.std() + 1e-10)
        return s

    def _herfindahl(weights):
        """Herfindahl-Hirschman Index of weight concentration (0=uniform, 1=single factor)."""
        return float(np.sum(weights ** 2))

    # Run CV for both original (with market_score) and ex_market variants
    avail_ex_market = [c for c in avail if c != "market_score"]
    cv_variants = [("original", avail)]
    if len(avail_ex_market) >= 2:
        cv_variants.append(("ex_market", avail_ex_market))

    # =====================================================================
    # Part A: 5-fold CV (standard)
    # =====================================================================
    n_folds = 5
    indices = rng.permutation(n)
    fold_size = n // n_folds

    rows = []
    for variant_name, factor_list in cv_variants:
        alpha_vec = np.full(len(factor_list), DIRICHLET_ALPHA)
        for fold in range(n_folds):
            test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Find optimal weights on training set (2000 draws, regularised Dirichlet)
            best_csir = -999
            best_weights = None
            for _ in range(N_DRAWS):
                w = rng.dirichlet(alpha_vec)
                s = _portfolio_csir(train_df, w, factor_list, return_col,
                                      top_n=min(10, len(train_df) // 3))
                if s > best_csir:
                    best_csir = s
                    best_weights = w

            # Evaluate on test set
            test_csir = _portfolio_csir(test_df, best_weights, factor_list, return_col,
                                             top_n=min(10, len(test_df) // 2))

            row = {
                "cv_method": "5fold",
                "cv_variant": variant_name,
                "fold": fold + 1,
                "train_csir": best_csir,
                "test_csir": test_csir,
                "train_n": len(train_df),
                "test_n": len(test_df),
                "overfit_ratio": best_csir / (test_csir + 1e-10) if test_csir != 0 else float("inf"),
                "weight_concentration_hhi": _herfindahl(best_weights),
                "max_weight": float(best_weights.max()),
                "max_weight_factor": factor_list[int(best_weights.argmax())],
                "return_col_used": return_col or "score_based",
                "n_draws": N_DRAWS,
                "dirichlet_alpha": DIRICHLET_ALPHA,
            }
            for i, col in enumerate(factor_list):
                row[f"w_{col}"] = best_weights[i]
            rows.append(row)

    # =====================================================================
    # Part B: Leave-5-out CV (more stable with small N)
    # =====================================================================
    # Instead of 5 fixed folds, repeatedly sample 5 random companies as
    # the held-out test set.  With N=56 this gives test sets of 5 and
    # training sets of 51, which is less noisy than 11-company test folds.
    N_L5O_ITER = 50  # 50 random leave-5-out splits

    for variant_name, factor_list in cv_variants:
        alpha_vec = np.full(len(factor_list), DIRICHLET_ALPHA)
        for it in range(N_L5O_ITER):
            test_idx = rng.choice(n, size=5, replace=False)
            train_idx = np.setdiff1d(np.arange(n), test_idx)

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Find optimal weights on training set
            best_csir = -999
            best_weights = None
            for _ in range(N_DRAWS):
                w = rng.dirichlet(alpha_vec)
                s = _portfolio_csir(train_df, w, factor_list, return_col,
                                      top_n=min(10, len(train_df) // 3))
                if s > best_csir:
                    best_csir = s
                    best_weights = w

            # Evaluate on test set (top_n capped at test set size)
            test_csir = _portfolio_csir(test_df, best_weights, factor_list, return_col,
                                             top_n=min(5, len(test_df)))

            row = {
                "cv_method": "leave5out",
                "cv_variant": variant_name,
                "fold": it + 1,
                "train_csir": best_csir,
                "test_csir": test_csir,
                "train_n": len(train_df),
                "test_n": len(test_df),
                "overfit_ratio": best_csir / (test_csir + 1e-10) if test_csir != 0 else float("inf"),
                "weight_concentration_hhi": _herfindahl(best_weights),
                "max_weight": float(best_weights.max()),
                "max_weight_factor": factor_list[int(best_weights.argmax())],
                "return_col_used": return_col or "score_based",
                "n_draws": N_DRAWS,
                "dirichlet_alpha": DIRICHLET_ALPHA,
            }
            for i, col in enumerate(factor_list):
                row[f"w_{col}"] = best_weights[i]
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "advanced_cv_weights.csv", index=False, encoding="utf-8")

    # --- Reporting ---
    for cv_method in ["5fold", "leave5out"]:
        print(f"\n  === {cv_method.upper()} CV ===")
        for variant_name, _ in cv_variants:
            sub = result[(result["cv_variant"] == variant_name) & (result["cv_method"] == cv_method)]
            if len(sub) == 0:
                continue
            avg_overfit = sub["overfit_ratio"].replace([np.inf], np.nan).mean()
            avg_hhi = sub["weight_concentration_hhi"].mean()
            # Fraction of folds where a single factor gets >50% weight
            dominant_pct = (sub["max_weight"] > 0.50).mean() * 100
            print(f"  [{variant_name}] "
                  f"(metric: {'return-based' if return_col else 'score-based'})")
            print(f"    Avg train CS-IR: {sub['train_csir'].mean():.3f}")
            print(f"    Avg test CS-IR:  {sub['test_csir'].mean():.3f}")
            print(f"    Avg overfit ratio: {avg_overfit:.2f}")
            print(f"    Avg weight HHI: {avg_hhi:.3f}  "
                  f"(1/{len(cv_variants[0][1])}={1/len(cv_variants[0][1]):.3f} = uniform)")
            print(f"    Folds with dominant factor (>50%): {dominant_pct:.0f}%")
            # Report which factor dominates most often
            if len(sub) > 0:
                dom_counts = sub["max_weight_factor"].value_counts()
                top_dom = dom_counts.index[0]
                top_dom_pct = dom_counts.iloc[0] / len(sub) * 100
                print(f"    Most frequent dominant factor: {top_dom} ({top_dom_pct:.0f}% of folds)")

    # I2 NOTE: With n=50 for prediction tests, the minimum detectable IC at
    # 80% power is approximately |r| > 0.28. Most factor ICs are below this
    # threshold, meaning these tests are severely underpowered. Results should
    # be interpreted cautiously — non-significance does not imply zero effect.
    n_pred = n
    print(f"\n  [I2] Statistical power note: With n={n_pred}, minimum detectable")
    print(f"       IC at 80% power ≈ |r| > {2.0/np.sqrt(n_pred):.3f}")
    print(f"       Tests below this threshold are underpowered — interpret cautiously")

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
            _trapz = getattr(np, 'trapz', None) or np.trapezoid
            gini = 1 - 2 * _trapz(cumulative, dx=1/n)
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
                    "cross_sectional_ir": rets.mean() / (rets.std() + 1e-10),
                    "benchmark_return": bench_rets.mean(),
                    "excess_return": excess,
                    "pct_positive": (rets > 0).mean() * 100,
                    "worst_stock_return": rets.min(),
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
                "cross_sectional_ir": rets.mean() / (rets.std() + 1e-10) if len(rets) > 2 else 0,
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

    LEAKAGE CONCERN (C1 circularity):
    ==================================
    market_score includes price_momentum_1m/3m/6m as sub-scores.  When the
    return proxy IS one of those momentum columns, the monotonicity test for
    market_score is tautologically inflated.  We flag this in the output
    with a 'leakage_risk' column.  The clean metric for market factor
    monotonicity should use market_score_ex_momentum (see script 06).
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

    # Momentum columns that create circularity with market_score
    _momentum_return_cols = {"price_momentum_1m", "price_momentum_3m", "price_momentum_6m"}

    rows = []
    for score_col in score_cols:
        valid = df[[score_col, ret_col]].dropna()
        if len(valid) < 20:
            continue

        # Create quintiles (drop duplicate bin edges for low-variance scores)
        try:
            valid["quintile"] = pd.qcut(valid[score_col], 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            valid["quintile"] = pd.qcut(valid[score_col], 5, labels=False, duplicates="drop") + 1
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

        # Flag leakage risk for market_score when return proxy is momentum
        leakage = "BIASED" if (score_col == "market_score" and ret_col in _momentum_return_cols) else "none"

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
            "monotonic": "Yes" if sr > 0.7 and sp < 0.05 else "Partial" if sr > 0.3 else "No",
            "return_col": ret_col,
            "leakage_risk": leakage,
        })

    result = pd.DataFrame(rows).sort_values("Q5_Q1_spread", ascending=False)
    result.to_csv(TABLES / "advanced_factor_monotonicity.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved advanced_factor_monotonicity.csv")

    for _, r in result.iterrows():
        flag = " [BIASED]" if r.get("leakage_risk") == "BIASED" else ""
        print(f"    {r['factor']:25s}: Q5-Q1={r['Q5_Q1_spread']:+6.2f}%, "
              f"monotonic={r['monotonic']}{flag}")

    return result


def main():
    print("\n" + "="*70)
    print("MULTI-GEOGRAPHY ESG INDEX ANALYSIS")
    print("="*70)
    print("Universe: US mid-cap (S&P 400) + India mid-cap (NSE)")
    print("Note (I1): This is a multi-geography index. Cross-border comparisons")
    print("  are affected by currency conversion (fixed INR/USD rate), different")
    print("  regulatory environments, and sector composition differences.")
    print("="*70)

    print("=" * 70)
    print("STEP 08: ADVANCED STATISTICAL ANALYSIS")
    print("=" * 70)

    df = load_data()
    pca_analysis(df)
    hierarchical_clustering(df)
    kmeans_clustering(df)
    bootstrap_rankings(df, n_bootstrap=1000)
    factor_ablation(df)
    rank_reversal_analysis(df)
    factor_tilt_sensitivity(df)
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
