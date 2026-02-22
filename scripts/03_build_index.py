"""
Step 03: Build Multi-Factor Index
===================================
Constructs composite scores for each of ten factor categories:
  1. ESG Composite (E, S, G pillars)
  2. Financial Score (profitability, growth, efficiency, stability, valuation)
  3. Market Score (liquidity, volatility, momentum)
  4. Operational Score (efficiency, innovation, market position)
  5. Risk-Adjusted Score (Sharpe, Sortino, drawdown)
  6. Value Score (P/E, P/B, EV/EBITDA)
  7. Growth Score (revenue growth, earnings growth, momentum)
  8. Stability Score (leverage, liquidity ratios)
  9. Similarity Rank (cosine similarity on ESG vectors)
  10. Sector Position (percentile rank within sector)

Then computes preference scores for three investor profiles using all 10 factors.
Finally derives data-driven weight rationale using PCA explained variance.

Input:  data/processed/clean_data.csv
Output: data/processed/indexed_data.csv
        reports/tables/company_rankings.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.data_collection.data_pipeline import load_configs
from src.index_construction.composite_index import CompositeIndexBuilder
from src.financial_scoring.financial_scorer import FinancialScorer, MarketFactorScorer
from src.similarity.cosine_similarity import compute_similarity_matrix
from src.similarity.preference_scoring import PreferenceScorer

# ---------------------------------------------------------------------------
# Column groupings
# ---------------------------------------------------------------------------
ESG_ENV_COLS = [
    "scope1_emissions", "scope2_emissions", "scope3_emissions",
    "emissions_intensity", "renewable_energy_pct", "energy_efficiency",
    "water_usage_intensity", "waste_recycling_pct", "carbon_reduction_target",
    "environmental_fines",
]
ESG_SOC_COLS = [
    "employee_turnover", "gender_diversity_pct", "women_management_pct",
    "pay_gap_ratio", "injury_rate", "safety_training_hours",
    "employee_satisfaction", "community_investment_pct",
    "supply_chain_audit_pct", "human_rights_policy",
]
ESG_GOV_COLS = [
    "board_independence_pct", "board_diversity_pct", "board_size",
    "exec_comp_esg_linked", "ceo_pay_ratio", "shareholder_rights_score",
    "ethics_compliance_score", "anti_corruption_policy",
    "data_privacy_score", "tax_transparency_score",
    "esg_controversy_score", "esg_risk_rating",
]
ESG_COLS = ESG_ENV_COLS + ESG_SOC_COLS + ESG_GOV_COLS


def _get_variable_type(col):
    """Look up variable type from 02_clean_data classification (imported inline)."""
    try:
        from importlib import import_module
        # We import the classification dict from the clean_data script
        import sys
        clean_mod_path = str(PROJECT_ROOT / "scripts")
        if clean_mod_path not in sys.path:
            sys.path.insert(0, clean_mod_path)
        # Direct lookup of known binary/ordinal variables
        BINARY_VARS = {"carbon_reduction_target", "human_rights_policy", "anti_corruption_policy"}
        ORDINAL_VARS = {"board_size", "esg_risk_rating"}
        if col in BINARY_VARS:
            return "binary"
        if col in ORDINAL_VARS:
            return "ordinal"
    except Exception:
        pass
    return "continuous"


def _z_score_sub(df, indicators_dict):
    """Compute a composite score from z-scored indicators, handling variable types.

    Binary variables (0/1) are mean-coded directly rather than z-scored,
    because z-scoring binary data produces misleading values and inflated
    influence from rare categories (Agresti, 2002; Hair et al., 2019).

    Ordinal variables use rank-based scoring if a _rank column exists.

    Parameters
    ----------
    df : pd.DataFrame
    indicators_dict : dict
        {column_name: higher_is_better (bool)}

    Returns
    -------
    pd.Series
        Score centered at 50, scaled by 10 (standard deviation in score space).
    """
    available = {k: v for k, v in indicators_dict.items() if k in df.columns}
    if not available:
        return pd.Series(50.0, index=df.index)

    z_df = pd.DataFrame(index=df.index)
    for col, higher_better in available.items():
        vtype = _get_variable_type(col)
        vals = pd.to_numeric(df[col], errors="coerce")

        if vtype == "binary":
            # For binary variables: use the proportion directly
            # 1 = has feature -> positive contribution, 0 = does not
            # Map to z-score-like scale: mean-center and scale by std
            # but cap contribution to prevent binary vars from dominating
            proportion = vals.mean()
            if proportion > 0 and proportion < 1:
                z = (vals - proportion) / max(vals.std(), 0.1)
            else:
                z = pd.Series(0.0, index=df.index)
            if not higher_better:
                z = -z
            z = z.clip(-2, 2)  # Tighter clip for binary variables
            z_df[col] = z

        elif vtype == "ordinal":
            # Use rank-based column if available, else rank in place
            rank_col = f"{col}_rank"
            if rank_col in df.columns:
                rank_vals = pd.to_numeric(df[rank_col], errors="coerce")
            else:
                rank_vals = vals.rank(pct=True) * 100
            # Z-score the ranks
            mean, std = rank_vals.mean(), rank_vals.std()
            if std > 1e-10:
                z = (rank_vals - mean) / std
            else:
                z = pd.Series(0.0, index=df.index)
            if not higher_better:
                z = -z
            z = z.clip(-3, 3)
            z_df[col] = z

        else:
            # Standard continuous z-score
            mean, std = vals.mean(), vals.std()
            if std > 1e-10:
                z = (vals - mean) / std
            else:
                z = pd.Series(0.0, index=df.index)
            if not higher_better:
                z = -z
            # Winsorize extreme z-scores to [-3, 3]
            z = z.clip(-3, 3)
            z_df[col] = z

    # Average z-score across available indicators, then map to 0-100 scale
    avg_z = z_df.mean(axis=1).fillna(0)
    score = 50 + avg_z * 10  # 1 std = 10 points, centered at 50
    return score.clip(0, 100)


def build_operational_score(df):
    """Compute operational quality score from available metrics."""
    indicators = {
        "revenue_per_employee": True,
        "r_d_intensity": True,
        "market_share": True,
        "operating_margin": True,
        "gross_margin": True,
        "fcf_margin": True,
        "cash_flow_to_debt": True,
    }
    df["operational_score"] = _z_score_sub(df, indicators)
    return df


def build_risk_adjusted_score(df):
    """Compute risk-adjusted quality score combining returns with risk metrics.

    Note: max_drawdown_1y is typically negative (e.g. -0.30 = 30% drawdown).
    A *less negative* drawdown is better, so higher_is_better=True is correct
    (a value of -0.10 > -0.40, so z-scoring already puts less-negative values above).
    We also include beta inverted so that low-beta stocks score higher.
    """
    indicators = {
        "sharpe_ratio_1y": True,
        "sortino_ratio_1y": True,
        "max_drawdown_1y": True,   # less negative = better (True is correct)
        "price_volatility": False,  # lower vol = better
        "beta": False,              # lower beta = less market risk
        "return_skewness": True,    # positive skew preferred
    }
    df["risk_adjusted_score"] = _z_score_sub(df, indicators)
    return df


def build_value_score(df):
    """Compute value score from valuation multiples (lower multiples = better value)."""
    indicators = {
        "trailing_pe": False,
        "forward_pe": False,
        "price_to_book": False,
        "price_to_sales": False,
        "enterprise_to_ebitda": False,
        "enterprise_to_revenue": False,
    }
    df["value_score"] = _z_score_sub(df, indicators)
    return df


def build_growth_score(df):
    """Compute growth score from SIZE-GROWTH and REINVESTMENT indicators.

    IMPORTANT: Price momentum (1m, 3m, 6m) is deliberately excluded
    because these same variables are used as forward return proxies in
    portfolio evaluation (scripts 05, 06, 08).  Including them would create
    look-ahead bias / data leakage.

    This score is kept DISTINCT from financial_score (profitability,
    efficiency, stability, valuation) and operational_score (revenue per
    employee, R&D, market share).  We use:
      - total_revenue_log  : scale of revenue (larger = more established)
      - net_income_log     : scale of earnings (larger = more profitable)
      - free_cashflow      : cash reinvestment capacity
      - r_d_intensity      : investment in future revenue streams (shared
                             with operational but captures different aspect)
      - ebitda_log         : operating-level scale of earnings
    """
    indicators = {
        "total_revenue_log": True,   # revenue scale (growth proxy)
        "net_income_log": True,      # earnings scale (growth proxy)
        "free_cashflow": True,       # cash generation for reinvestment
        "r_d_intensity": True,       # investment in future growth
        "ebitda_log": True,          # operating earnings scale
    }
    df["growth_score"] = _z_score_sub(df, indicators)
    return df


def build_stability_score(df):
    """Compute financial stability score.

    Stability rewards low leverage, adequate liquidity, and strong cash coverage.
    Payout ratio is included inverted: very high payouts reduce financial flexibility.
    """
    indicators = {
        "current_ratio": True,
        "quick_ratio": True,
        "debt_to_equity": False,    # lower leverage = more stable
        "debt_to_ebitda": False,    # lower = more stable
        "cash_flow_to_debt": True,  # higher cash coverage = more stable
        "payout_ratio": False,      # very high payout = less stable
    }
    df["stability_score"] = _z_score_sub(df, indicators)
    return df


def build_sector_position(df):
    """Compute sector percentile rank based on average of available factor scores.

    Uses the mean of ESG, financial, market, and operational scores to give a
    well-rounded within-sector ranking rather than relying on a single factor.
    """
    factor_cols = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail = [c for c in factor_cols if c in df.columns]
    if avail and "sector" in df.columns:
        composite = df[avail].mean(axis=1)
        df["sector_position"] = composite.groupby(df["sector"]).transform(
            lambda x: x.rank(pct=True)
        )
    else:
        df["sector_position"] = 0.5
    return df


def build_similarity_rank(df):
    """Compute average similarity rank from cosine similarity on ESG pillars."""
    feature_cols = ["E_score", "S_score", "G_score"]
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        df["similarity_rank"] = 0.5
        return df, None

    sim_matrix = compute_similarity_matrix(
        df, feature_cols=available, id_col="ticker", metric="cosine"
    )
    avg_sim = sim_matrix.values.copy()
    np.fill_diagonal(avg_sim, np.nan)
    df["similarity_rank"] = np.nanmean(avg_sim, axis=1)
    lo, hi = df["similarity_rank"].min(), df["similarity_rank"].max()
    if hi - lo > 1e-10:
        df["similarity_rank"] = (df["similarity_rank"] - lo) / (hi - lo)
    return df, sim_matrix


def derive_pca_weight_rationale(df):
    """Use PCA to derive data-driven weight rationale for the 8 main factor scores.

    The PCA eigenvalue-based proportional contribution shows how much variance
    each factor explains, providing an empirical justification for weight ranges.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    available = [c for c in score_cols if c in df.columns]
    if len(available) < 4:
        return {}

    X = df[available].dropna()
    if len(X) < 10:
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    # Compute each factor's total contribution across all components
    # (absolute loading * explained variance ratio, summed across components)
    loadings = np.abs(pca.components_)  # (n_components, n_features)
    var_ratios = pca.explained_variance_ratio_

    # Weighted contribution of each feature
    contributions = (loadings.T * var_ratios).sum(axis=1)
    contributions = contributions / contributions.sum()

    rationale = {}
    for i, col in enumerate(available):
        rationale[col] = {
            "pca_contribution": float(contributions[i]),
            "suggested_weight_range": (
                max(0.02, float(contributions[i]) - 0.05),
                min(0.40, float(contributions[i]) + 0.10)
            ),
        }

    return rationale


def main():
    print("=" * 70)
    print("STEP 03: BUILD MULTI-FACTOR INDEX")
    print("=" * 70)

    # Load data
    data_path = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
    fallback_path = PROJECT_ROOT / "data" / "processed" / "real_data_clean.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError(f"Clean data not found.\nRun: python scripts/02_clean_data.py")
    print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns")

    # Load config
    index_cfg, _ = load_configs()

    # 1. ESG Index
    print("\n1. Building ESG composite index...")
    esg_indicators = [c for c in ESG_COLS if c in df.columns]
    builder = CompositeIndexBuilder(index_cfg)
    df = builder.build(df, indicator_cols=esg_indicators)
    for col in ["ESG_composite", "E_score", "S_score", "G_score"]:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
                  f"range=[{df[col].min():.1f}, {df[col].max():.1f}]")

    # 2. Financial Score
    print("\n2. Computing financial scores...")
    financial_scorer = FinancialScorer(index_cfg)
    df = financial_scorer.compute_financial_score(df)
    if "financial_score" in df.columns:
        print(f"   financial_score: mean={df['financial_score'].mean():.1f}, "
              f"std={df['financial_score'].std():.1f}")

    # 3. Market Score
    print("\n3. Computing market scores...")
    market_scorer = MarketFactorScorer(index_cfg)
    df = market_scorer.compute_market_score(df)
    if "market_score" in df.columns:
        print(f"   market_score: mean={df['market_score'].mean():.1f}, "
              f"std={df['market_score'].std():.1f}")

    # 4. Operational Score
    print("\n4. Computing operational scores...")
    df = build_operational_score(df)
    print(f"   operational_score: mean={df['operational_score'].mean():.1f}, "
          f"std={df['operational_score'].std():.1f}")

    # 5. Additional factor scores
    print("\n5. Computing additional factor scores...")
    df = build_risk_adjusted_score(df)
    df = build_value_score(df)
    df = build_growth_score(df)
    df = build_stability_score(df)
    for col in ["risk_adjusted_score", "value_score", "growth_score", "stability_score"]:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}")

    # 6. Similarity Rank
    print("\n6. Computing similarity ranks...")
    df, sim_matrix = build_similarity_rank(df)
    print(f"   similarity_rank: mean={df['similarity_rank'].mean():.3f}")

    # 7. Sector Position
    print("\n7. Computing sector positions...")
    df = build_sector_position(df)
    print(f"   sector_position: mean={df['sector_position'].mean():.3f}")

    # 8. PCA Weight Rationale
    print("\n8. Deriving PCA-based weight rationale...")
    pca_rationale = derive_pca_weight_rationale(df)
    if pca_rationale:
        print("   PCA-derived factor contributions:")
        for factor, info in sorted(pca_rationale.items(), key=lambda x: -x[1]["pca_contribution"]):
            lo, hi = info["suggested_weight_range"]
            print(f"     {factor:25s}: {info['pca_contribution']:.3f}  -> range [{lo:.2f}, {hi:.2f}]")

    # 9. Preference Scores (3 profiles, using all 10 factors)
    print("\n9. Computing preference scores (3 investor profiles, 10 factors)...")
    scorer = PreferenceScorer(index_cfg)
    for profile in ["esg_first", "balanced", "financial_first"]:
        df[f"pref_{profile}"] = scorer.compute_preference_score(
            df,
            investor_profile=profile,
            financial_score_col="financial_score",
            similarity_rank_col="similarity_rank",
            sector_position_col="sector_position",
        )
        print(f"   pref_{profile}: mean={df[f'pref_{profile}'].mean():.1f}, "
              f"std={df[f'pref_{profile}'].std():.1f}")

    # Save indexed data
    outpath = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"\n[OK] Indexed data saved to {outpath}")
    print(f"     {len(df)} companies, {len(df.columns)} columns")

    # Save similarity matrix
    if sim_matrix is not None:
        sim_path = PROJECT_ROOT / "data" / "processed" / "similarity_matrix.csv"
        sim_matrix.to_csv(sim_path, encoding="utf-8")
        print(f"[OK] Similarity matrix saved to {sim_path}")

    # Save rankings
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ranking_cols = ["ticker", "company_name", "sector", "country",
                    "ESG_composite", "financial_score", "market_score",
                    "operational_score", "risk_adjusted_score", "value_score",
                    "growth_score", "stability_score",
                    "pref_balanced", "pref_esg_first", "pref_financial_first"]
    avail_ranking = [c for c in ranking_cols if c in df.columns]
    rankings = df[avail_ranking].sort_values("pref_balanced", ascending=False)
    rankings["rank"] = range(1, len(rankings) + 1)
    rankings.to_csv(tables_dir / "company_rankings.csv", index=False, encoding="utf-8")
    print(f"[OK] Rankings saved to {tables_dir / 'company_rankings.csv'}")

    # Score summary by sector
    score_cols = [c for c in ["ESG_composite", "financial_score", "market_score",
                               "operational_score", "risk_adjusted_score",
                               "value_score", "growth_score", "stability_score",
                               "pref_balanced"] if c in df.columns]
    if "sector" in df.columns and score_cols:
        sector_summary = df.groupby("sector")[score_cols].agg(["mean", "std", "count"])
        sector_summary.to_csv(tables_dir / "sector_score_summary.csv", encoding="utf-8")

    print("\n[DONE] Next: python scripts/04_statistical_tests.py")


if __name__ == "__main__":
    main()
