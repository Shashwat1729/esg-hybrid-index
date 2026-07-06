"""
Step 06: Benchmark Index Comparison
=====================================
Compares our multi-factor index against single-factor and equal-weight benchmarks:
  1. Constituent overlap analysis
  2. Sector composition comparison
  3. Score distribution comparison
  4. Performance simulation (equal-weight portfolio returns)
  5. Cross-sectional dispersion metrics (information ratio, selection quality)
  6. Multi-factor advantage analysis (diversification benefit)
  7. US vs India sub-index comparison

Input:  data/processed/indexed_data.csv
Output: reports/tables/benchmark_*.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, load_profile_weights
from src.constants import RANDOM_SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES = PROJECT_ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PROXY_HEADER_COMMENT = "# Cross-sectional momentum proxy, not time-series returns"


def save_table(df, filename, *, index=False, proxy_header=False):
    path = TABLES / filename
    if proxy_header:
        with open(path, "w", encoding="utf-8") as f:
            f.write(PROXY_HEADER_COMMENT + "\n")
            df.to_csv(f, index=index)
    else:
        df.to_csv(path, index=index)

# Risk-free rate assumption (currently unused).
# Set to 0.0 because the cross-sectional dispersion metrics below are NOT
# time-series risk-adjusted returns; they measure relative stock-selection
# quality across N companies at a single point in time.
ANNUAL_RISK_FREE_RATE = 0.0

# ---------------------------------------------------------------------------
# LOOK-AHEAD BIAS NOTE
# ---------------------------------------------------------------------------
# market_score is composed of three sub-categories (see config/index_config.yaml):
#   - liquidity  (40%): avg_daily_volume (bid_ask_spread, free_float_pct removed — Issue M6: synthetic noise)
#   - volatility (30%): price_volatility, beta
#   - momentum   (30%): price_momentum_1m, price_momentum_3m, price_momentum_6m
#
# The standard IC test correlates each factor score with a trailing momentum
# column (e.g. price_momentum_6m) as a return proxy.  Because market_score
# *includes* that same momentum data as an input, the resulting IC ≈ 0.47 is
# partially tautological — it measures self-correlation, not genuine
# predictive power.
#
# To address this we compute:
#   1. "biased IC"  — original market_score vs return proxy (for reference)
#   2. "clean IC"   — market_score_ex_momentum (liquidity + volatility only)
#                      vs the same return proxy, removing the circular dependency
#
# The momentum sub-scores that overlap with the return proxy are:
#   price_momentum_1m_norm, price_momentum_3m_norm, price_momentum_6m_norm
# ---------------------------------------------------------------------------

# Momentum-related normalised columns that feed into market_score and also
# serve as return proxies — these create the circular dependency.
_MOMENTUM_NORM_COLS = [
    "price_momentum_1m_norm",
    "price_momentum_3m_norm",
    "price_momentum_6m_norm",
]


def load_data():
    df = load_indexed_data(PROJECT_ROOT)
    print(f"[OK] Loaded {len(df)} companies")
    return df


# ---------------------------------------------------------------------------
# 1. Sector Composition
# ---------------------------------------------------------------------------
def sector_composition(df):
    print("\n--- Benchmark: Sector Composition ---")
    if "sector" not in df.columns:
        return

    top30 = df.nlargest(30, "pref_balanced") if "pref_balanced" in df.columns else df.head(30)

    full_sectors = df["sector"].value_counts(normalize=True).rename("full_universe")
    top_sectors = top30["sector"].value_counts(normalize=True).rename("our_top30")

    rows = []
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        top_sub = sub.nlargest(min(15, len(sub)), "pref_balanced") if "pref_balanced" in sub.columns else sub
        for sector in sub["sector"].unique():
            rows.append({
                "country": country, "sector": sector,
                "count_universe": len(sub[sub["sector"] == sector]),
                "pct_universe": len(sub[sub["sector"] == sector]) / len(sub) * 100,
                "count_top": len(top_sub[top_sub["sector"] == sector]),
                "pct_top": len(top_sub[top_sub["sector"] == sector]) / max(1, len(top_sub)) * 100,
            })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_sector_composition.csv", index=False)

    summary = pd.DataFrame({"full_universe": full_sectors, "our_top30": top_sectors}).fillna(0)
    summary.to_csv(TABLES / "benchmark_sector_summary.csv")
    print(f"  [OK] Saved benchmark_sector_composition.csv, benchmark_sector_summary.csv")
    return result


# ---------------------------------------------------------------------------
# 2. Score Comparison: Multiple Strategies
# ---------------------------------------------------------------------------
def score_comparison(df):
    print("\n--- Benchmark: Score Comparison ---")

    rankings = {}
    if "pref_balanced" in df.columns:
        rankings["our_balanced"] = df.nlargest(20, "pref_balanced")["ticker"].tolist()
    if "pref_esg_first" in df.columns:
        rankings["our_esg_first"] = df.nlargest(20, "pref_esg_first")["ticker"].tolist()
    if "pref_financial_first" in df.columns:
        rankings["our_fin_first"] = df.nlargest(20, "pref_financial_first")["ticker"].tolist()
    if "ESG_composite" in df.columns:
        rankings["esg_only"] = df.nlargest(20, "ESG_composite")["ticker"].tolist()
    if "financial_score" in df.columns:
        rankings["financial_only"] = df.nlargest(20, "financial_score")["ticker"].tolist()
    if "risk_adjusted_score" in df.columns:
        rankings["risk_adj_only"] = df.nlargest(20, "risk_adjusted_score")["ticker"].tolist()
    if "growth_score" in df.columns:
        rankings["growth_only"] = df.nlargest(20, "growth_score")["ticker"].tolist()

    # Overlap matrix
    names = list(rankings.keys())
    overlap_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for n1 in names:
        for n2 in names:
            overlap_matrix.loc[n1, n2] = len(set(rankings[n1]) & set(rankings[n2]))

    overlap_matrix.to_csv(TABLES / "benchmark_ranking_overlap.csv")

    # Per-strategy mean scores
    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score", "stability_score"]
    avail = [c for c in score_cols if c in df.columns]
    rows = []
    for name, tickers in rankings.items():
        sub = df[df["ticker"].isin(tickers)]
        row = {"strategy": name, "n_companies": len(sub)}
        for col in avail:
            row[f"avg_{col}"] = sub[col].mean()
        rows.append(row)

    strat_scores = pd.DataFrame(rows)
    strat_scores.to_csv(TABLES / "benchmark_strategy_scores.csv", index=False)
    print(f"  [OK] Saved benchmark_ranking_overlap.csv, benchmark_strategy_scores.csv")
    return strat_scores


# ---------------------------------------------------------------------------
# 3. Simulated Portfolio Performance (with Information Ratio)
# ---------------------------------------------------------------------------
def simulated_performance(df):
    """Cross-sectional stock-selection quality comparison across strategies.

    IMPORTANT METHODOLOGICAL CAVEAT — READ BEFORE INTERPRETING RESULTS:
    ====================================================================
    These metrics are computed CROSS-SECTIONALLY from trailing momentum
    values (price_momentum_Xm), NOT from time-series portfolio returns.
    They measure relative stock-selection quality, not absolute portfolio
    performance.

    Specifically:
    - "cross_sectional_ir" = mean(momentum) / std(momentum) across N selected
      stocks.  This is analogous to a cross-sectional information ratio
      measuring selection dispersion, NOT a time-series Sharpe ratio.
    - "cross_sectional_sortino_proxy" = mean(momentum) / downside_deviation
      across N stocks.  Measures downside dispersion of selected stocks, NOT
      a time-series Sortino ratio.
    - "info_ratio" = (portfolio_mean - universe_mean) / tracking_error.
      Measures differentiation from the universe average momentum, NOT
      genuine active return.

    These metrics are presented for COMPARISON across strategies (relative
    assessment) rather than as evidence of absolute predictive performance.
    A proper backtest would require time-series data with out-of-sample
    forward returns at each rebalance date.
    """
    # M4 DISCLOSURE: All performance metrics in this analysis are cross-sectional
    # momentum proxies measured at a SINGLE point in time. They represent the
    # trailing price momentum dispersion between portfolios, NOT actual
    # time-series investment returns from holding the portfolio.
    # To convert these to investable claims, a proper walk-forward backtest
    # with quarterly rebalancing and transaction costs is required.
    print("\n--- Benchmark: Simulated Portfolio Performance ---")
    print("  NOTE: Using trailing momentum as return proxy (relative comparison only)")

    return_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
    avail_ret = [c for c in return_cols if c in df.columns and df[c].notna().sum() > 5]
    if not avail_ret:
        print("  [SKIP] No return data available")
        return

    # Build portfolios from different strategies
    portfolios = {}
    if "pref_balanced" in df.columns:
        portfolios["our_balanced_top20"] = df.nlargest(20, "pref_balanced")
        portfolios["our_balanced_top30"] = df.nlargest(30, "pref_balanced")
    if "pref_esg_first" in df.columns:
        portfolios["our_esg_first_top20"] = df.nlargest(20, "pref_esg_first")
    if "pref_financial_first" in df.columns:
        portfolios["our_fin_first_top20"] = df.nlargest(20, "pref_financial_first")
    if "ESG_composite" in df.columns:
        portfolios["esg_only_top20"] = df.nlargest(20, "ESG_composite")
    if "financial_score" in df.columns:
        portfolios["financial_only_top20"] = df.nlargest(20, "financial_score")
    if "risk_adjusted_score" in df.columns:
        portfolios["risk_adj_only_top20"] = df.nlargest(20, "risk_adjusted_score")
    if "growth_score" in df.columns:
        portfolios["growth_only_top20"] = df.nlargest(20, "growth_score")
    portfolios["full_universe"] = df

    # Compute universe benchmark returns for information ratio
    universe_returns = {}
    for rc in avail_ret:
        universe_returns[rc] = df[rc].dropna().mean()

    rows = []
    for name, port_df in portfolios.items():
        row = {"portfolio": name, "n_companies": len(port_df)}
        for rc in avail_ret:
            rets = port_df[rc].dropna()
            row[f"avg_{rc}"] = rets.mean()
            row[f"std_{rc}"] = rets.std()
            # Cross-sectional information ratio: mean/std across N stocks
            if rets.std() > 1e-10:
                row[f"cross_sectional_ir_{rc}"] = rets.mean() / rets.std()
            else:
                row[f"cross_sectional_ir_{rc}"] = 0
            # Cross-sectional Sortino proxy (only downside deviation)
            downside_dev = np.sqrt(np.mean(np.minimum(rets, 0)**2))
            if downside_dev > 1e-10:
                row[f"cross_sectional_sortino_proxy_{rc}"] = rets.mean() / downside_dev
            else:
                row[f"cross_sectional_sortino_proxy_{rc}"] = np.nan
            # NOTE: max_drawdown removed — cumprod on cross-sectional data is
            # nonsensical (stocks are sorted by score, not ordered in time).
            # Information ratio vs full universe
            excess = rets.mean() - universe_returns.get(rc, 0)
            tracking_error = (rets - universe_returns.get(rc, 0)).std()
            if tracking_error > 1e-10:
                row[f"info_ratio_{rc}"] = excess / tracking_error
            else:
                row[f"info_ratio_{rc}"] = 0
            # Percentage of companies with positive returns
            row[f"pct_positive_{rc}"] = (rets > 0).mean() * 100

        # Score quality of portfolio
        for sc in ["ESG_composite", "financial_score", "risk_adjusted_score"]:
            if sc in port_df.columns:
                row[f"avg_{sc}"] = port_df[sc].mean()

        rows.append(row)

    result = pd.DataFrame(rows)
    save_table(result, "benchmark_portfolio_performance.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_portfolio_performance.csv ({len(portfolios)} strategies)")

    # Print key comparison
    for rc in avail_ret[:1]:  # Just the first return col
        print(f"\n  Portfolio momentum proxies ({rc}, cross-sectional):")
        for _, r in result.iterrows():
            avg_key = f"avg_{rc}"
            csir_key = f"cross_sectional_ir_{rc}"
            ir_key = f"info_ratio_{rc}"
            if avg_key in r:
                print(f"    {r['portfolio']:30s}: momentum_proxy={r[avg_key]:+6.2f}%, "
                      f"CS-IR={r.get(csir_key, 0):+.3f}, IR={r.get(ir_key, 0):+.3f}")

    return result


# ---------------------------------------------------------------------------
# 4. US vs India Sub-Index
# ---------------------------------------------------------------------------
def us_vs_india(df):
    print("\n--- Benchmark: US vs India Sub-Index ---")
    if "country" not in df.columns:
        return

    score_cols = ["ESG_composite", "E_score", "S_score", "G_score",
                  "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score",
                  "stability_score", "pref_balanced"]
    avail = [c for c in score_cols if c in df.columns]

    rows = []
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        row = {"country": country, "n_companies": len(sub)}
        for col in avail:
            row[f"mean_{col}"] = sub[col].mean()
            row[f"std_{col}"] = sub[col].std()
            row[f"median_{col}"] = sub[col].median()
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_us_vs_india.csv", index=False)

    if "sector" in df.columns:
        sector_country = df.groupby(["country", "sector"])[avail].mean()
        sector_country.to_csv(TABLES / "benchmark_sector_by_country.csv")

    print(f"  [OK] Saved benchmark_us_vs_india.csv, benchmark_sector_by_country.csv")
    return result


# ---------------------------------------------------------------------------
# 5. Benchmark Summary
# ---------------------------------------------------------------------------
def benchmark_summary(df):
    print("\n--- Benchmark: Summary ---")

    def _build_row(name, sub):
        ret_6m_col = "price_momentum_6m"
        ret_12m_col = "price_momentum_12m"

        ret_6m = sub[ret_6m_col].dropna() if ret_6m_col in sub.columns else pd.Series(dtype=float)
        ret_12m = sub[ret_12m_col].dropna() if ret_12m_col in sub.columns else pd.Series(dtype=float)

        row = {
            "strategy": name,
            "n_companies": len(sub),
            "avg_ESG": sub.get("ESG_composite", pd.Series()).mean(),
            "avg_financial": sub.get("financial_score", pd.Series()).mean(),
            "avg_momentum_proxy_6m": ret_6m.mean() if len(ret_6m) else np.nan,
            "avg_momentum_proxy_12m": ret_12m.mean() if len(ret_12m) else np.nan,
            "cross_sectional_ir_6m": ret_6m.mean() / (ret_6m.std() + 1e-10) if len(ret_6m) else np.nan,
            "cross_sectional_ir_12m": ret_12m.mean() / (ret_12m.std() + 1e-10) if len(ret_12m) else np.nan,
            "pct_positive_momentum_proxy": (ret_6m > 0).mean() * 100 if len(ret_6m) else np.nan,
            "n_sectors": sub["sector"].nunique() if "sector" in sub.columns else None,
        }
        return row

    rows = []
    if "pref_balanced" in df.columns:
        rows.append(_build_row("Our Multi-Factor (Top 20)", df.nlargest(20, "pref_balanced")))
    if "ESG_composite" in df.columns:
        rows.append(_build_row("ESG-Only (Top 20)", df.nlargest(20, "ESG_composite")))
    if "financial_score" in df.columns:
        rows.append(_build_row("Financial-Only (Top 20)", df.nlargest(20, "financial_score")))
    if "growth_score" in df.columns:
        rows.append(_build_row("Growth-Only (Top 20)", df.nlargest(20, "growth_score")))
    rows.append(_build_row("Full Universe", df))

    result = pd.DataFrame(rows)
    save_table(result, "benchmark_summary.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_summary.csv")
    return result


# ---------------------------------------------------------------------------
# 6. Multi-Horizon Return Comparison
# ---------------------------------------------------------------------------
def multi_horizon_comparison(df):
    """Create a clean comparison table across 1m, 3m, 6m, 12m horizons."""
    print("\n--- Benchmark: Multi-Horizon Momentum Proxy Comparison ---")

    horizons = {
        "1m": "price_momentum_1m", "3m": "price_momentum_3m",
        "6m": "price_momentum_6m", "12m": "price_momentum_12m",
    }
    avail_horizons = {k: v for k, v in horizons.items() if v in df.columns and df[v].notna().sum() > 5}

    if not avail_horizons:
        print("  [SKIP] No return data")
        return

    # Define strategies
    strategies = {}
    if "pref_balanced" in df.columns:
        strategies["Our_MultiF_Top20"] = df.nlargest(20, "pref_balanced")
    if "pref_esg_first" in df.columns:
        strategies["Our_ESGFirst_Top20"] = df.nlargest(20, "pref_esg_first")
    if "ESG_composite" in df.columns:
        strategies["ESG_Only_Top20"] = df.nlargest(20, "ESG_composite")
    if "financial_score" in df.columns:
        strategies["Financial_Only_Top20"] = df.nlargest(20, "financial_score")
    if "growth_score" in df.columns:
        strategies["Growth_Only_Top20"] = df.nlargest(20, "growth_score")
    strategies["Full_Universe"] = df

    rows = []
    for strat_name, sub in strategies.items():
        row = {"strategy": strat_name, "n": len(sub)}
        for horizon_label, col in avail_horizons.items():
            rets = sub[col].dropna()
            row[f"avg_momentum_proxy_{horizon_label}"] = rets.mean()
            row[f"cross_sectional_ir_{horizon_label}"] = rets.mean() / (rets.std() + 1e-10)
            row[f"worst_momentum_proxy_{horizon_label}"] = rets.min()
            row[f"pct_positive_momentum_proxy_{horizon_label}"] = (rets > 0).mean() * 100
        # Diversification
        if "sector" in sub.columns:
            sector_pcts = sub["sector"].value_counts(normalize=True)
            row["sector_hhi"] = (sector_pcts ** 2).sum()
        rows.append(row)

    result = pd.DataFrame(rows)
    save_table(result, "benchmark_multi_horizon.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_multi_horizon.csv ({len(strategies)} strategies x {len(avail_horizons)} horizons)")

    # Print comparison
    for h_label, col in list(avail_horizons.items())[:2]:
        print(f"\n  {h_label} horizon:")
        for _, r in result.iterrows():
            ret_key = f"avg_momentum_proxy_{h_label}"
            csir_key = f"cross_sectional_ir_{h_label}"
            print(f"    {r['strategy']:25s}: momentum_proxy={r.get(ret_key, 0):+6.2f}%, CS-IR={r.get(csir_key, 0):+.3f}")

    return result


# ---------------------------------------------------------------------------
# 7. Alpha/Beta Decomposition
# ---------------------------------------------------------------------------
def alpha_beta_analysis(df):
    """Decompose each strategy's return into excess momentum + beta * benchmark.

    For cross-sectional data (single time point, multiple companies), we
    compute portfolio-level beta as the weighted average of member betas
    (from the 'beta' column).  Excess average momentum is then:
        excess_avg_momentum = R_portfolio - beta_portfolio * R_benchmark
    This avoids the trivial regression problem where regressing a subset's
    returns against the same returns always yields alpha=0, beta=1.

    NOTE: "excess_avg_momentum" is a single-point cross-sectional estimate,
    NOT a CAPM regression intercept (Jensen's alpha).  It measures how much
    the selected stocks' average trailing momentum exceeds what their
    aggregate beta exposure would predict from the universe average.
    """
    print("\n--- Benchmark: Excess Momentum / Beta Decomposition ---")

    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break
    if return_col is None:
        print("  [SKIP] No return data")
        return

    print(f"  ⚠ IMPORTANT: 'excess return' = cross-sectional momentum dispersion, "
          f"NOT actual portfolio P&L. See M4 disclosure.")

    # Benchmark: full universe return
    bench_mean = df[return_col].dropna().mean()
    has_beta = "beta" in df.columns and df["beta"].notna().sum() > 10

    strategies = {}
    if "pref_balanced" in df.columns:
        strategies["Our_MultiF_Top20"] = df.nlargest(20, "pref_balanced")
    if "pref_esg_first" in df.columns:
        strategies["Our_ESGFirst_Top20"] = df.nlargest(20, "pref_esg_first")
    if "ESG_composite" in df.columns:
        strategies["ESG_Only_Top20"] = df.nlargest(20, "ESG_composite")
    if "financial_score" in df.columns:
        strategies["Financial_Only_Top20"] = df.nlargest(20, "financial_score")
    if "growth_score" in df.columns:
        strategies["Growth_Only_Top20"] = df.nlargest(20, "growth_score")

    rows = []
    for name, sub in strategies.items():
        port_rets = sub[return_col].dropna()
        port_mean = port_rets.mean()
        port_std = port_rets.std()

        # Portfolio beta = average beta of member stocks
        if has_beta:
            port_beta = sub["beta"].dropna().mean()
        else:
            port_beta = 1.0

        # Cross-sectional excess momentum: R_p - beta_p * R_benchmark
        # NOTE: This is NOT Jensen's alpha (a regression intercept).
        excess_avg_momentum = port_mean - port_beta * bench_mean

        # Excess return and information ratio
        excess = port_mean - bench_mean
        tracking_error = (port_rets - bench_mean).std()
        info_ratio = excess / (tracking_error + 1e-10)

        rows.append({
            "strategy": name,
            "avg_momentum_proxy": port_mean,
            "benchmark_momentum_proxy": bench_mean,
            "excess_momentum_proxy": excess,
            "excess_avg_momentum_proxy": excess_avg_momentum,
            "beta": port_beta,
            "cross_sectional_ir": port_mean / (port_std + 1e-10),
            "information_ratio": info_ratio,
            "tracking_error": tracking_error,
            "momentum_proxy_col": return_col,
            "metric_type": "cross_sectional_proxy",
            "caveat": "Not time-series portfolio returns",
        })

    result = pd.DataFrame(rows)
    if "gross_excess_return_annual" in result.columns:
        result = result.rename(columns={
            "gross_excess_return_annual": "cross_sectional_momentum_dispersion_annual_proxy"
        })
    save_table(result, "benchmark_alpha_beta.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_alpha_beta.csv")

    for _, r in result.iterrows():
        print(f"    {r['strategy']:25s}: excess_mom={r['excess_avg_momentum_proxy']:+.2f}%, beta={r['beta']:.2f}, "
              f"excess={r['excess_momentum_proxy']:+.2f}%, IR={r['information_ratio']:+.3f}")

    return result


# ---------------------------------------------------------------------------
# 8. Equal-Weighted and Value-Weighted Benchmark
# ---------------------------------------------------------------------------
def equal_vs_value_weighted(df):
    """Compare equal-weighted, value-weighted (by market cap), and our score-weighted approaches."""
    print("\n--- Benchmark: Equal-Weight vs Value-Weight vs Score-Weight ---")

    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break
    if return_col is None:
        print("  [SKIP] No return data")
        return

    rows = []

    # 1. Equal-weighted full universe
    rets = df[return_col].dropna()
    eq_ret = rets.mean()
    eq_std = rets.std()
    rows.append({
        "method": "Equal-Weight (Full Universe)", "n": len(rets),
        "momentum_proxy": eq_ret, "std": eq_std,
        "cross_sectional_ir": eq_ret / (eq_std + 1e-10),
    })

    # 2. Value-weighted (by market_cap) full universe
    if "market_cap" in df.columns:
        valid = df[[return_col, "market_cap"]].dropna()
        if len(valid) > 5 and valid["market_cap"].sum() > 0:
            weights = valid["market_cap"] / valid["market_cap"].sum()
            vw_ret = (valid[return_col] * weights).sum()
            # Weighted std: sqrt(sum(w_i * (r_i - vw_ret)^2))
            vw_std = np.sqrt((weights * (valid[return_col] - vw_ret) ** 2).sum())
            rows.append({
                "method": "Value-Weight (Full Universe)", "n": len(valid),
                "momentum_proxy": vw_ret, "std": vw_std,
                "cross_sectional_ir": vw_ret / (vw_std + 1e-10),
            })

    # 3. Our score-weighted top 20
    if "pref_balanced" in df.columns:
        top20 = df.nlargest(20, "pref_balanced")
        t20_rets = top20[return_col].dropna()
        rows.append({
            "method": "Score-Weight Top 20 (Ours)", "n": len(t20_rets),
            "momentum_proxy": t20_rets.mean(), "std": t20_rets.std(),
            "cross_sectional_ir": t20_rets.mean() / (t20_rets.std() + 1e-10),
        })

        # 4. Score-weighted using preference as portfolio weight
        valid = top20[[return_col, "pref_balanced"]].dropna()
        if len(valid) > 3 and valid["pref_balanced"].sum() > 0:
            sw = valid["pref_balanced"] / valid["pref_balanced"].sum()
            sw_ret = (valid[return_col] * sw).sum()
            sw_std = np.sqrt((sw * (valid[return_col] - sw_ret) ** 2).sum())
            rows.append({
                "method": "Preference-Weight Top 20 (Ours)", "n": len(valid),
                "momentum_proxy": sw_ret, "std": sw_std,
                "cross_sectional_ir": sw_ret / (sw_std + 1e-10),
            })

    # 5. Random top 20 baseline (average of 50 random selections)
    # Use the AVERAGE within-portfolio CS-IR, not cross-draw std
    rng = np.random.default_rng(RANDOM_SEED)
    random_rets_list = []
    random_csirs = []
    for _ in range(50):
        sample = df.sample(min(20, len(df)), replace=False, random_state=rng.integers(2**31))
        r = sample[return_col].dropna()
        random_rets_list.append(r.mean())
        if len(r) > 2 and r.std() > 1e-10:
            random_csirs.append(r.mean() / r.std())
        else:
            random_csirs.append(0.0)
    random_avg = np.mean(random_rets_list)
    random_within_std = np.mean([df.sample(20, replace=False, random_state=rng.integers(2**31))[return_col].dropna().std()
                                  for _ in range(50)])
    rows.append({
        "method": "Random Top 20 (50 draws avg)", "n": 20,
        "momentum_proxy": random_avg, "std": random_within_std,
        "cross_sectional_ir": np.mean(random_csirs),
    })

    result = pd.DataFrame(rows)
    save_table(result, "benchmark_weighting_methods.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_weighting_methods.csv")
    for _, r in result.iterrows():
        print(f"    {r['method']:40s}: momentum_proxy={r['momentum_proxy']:+6.2f}%, CS-IR={r['cross_sectional_ir']:+.3f}")
    return result


# ---------------------------------------------------------------------------
# 9. Clean Factor Validity (IC) — corrected for look-ahead bias
# ---------------------------------------------------------------------------

def compute_clean_ic(df, score_col, return_col):
    """Compute Spearman rank IC between *score_col* and *return_col*.

    Parameters
    ----------
    df : pd.DataFrame
    score_col : str   – factor score column
    return_col : str  – return-proxy column

    Returns
    -------
    dict with keys: score, return_proxy, ic_spearman, ic_pvalue, n, significant
    """
    valid = df[[score_col, return_col]].dropna()
    if len(valid) < 10:
        return {
            "score": score_col,
            "return_proxy": return_col,
            "ic_spearman": np.nan,
            "ic_pvalue": np.nan,
            "n": len(valid),
            "significant": "N/A",
        }
    rho, p = spearmanr(valid[score_col], valid[return_col])
    return {
        "score": score_col,
        "return_proxy": return_col,
        "ic_spearman": round(rho, 4),
        "ic_pvalue": round(p, 6),
        "n": len(valid),
        "significant": "Yes" if p < 0.05 else "No",
    }


def _build_market_score_ex_momentum(df):
    """Reconstruct market_score excluding momentum sub-category.

    Uses only the liquidity and volatility sub-scores that are already
    present in the dataframe (market_liquidity_score, market_volatility_score).
    If those are missing, falls back to averaging available non-momentum
    normalised indicators.

    Returns
    -------
    pd.Series  — market_score_ex_momentum (same scale as market_score)
    """
    liq_col = "market_liquidity_score"
    vol_col = "market_volatility_score"

    if liq_col in df.columns and vol_col in df.columns:
        # Original weights from config: liquidity 0.40, volatility 0.30, momentum 0.30
        # Excluding momentum → renormalise: liq = 0.40/0.70, vol = 0.30/0.70
        w_liq = 0.40 / 0.70
        w_vol = 0.30 / 0.70
        raw = w_liq * df[liq_col].fillna(0) + w_vol * df[vol_col].fillna(0)
        # Rescale to [0, 100] using same transform as MarketFactorScorer
        score = 50 + (raw * 20)
        return score.clip(0, 100)

    # Fallback: average all non-momentum market norm cols
    non_mom_norms = [
        "avg_daily_volume_norm",
        # Removed (Issue M6): "bid_ask_spread_norm", "free_float_pct_norm" — synthetic noise
        "price_volatility_norm", "beta_norm",
    ]
    avail = [c for c in non_mom_norms if c in df.columns]
    if not avail:
        return pd.Series(np.nan, index=df.index, name="market_score_ex_momentum")
    raw = df[avail].mean(axis=1)
    score = 50 + (raw * 20)
    return score.clip(0, 100)


def factor_validity_clean(df):
    """Run IC tests on all factor scores with a clean return proxy.

    CIRCULARITY FIX (C1):
    =====================
    market_score includes price_momentum_1m/3m/6m as inputs (30% weight via
    momentum subcategory).  These same momentum columns are used as "return
    proxies" for portfolio performance evaluation.  Correlating market_score
    against momentum returns therefore measures partial self-correlation,
    inflating IC (observed ~0.47) to a tautological level.

    This function reports:
      - "clean IC" (PRIMARY) using market_score_ex_momentum — the corrected
        metric stripped of circular momentum inputs
      - "raw IC" (REFERENCE) using original market_score — for transparency,
        clearly flagged as biased

    Saves: reports/tables/benchmark_factor_validity_clean.csv
    """
    print("\n--- Factor Validity: Clean IC (PRIMARY — circularity-corrected) ---")

    # Pick best return proxy
    return_col = None
    for rc in ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return_col = rc
            break
    if return_col is None:
        print("  [SKIP] No return proxy available")
        return None

    print(f"  Return proxy: {return_col}")

    # Build momentum-excluded market score
    df = df.copy()
    df["market_score_ex_momentum"] = _build_market_score_ex_momentum(df)

    factor_scores = [
        "ESG_composite", "financial_score", "market_score", "operational_score",
        "risk_adjusted_score", "growth_score", "value_score", "stability_score",
    ]
    avail = [f for f in factor_scores if f in df.columns]

    rows = []
    for factor in avail:
        ic = compute_clean_ic(df, factor, return_col)

        if factor == "market_score":
            # ---- BIASED (raw) IC — flagged for reference only ----
            ic["ic_type"] = "raw_ic"
            ic["bias_flag"] = "BIASED — momentum overlap with return proxy"
            ic["note"] = (
                f"market_score includes momentum sub-scores derived from "
                f"{return_col}; IC is partially tautological.  "
                f"DO NOT use as headline metric."
            )
            rows.append(ic)

            # ---- CLEAN IC — PRIMARY metric ----
            ic_clean = compute_clean_ic(df, "market_score_ex_momentum", return_col)
            ic_clean["score"] = "market_score_ex_momentum"
            ic_clean["ic_type"] = "clean_ic"
            ic_clean["bias_flag"] = "CLEAN — momentum sub-scores removed (PRIMARY)"
            ic_clean["note"] = (
                "Liquidity + volatility sub-categories only "
                "(momentum indicators excluded to avoid circular IC).  "
                "This is the CORRECT metric for market factor validity."
            )
            rows.append(ic_clean)
        else:
            ic["ic_type"] = "clean_ic"
            ic["bias_flag"] = "none"
            ic["note"] = ""
            rows.append(ic)

    result = pd.DataFrame(rows)

    # Reorder columns for readability — ic_type is now prominent
    col_order = [
        "score", "ic_type", "return_proxy", "ic_spearman", "ic_pvalue",
        "significant", "n", "bias_flag", "note",
    ]
    result = result[[c for c in col_order if c in result.columns]]

    save_table(result, "benchmark_factor_validity_clean.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_factor_validity_clean.csv ({len(result)} rows)")

    # ---- Print table: CLEAN IC as primary, raw IC clearly secondary ----
    print(f"\n  {'Factor':<30s} {'Type':>10s} {'IC':>8s} {'p':>10s} {'Sig':>5s}  Bias Flag")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*10} {'-'*5}  {'-'*45}")
    for _, r in result.iterrows():
        ic_val = r["ic_spearman"]
        ic_str = f"{ic_val:+.4f}" if not np.isnan(ic_val) else "   N/A"
        p_val = r["ic_pvalue"]
        p_str = f"{p_val:.6f}" if not np.isnan(p_val) else "      N/A"
        print(f"  {r['score']:<30s} {r['ic_type']:>10s} {ic_str:>8s} {p_str:>10s} "
              f"{r['significant']:>5s}  {r['bias_flag']}")

    # Highlight the key finding
    mkt_rows = result[result["score"].isin(["market_score", "market_score_ex_momentum"])]
    if len(mkt_rows) == 2:
        biased_ic = mkt_rows.loc[mkt_rows["score"] == "market_score", "ic_spearman"].iloc[0]
        clean_ic = mkt_rows.loc[mkt_rows["score"] == "market_score_ex_momentum", "ic_spearman"].iloc[0]
        if not np.isnan(biased_ic) and not np.isnan(clean_ic):
            drop = biased_ic - clean_ic
            print(f"\n  ** HEADLINE: Clean market IC = {clean_ic:+.4f} "
                  f"(was {biased_ic:+.4f} biased, delta = {drop:+.4f}) **")
            if abs(biased_ic) > 1e-6:
                print(f"  ** {drop/biased_ic*100:.0f}% of the raw IC was due to momentum "
                      f"sub-scores overlapping with {return_col} **")
            print(f"  ** The clean IC is the CORRECT metric for market factor validity **")

    return result


# ---------------------------------------------------------------------------
# 10. Composite Ex-Market Preference Score & Unbiased Portfolio Evaluation
# ---------------------------------------------------------------------------

def _get_composite_ex_market_col(df):
    """Return the name of the ex-market preference column if present.

    After the circularity fix in 03_build_index.py, `pref_balanced` IS the
    clean (ex-market) version.  The contaminated original is stored as
    `pref_balanced_with_market`.  This function returns `pref_balanced`
    (the primary clean column).

    Falls back to `pref_balanced_ex_market` for backward compatibility with
    older indexed_data.csv files generated before the rename.
    """
    if "pref_balanced" in df.columns:
        return "pref_balanced"
    if "pref_balanced_ex_market" in df.columns:
        return "pref_balanced_ex_market"
    return None


def composite_ex_market_evaluation(df):
    """Evaluate portfolio performance using clean (ex-market) selection criterion.

    CIRCULARITY FIX:
    ================
    After the rename in 03_build_index.py:
      - pref_balanced            = clean (ex-market) — PRIMARY
      - pref_balanced_with_market = original (contaminated) — for audit

    This function:
      1. Uses pref_balanced (now clean, ex-market) as the primary selection criterion
      2. Measures the SAME return proxies — selection is independent of
         the return measure, eliminating the circular dependency
      3. Reports side-by-side: contaminated (_with_market) vs clean metrics

    Saves: reports/tables/benchmark_ex_market_performance.csv
    """
    print("\n--- Composite Ex-Market Portfolio Evaluation (circularity-corrected) ---")

    return_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
    avail_ret = [c for c in return_cols if c in df.columns and df[c].notna().sum() > 5]
    if not avail_ret:
        print("  [SKIP] No return data available")
        return

    # Determine ex-market column
    ex_market_col = _get_composite_ex_market_col(df)
    if ex_market_col is None:
        # Compute on the fly: simple weighted average excluding market_score
        print("  [INFO] pref_balanced_ex_market not found; computing inline...")
        _weights = {
            "ESG_composite": 0.20, "financial_score": 0.20,
            "operational_score": 0.10, "risk_adjusted_score": 0.08,
            "growth_score": 0.10, "value_score": 0.08, "stability_score": 0.05,
        }
        avail_w = {k: v for k, v in _weights.items() if k in df.columns}
        total_w = sum(avail_w.values())
        if total_w == 0:
            print("  [SKIP] Insufficient factor columns for ex-market composite")
            return
        avail_w = {k: v / total_w for k, v in avail_w.items()}
        df = df.copy()
        ex_score = pd.Series(0.0, index=df.index)
        for col, w in avail_w.items():
            ex_score += w * df[col].fillna(50)
        df["pref_balanced_ex_market"] = ex_score
        ex_market_col = "pref_balanced_ex_market"

    # Build portfolios: clean (primary) vs contaminated (_with_market)
    # After circularity fix: pref_balanced IS the clean version
    portfolios = {}
    if "pref_balanced_with_market" in df.columns:
        portfolios["balanced_CONTAMINATED_top20"] = df.nlargest(20, "pref_balanced_with_market")
        portfolios["balanced_CONTAMINATED_top30"] = df.nlargest(30, "pref_balanced_with_market")
    portfolios["balanced_CLEAN_top20"] = df.nlargest(20, ex_market_col)
    portfolios["balanced_CLEAN_top30"] = df.nlargest(30, ex_market_col)

    # Also ex-market variants for other profiles if available
    for profile in ["esg_first", "financial_first"]:
        # After rename: pref_{profile} = clean, pref_{profile}_with_market = contaminated
        orig_col = f"pref_{profile}"
        contaminated_col = f"pref_{profile}_with_market"
        if orig_col in df.columns:
            portfolios[f"{profile}_CLEAN_top20"] = df.nlargest(20, orig_col)
        if contaminated_col in df.columns:
            portfolios[f"{profile}_CONTAMINATED_top20"] = df.nlargest(20, contaminated_col)

    portfolios["full_universe"] = df

    # Universe benchmark returns
    universe_returns = {}
    for rc in avail_ret:
        universe_returns[rc] = df[rc].dropna().mean()

    rows = []
    for name, port_df in portfolios.items():
        row = {"portfolio": name, "n_companies": len(port_df)}
        # Tag whether this is a clean or contaminated selection
        row["selection_type"] = "CLEAN" if "CLEAN" in name else ("CONTAMINATED" if "CONTAMINATED" in name else "universe")

        for rc in avail_ret:
            rets = port_df[rc].dropna()
            row[f"avg_{rc}"] = rets.mean()
            row[f"std_{rc}"] = rets.std()
            if rets.std() > 1e-10:
                row[f"sharpe_{rc}"] = rets.mean() / rets.std()
            else:
                row[f"sharpe_{rc}"] = 0
            # Sortino
            downside_dev = np.sqrt(np.mean(np.minimum(rets, 0)**2))
            if downside_dev > 1e-10:
                row[f"sortino_{rc}"] = rets.mean() / downside_dev
            else:
                row[f"sortino_{rc}"] = np.nan
            # Information ratio
            excess = rets.mean() - universe_returns.get(rc, 0)
            tracking_error = (rets - universe_returns.get(rc, 0)).std()
            if tracking_error > 1e-10:
                row[f"info_ratio_{rc}"] = excess / tracking_error
            else:
                row[f"info_ratio_{rc}"] = 0

        for sc in ["ESG_composite", "financial_score", "risk_adjusted_score"]:
            if sc in port_df.columns:
                row[f"avg_{sc}"] = port_df[sc].mean()

        rows.append(row)

    result = pd.DataFrame(rows)
    save_table(result, "benchmark_ex_market_performance.csv", index=False, proxy_header=True)
    print(f"  [OK] Saved benchmark_ex_market_performance.csv ({len(portfolios)} portfolios)")

    # Print side-by-side comparison for the primary return col
    rc = avail_ret[0]
    print(f"\n  Side-by-side ({rc}): CONTAMINATED vs CLEAN selection")
    print(f"  {'Portfolio':<35s} {'Type':>12s} {'Return':>8s} {'Sharpe':>8s} {'IR':>8s}")
    print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for _, r in result.iterrows():
        avg_key = f"avg_{rc}"
        sh_key = f"sharpe_{rc}"
        ir_key = f"info_ratio_{rc}"
        sel_type = r.get("selection_type", "")
        print(f"  {r['portfolio']:<35s} {sel_type:>12s} "
              f"{r.get(avg_key, 0):+7.2f}% {r.get(sh_key, 0):+7.3f} "
              f"{r.get(ir_key, 0):+7.3f}")

    # Quantify inflation
    contaminated_rows = result[result["selection_type"] == "CONTAMINATED"]
    clean_rows = result[result["selection_type"] == "CLEAN"]
    if len(contaminated_rows) > 0 and len(clean_rows) > 0:
        contaminated_sharpe = contaminated_rows[f"sharpe_{rc}"].iloc[0]
        clean_sharpe = clean_rows[f"sharpe_{rc}"].iloc[0]
        if abs(contaminated_sharpe) > 1e-6:
            inflation = (contaminated_sharpe - clean_sharpe) / abs(contaminated_sharpe) * 100
            print(f"\n  ** Sharpe inflation from circularity: "
                  f"{contaminated_sharpe:+.3f} (contaminated) -> {clean_sharpe:+.3f} (clean), "
                  f"delta = {inflation:+.1f}% **")

    return result


# ---------------------------------------------------------------------------
# 11. ESG Premium Analysis
# ---------------------------------------------------------------------------
def esg_premium_analysis(df):
    """Compute return premium for high-ESG vs low-ESG within same financial quality."""
    print("\n--- Benchmark: ESG Premium Analysis ---")

    rows = []
    return_cols = {"1m": "price_momentum_1m", "3m": "price_momentum_3m",
                   "6m": "price_momentum_6m"}

    # Split into financial quality terciles
    if "financial_score" not in df.columns or "ESG_composite" not in df.columns:
        print("  [SKIP] Missing financial_score or ESG_composite")
        return

    df = df.dropna(subset=["financial_score", "ESG_composite"])
    df["fin_tercile"] = pd.qcut(df["financial_score"], 3, labels=["Low", "Mid", "High"])

    for tercile in ["Low", "Mid", "High"]:
        sub = df[df["fin_tercile"] == tercile]
        if len(sub) < 10:
            continue
        median_esg = sub["ESG_composite"].median()
        high_esg = sub[sub["ESG_composite"] >= median_esg]
        low_esg = sub[sub["ESG_composite"] < median_esg]

        for horizon, col in return_cols.items():
            if col not in df.columns:
                continue
            high_ret = high_esg[col].dropna().mean()
            low_ret = low_esg[col].dropna().mean()
            premium = high_ret - low_ret
            rows.append({
                "financial_tercile": tercile,
                "horizon": horizon,
                "high_esg_momentum_proxy": round(high_ret, 4),
                "low_esg_momentum_proxy": round(low_ret, 4),
                "esg_premium": round(premium, 4),
                "n_high": len(high_esg),
                "n_low": len(low_esg),
            })

    if rows:
        premium_df = pd.DataFrame(rows)
        save_table(premium_df, "esg_premium_analysis.csv", index=False, proxy_header=True)
        print(f"  [OK] ESG premium analysis saved: {len(premium_df)} rows")

        # Create visualization
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            terciles = premium_df["financial_tercile"].unique()
            horizons = premium_df["horizon"].unique()
            x = np.arange(len(terciles))
            width = 0.25

            for i, h in enumerate(horizons):
                vals = premium_df[premium_df["horizon"] == h]["esg_premium"].values
                if len(vals) == len(terciles):
                    bars = ax.bar(x + i * width, vals * 100, width, label=f"{h} horizon")
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

            ax.set_xlabel("Financial Quality Tercile")
            ax.set_ylabel("ESG Premium (pp)")
            ax.set_title("ESG Premium by Financial Quality Tier\n(High ESG minus Low ESG returns)")
            ax.set_xticks(x + width)
            ax.set_xticklabels(terciles)
            ax.legend()
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(FIGURES / "benchmark_esg_premium.png", dpi=150)
            plt.close(fig)
            print("  [OK] ESG premium figure saved")
        except Exception as e:
            print(f"  [WARN] Could not create ESG premium figure: {e}")
    else:
        print("  [SKIP] No ESG premium rows generated")


# ---------------------------------------------------------------------------
# 12. Factor Return Attribution
# ---------------------------------------------------------------------------
def factor_return_attribution(df):
    """Attribute portfolio returns to individual factors."""
    print("\n--- Benchmark: Factor Return Attribution ---")

    score_cols = ["ESG_composite", "financial_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score",
                  "stability_score"]
    return_col = "price_momentum_6m"

    if return_col not in df.columns:
        print(f"  [SKIP] {return_col} not in data")
        return

    rows = []
    clean = df.dropna(subset=[return_col])

    for col in score_cols:
        if col not in clean.columns:
            continue
        valid = clean.dropna(subset=[col])
        if len(valid) < 20:
            continue

        # Spearman correlation (IC)
        rho, pval = spearmanr(valid[col], valid[return_col])

        # Top vs Bottom quintile return spread
        q = pd.qcut(valid[col], 5, labels=False, duplicates="drop")
        top_ret = valid[q == q.max()][return_col].mean()
        bot_ret = valid[q == q.min()][return_col].mean()
        spread = top_ret - bot_ret

        rows.append({
            "factor": col,
            "ic_spearman": round(rho, 4),
            "ic_pvalue": round(pval, 4),
            "top_quintile_momentum_proxy": round(top_ret, 4),
            "bottom_quintile_momentum_proxy": round(bot_ret, 4),
            "quintile_spread": round(spread, 4),
            "n_companies": len(valid),
        })

    if rows:
        attr_df = pd.DataFrame(rows)
        save_table(attr_df, "factor_return_attribution.csv", index=False, proxy_header=True)
        print(f"  [OK] Factor return attribution saved ({len(rows)} factors)")
    else:
        print("  [SKIP] No factor attribution rows generated")


# ---------------------------------------------------------------------------
# 13. ESG Quintile Performance (momentum proxy attribution)
# ---------------------------------------------------------------------------
def esg_quintile_performance(df):
    print("\n--- Benchmark: ESG Quintile Momentum Proxy Performance ---")

    required = ["ESG_composite", "financial_score"]
    if any(c not in df.columns for c in required):
        print("  [SKIP] Missing ESG_composite or financial_score")
        return

    work = df.dropna(subset=["ESG_composite"]).copy()
    quintiles = pd.qcut(work["ESG_composite"], 5, labels=False, duplicates="drop")
    work["quintile"] = quintiles + 1

    rows = []
    for q in sorted(work["quintile"].dropna().unique()):
        sub = work[work["quintile"] == q]
        rows.append({
            "quintile": int(q),
            "avg_esg": sub["ESG_composite"].mean(),
            "avg_financial": sub["financial_score"].mean(),
            "avg_momentum_1m": sub["price_momentum_1m"].mean() if "price_momentum_1m" in sub.columns else np.nan,
            "avg_momentum_3m": sub["price_momentum_3m"].mean() if "price_momentum_3m" in sub.columns else np.nan,
            "avg_momentum_6m": sub["price_momentum_6m"].mean() if "price_momentum_6m" in sub.columns else np.nan,
            "avg_momentum_12m": sub["price_momentum_12m"].mean() if "price_momentum_12m" in sub.columns else np.nan,
            "n_companies": len(sub),
        })

    result = pd.DataFrame(rows)
    save_table(result, "benchmark_esg_quintile_performance.csv", index=False, proxy_header=True)
    print("  [OK] Saved benchmark_esg_quintile_performance.csv")
    return result


# ---------------------------------------------------------------------------
# 14. ESG Outperformance Contribution Waterfall
# ---------------------------------------------------------------------------
def benchmark_factor_contribution(df):
    print("\n--- Benchmark: Factor Contribution Waterfall ---")

    if "pref_balanced" not in df.columns:
        print("  [SKIP] Missing pref_balanced")
        return

    weights = load_profile_weights("balanced", project_root=PROJECT_ROOT, as_column_names=True)
    top20 = df.nlargest(20, "pref_balanced")

    rows = []
    for factor, weight in weights.items():
        if factor not in df.columns:
            continue
        top_avg = top20[factor].mean()
        universe_avg = df[factor].mean()
        delta = top_avg - universe_avg
        rows.append({
            "factor": factor,
            "profile_weight": weight,
            "top20_avg": top_avg,
            "universe_avg": universe_avg,
            "delta_vs_universe": delta,
            "factor_contribution": weight * delta,
        })

    if not rows:
        print("  [SKIP] No factor columns available for contribution analysis")
        return

    result = pd.DataFrame(rows).sort_values("factor_contribution", ascending=False)
    save_table(result, "benchmark_factor_contribution.csv", index=False)
    print("  [OK] Saved benchmark_factor_contribution.csv")
    return result


def main():
    print("=" * 70)
    print("STEP 06: BENCHMARK INDEX COMPARISON")
    print("=" * 70)

    df = load_data()
    # Mo2 DISCLOSURE: This analysis uses a contemporaneous universe as of the
    # data collection date. Companies that were delisted, merged, or went
    # bankrupt prior to this date are excluded (survivorship bias).
    # Historical returns for surviving companies overstate true investable
    # returns by an estimated 2-5% per annum (Elton, Gruber & Blake, 1996).
    n_companies = len(df)
    print(f"\n  [Mo2] Universe: {n_companies} companies (survivorship-biased contemporaneous sample)")
    sector_composition(df)
    score_comparison(df)
    simulated_performance(df)
    us_vs_india(df)
    benchmark_summary(df)
    multi_horizon_comparison(df)
    alpha_beta_analysis(df)
    equal_vs_value_weighted(df)
    factor_validity_clean(df)
    composite_ex_market_evaluation(df)
    esg_premium_analysis(df)
    factor_return_attribution(df)
    esg_quintile_performance(df)
    benchmark_factor_contribution(df)

    print(f"\n[DONE] Benchmark comparison complete. Results in {TABLES}/")
    print("Next: python scripts/07_visualizations.py")


if __name__ == "__main__":
    main()
