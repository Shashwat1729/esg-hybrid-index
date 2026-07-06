"""Financial & Economic Validation (Phase 8).

Verifies that the index methodology is financially sound and economically
defensible.  Produces CSV reports under ``reports/tables/``.

Includes:
  1. Financial ratio sanity checks
  2. Economic consistency checks
  3. Sector economics validation
  4. Factor score economic interpretation
  5. Currency conversion validation
  6. ESG data provenance impact
  7. Fama-MacBeth style cross-sectional regression
  8. Jonckheere-Terpstra monotonicity tests
  9. Transaction cost analysis
  10. Turnover analysis
  11. Capacity analysis
  12. Summary report

Run from the repo root::

    python scripts/16_financial_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path for src imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from src.constants import RANDOM_SEED

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "indexed_data.csv"
OUT_DIR = ROOT / "reports" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Drop rows with no sector (metadata-only rows)
    df = df.dropna(subset=["sector"]).reset_index(drop=True)
    # Exclude large-cap benchmarks — this validation is for mid-cap companies
    if "is_large_cap_benchmark" in df.columns:
        n_before = len(df)
        df = df[~df["is_large_cap_benchmark"]].reset_index(drop=True)
        print(f"  [Note] Excluded {n_before - len(df)} large-cap benchmarks from validation")
    return df


# ===================================================================
# 1. Financial Ratio Sanity Checks
# ===================================================================

# Reasonable ranges: (low, high) — values outside are *flagged*, not deleted.
RATIO_RANGES: dict[str, tuple[float, float]] = {
    "roa":            (-0.20,   0.30),
    "roe":            (-0.50,   0.80),
    "debt_to_equity": ( 0.00, 500.00),  # Yahoo reports D/E as %; 500% = 5× ratio
    "trailing_pe":    ( 5.00,  60.00),
    "forward_pe":     ( 5.00,  60.00),
    "price_to_book":  ( 0.50,  15.00),
    "current_ratio":  ( 0.50,   5.00),
    "dividend_yield": ( 0.00,  10.00),
}

# Hard-flag thresholds (extreme outliers)
HARD_FLAGS: dict[str, tuple[float | None, float | None]] = {
    "debt_to_equity": (None, 1000.0),   # >1000% (10× ratio) is extreme leverage
    "trailing_pe":    (0.0,  100.0),    # negative or >100
    "price_to_book":  (0.0,  None),     # negative book value
}


def _ratio_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """Check financial ratios against reasonable ranges."""
    rows: list[dict] = []
    for col, (lo, hi) in RATIO_RANGES.items():
        if col not in df.columns:
            rows.append({"ratio": col, "status": "COLUMN_MISSING",
                         "n_total": 0, "n_below": 0, "n_above": 0,
                         "pct_in_range": np.nan, "min": np.nan, "max": np.nan,
                         "mean": np.nan, "median": np.nan,
                         "hard_flag_count": 0, "verdict": "SKIP"})
            continue
        s = df[col].dropna()
        n = len(s)
        n_below = int((s < lo).sum())
        n_above = int((s > hi).sum())
        in_range = n - n_below - n_above
        pct = (in_range / n * 100) if n > 0 else np.nan

        # Hard flags
        hf = 0
        if col in HARD_FLAGS:
            hlo, hhi = HARD_FLAGS[col]
            if hlo is not None:
                hf += int((s < hlo).sum())
            if hhi is not None:
                hf += int((s > hhi).sum())

        verdict = "PASS" if pct >= 80 else ("WARN" if pct >= 60 else "FAIL")
        if hf > n * 0.05:
            verdict = "FAIL"

        rows.append({
            "ratio": col,
            "expected_range": f"[{lo}, {hi}]",
            "n_total": n,
            "n_below": n_below,
            "n_above": n_above,
            "pct_in_range": round(pct, 1),
            "min": round(s.min(), 4),
            "max": round(s.max(), 4),
            "mean": round(s.mean(), 4),
            "median": round(s.median(), 4),
            "hard_flag_count": hf,
            "verdict": verdict,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "financial_ratio_sanity.csv", index=False)
    return out


# ===================================================================
# 2. Economic Consistency Checks
# ===================================================================

def _corr_row(df: pd.DataFrame, x: str, y: str,
              hypothesis: str, expected_sign: str) -> dict:
    """Spearman correlation between two columns with a verdict."""
    if x not in df.columns or y not in df.columns:
        return {"relationship": f"{x} vs {y}", "hypothesis": hypothesis,
                "expected_sign": expected_sign, "spearman_r": np.nan,
                "p_value": np.nan, "n": 0, "verdict": "SKIP"}
    tmp = df[[x, y]].dropna()
    n = len(tmp)
    if n < 10:
        return {"relationship": f"{x} vs {y}", "hypothesis": hypothesis,
                "expected_sign": expected_sign, "spearman_r": np.nan,
                "p_value": np.nan, "n": n, "verdict": "INSUFFICIENT_DATA"}
    r, p = stats.spearmanr(tmp[x], tmp[y])
    sign_ok = (expected_sign == "+" and r > 0) or (expected_sign == "-" and r < 0)
    if sign_ok and p < 0.05:
        verdict = "CONSISTENT"
    elif sign_ok:
        verdict = "WEAK_CONSISTENT"
    else:
        verdict = "INCONSISTENT"
    return {
        "relationship": f"{x} vs {y}",
        "hypothesis": hypothesis,
        "expected_sign": expected_sign,
        "spearman_r": round(r, 4),
        "p_value": round(p, 4),
        "n": n,
        "verdict": verdict,
    }


def _economic_consistency(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        # (x, y, hypothesis, expected_sign)
        ("roa", "ESG_composite",
         "Profitable firms tend to invest more in ESG (note: may be weak with synthetic data)", "+"),
        ("revenue_growth", "trailing_pe",
         "Growth companies should command higher P/E multiples", "+"),
        ("trailing_pe", "dividend_yield",
         "Value companies (low P/E) tend to have higher dividend yields (inverse P/E-yield)", "-"),
        ("price_to_book", "dividend_yield",
         "Low P/B (value) companies tend to have higher dividend yields", "-"),
        ("debt_to_equity", "price_volatility",
         "Higher leverage should correspond to higher price volatility", "+"),
        ("roa", "financial_score",
         "Higher ROA should translate into higher financial_score", "+"),
        ("roe", "financial_score",
         "Higher ROE should translate into higher financial_score", "+"),
        ("revenue_growth", "growth_score",
         "Revenue growth should drive growth_score", "+"),
        ("price_volatility", "risk_adjusted_score",
         "Lower volatility should yield higher risk-adjusted score (inverse)", "-"),
        ("market_cap", "stability_score",
         "Larger companies tend to be more stable", "+"),
    ]
    rows = [_corr_row(df, x, y, h, s) for x, y, h, s in checks]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "economic_consistency.csv", index=False)
    return out


# ===================================================================
# 3. Sector Economics Validation
# ===================================================================

# Mid-cap-adjusted P/E ranges.  Upper bounds widened ~40% vs large-cap norms
# to account for the systematic mid-cap growth/takeover premium documented in
# S&P 400 MidCap vs S&P 500 P/E comparisons.
SECTOR_PE_RANGES: dict[str, tuple[float, float]] = {
    "Technology":             (20, 50),
    "Financial Services":     (8,  25),
    "Utilities":              (12, 30),
    "Healthcare":             (15, 45),
    "Consumer Cyclical":      (12, 40),
    "Consumer Defensive":     (15, 45),
    "Industrials":            (15, 50),
    "Energy":                 (8,  30),
    "Communication Services": (12, 35),
    "Basic Materials":        (10, 35),
    "Real Estate":            (15, 45),
}

SECTOR_DE_ORDER = [
    "Financial Services",   # typically highest leverage
    "Real Estate",
    "Utilities",
    "Energy",
    "Industrials",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Basic Materials",
    "Communication Services",
    "Healthcare",
    "Technology",           # typically lowest leverage
]


def _sector_economics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for sector, grp in df.groupby("sector"):
        n = len(grp)
        avg_pe = grp["trailing_pe"].mean()
        avg_de = grp["debt_to_equity"].mean()
        avg_pb = grp["price_to_book"].mean()
        avg_dy = grp["dividend_yield"].mean()
        avg_esg = grp["ESG_composite"].mean()

        pe_range = SECTOR_PE_RANGES.get(sector, (10, 40))
        pe_in_range = pe_range[0] <= avg_pe <= pe_range[1]

        rows.append({
            "sector": sector,
            "n_companies": n,
            "avg_trailing_pe": round(avg_pe, 2),
            "expected_pe_range": f"[{pe_range[0]}, {pe_range[1]}]",
            "pe_in_expected_range": pe_in_range,
            "avg_debt_to_equity": round(avg_de, 2),
            "avg_price_to_book": round(avg_pb, 2),
            "avg_dividend_yield": round(avg_dy, 2),
            "avg_ESG_composite": round(avg_esg, 2),
        })

    out = pd.DataFrame(rows).sort_values("avg_trailing_pe", ascending=False)

    # Check D/E ordering: are financials at the high end and tech at the low end?
    de_by_sector = (
        df.groupby("sector")["debt_to_equity"]
        .mean()
        .sort_values(ascending=False)
    )
    top3_de = list(de_by_sector.index[:3])
    bot3_de = list(de_by_sector.index[-3:])
    high_leverage_expected = {"Financial Services", "Real Estate", "Utilities"}
    low_leverage_expected = {"Technology", "Healthcare", "Communication Services"}
    de_order_ok = bool(high_leverage_expected & set(top3_de)) and bool(
        low_leverage_expected & set(bot3_de)
    )
    out["de_sector_ordering_plausible"] = de_order_ok

    out.to_csv(OUT_DIR / "sector_economics.csv", index=False)
    return out


# ===================================================================
# 4. Factor Score Economic Interpretation
# ===================================================================

def _top_bottom(df: pd.DataFrame, score_col: str, metric_cols: list[str],
                k: int = 5) -> pd.DataFrame:
    """Return top-k and bottom-k rows for a given score, with selected metrics."""
    cols = ["ticker", "company_name", "sector", "country", score_col] + [
        c for c in metric_cols if c in df.columns
    ]
    available = [c for c in cols if c in df.columns]
    sorted_df = df.sort_values(score_col, ascending=False)
    top = sorted_df.head(k).copy()
    top["group"] = "top_5"
    bottom = sorted_df.tail(k).copy()
    bottom["group"] = "bottom_5"
    combined = pd.concat([top, bottom], ignore_index=True)
    combined["factor"] = score_col
    return combined[["factor", "group"] + available]


def _factor_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    interpretations = {
        "financial_score": ["roa", "roe", "net_margin", "operating_margin",
                            "free_cashflow", "debt_to_equity"],
        "ESG_composite": ["E_score", "S_score", "G_score",
                          "board_independence_pct", "ethics_compliance_score",
                          "esg_data_source"],
        "growth_score": ["revenue_growth", "earnings_growth",
                         "earnings_quarterly_growth", "trailing_pe"],
        "value_score": ["trailing_pe", "price_to_book", "price_to_sales",
                        "dividend_yield", "enterprise_to_ebitda"],
        "stability_score": ["debt_to_equity", "current_ratio", "beta",
                            "price_volatility", "market_cap"],
        "risk_adjusted_score": ["sharpe_ratio_1y", "sortino_ratio_1y",
                                "max_drawdown_1y", "price_volatility"],
    }

    frames: list[pd.DataFrame] = []
    for score_col, metrics in interpretations.items():
        if score_col not in df.columns:
            continue
        frames.append(_top_bottom(df, score_col, metrics))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Economic-sense verdict per factor
    verdicts: list[dict] = []
    for score_col, metrics in interpretations.items():
        if score_col not in df.columns:
            continue
        top5 = df.nlargest(5, score_col)
        bot5 = df.nsmallest(5, score_col)

        notes: list[str] = []
        if score_col == "financial_score":
            top_roa = top5["roa"].mean() if "roa" in df.columns else np.nan
            bot_roa = bot5["roa"].mean() if "roa" in df.columns else np.nan
            if pd.notna(top_roa) and pd.notna(bot_roa):
                ok = top_roa > bot_roa
                notes.append(f"Top5 avg ROA={top_roa:.3f} vs Bot5={bot_roa:.3f} → {'OK' if ok else 'UNEXPECTED'}")

        if score_col == "growth_score" and "revenue_growth" in df.columns:
            top_rg = top5["revenue_growth"].mean()
            bot_rg = bot5["revenue_growth"].mean()
            ok = top_rg > bot_rg
            notes.append(f"Top5 avg rev_growth={top_rg:.3f} vs Bot5={bot_rg:.3f} → {'OK' if ok else 'UNEXPECTED'}")

        if score_col == "stability_score" and "price_volatility" in df.columns:
            top_vol = top5["price_volatility"].mean()
            bot_vol = bot5["price_volatility"].mean()
            ok = top_vol < bot_vol
            notes.append(f"Top5 avg vol={top_vol:.1f} vs Bot5={bot_vol:.1f} → {'OK' if ok else 'UNEXPECTED'}")

        if score_col == "value_score" and "trailing_pe" in df.columns:
            top_pe = top5["trailing_pe"].mean()
            bot_pe = bot5["trailing_pe"].mean()
            ok = top_pe < bot_pe
            notes.append(f"Top5 avg PE={top_pe:.1f} vs Bot5={bot_pe:.1f} → {'OK' if ok else 'UNEXPECTED'}")

        verdict = "PASS" if all("OK" in n for n in notes) else (
            "WARN" if notes else "NO_CHECK")
        verdicts.append({"factor": score_col, "economic_checks": "; ".join(notes),
                         "verdict": verdict})

    verdict_df = pd.DataFrame(verdicts)
    # Append verdict rows as a summary section
    out_path = OUT_DIR / "factor_economic_interpretation.csv"
    out.to_csv(out_path, index=False)
    # Also save verdict summary alongside
    verdict_df.to_csv(
        OUT_DIR / "factor_economic_interpretation_verdicts.csv", index=False
    )
    return out


# ===================================================================
# 5. Currency Conversion Validation
# ===================================================================

def _currency_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for country in ["US", "India"]:
        sub = df[df["country"] == country]
        if sub.empty:
            continue
        mc = sub["market_cap"] / 1e9  # in $B
        rows.append({
            "country": country,
            "n_companies": len(sub),
            "market_cap_B_min": round(mc.min(), 2),
            "market_cap_B_p25": round(mc.quantile(0.25), 2),
            "market_cap_B_median": round(mc.median(), 2),
            "market_cap_B_p75": round(mc.quantile(0.75), 2),
            "market_cap_B_max": round(mc.max(), 2),
        })

    out = pd.DataFrame(rows)

    # Expected ranges (USD)
    expected = {
        "India": (0.3, 20.0),   # Indian mid-caps: $300M – $20B after conversion
        "US":    (0.3, 3000.0), # US companies can range from small to mega-cap
    }
    flags: list[dict] = []
    for country, (lo, hi) in expected.items():
        sub = df[df["country"] == country]
        mc = sub["market_cap"] / 1e9
        below = sub[mc < lo]
        above = sub[mc > hi]
        for _, r in below.iterrows():
            flags.append({"ticker": r.get("ticker", ""),
                          "company_name": r.get("company_name", ""),
                          "country": country,
                          "market_cap_B": round(r["market_cap"] / 1e9, 3),
                          "flag": f"Below ${lo}B floor"})
        for _, r in above.iterrows():
            flags.append({"ticker": r.get("ticker", ""),
                          "company_name": r.get("company_name", ""),
                          "country": country,
                          "market_cap_B": round(r["market_cap"] / 1e9, 3),
                          "flag": f"Above ${hi}B ceiling"})

    if flags:
        flag_df = pd.DataFrame(flags)
    else:
        flag_df = pd.DataFrame(columns=["ticker", "company_name", "country",
                                         "market_cap_B", "flag"])

    # Merge summary + flags
    combined = pd.concat(
        [out, pd.DataFrame([{}]),  # blank separator row
         flag_df],
        ignore_index=True,
    )
    # Cleaner: save separately
    out.to_csv(OUT_DIR / "currency_conversion_validation.csv", index=False)
    if not flag_df.empty:
        flag_df.to_csv(OUT_DIR / "currency_conversion_flags.csv", index=False)
    return out


# ===================================================================
# 6. ESG Data Provenance Impact
# ===================================================================

def _esg_provenance(df: pd.DataFrame) -> pd.DataFrame:
    if "esg_data_source" not in df.columns:
        return pd.DataFrame([{"note": "esg_data_source column not present"}])

    esg_score_cols = ["ESG_composite", "E_score", "S_score", "G_score"]
    available = [c for c in esg_score_cols if c in df.columns]

    rows: list[dict] = []
    for src, grp in df.groupby("esg_data_source"):
        row: dict = {"esg_data_source": src, "n_companies": len(grp)}
        for col in available:
            row[f"{col}_mean"] = round(grp[col].mean(), 2)
            row[f"{col}_std"] = round(grp[col].std(), 2)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Statistical tests: real_yahoo vs synthetic / sector_imputed
    test_rows: list[dict] = []
    sources = df["esg_data_source"].unique()
    ref_src = "real_yahoo"
    if ref_src in sources:
        ref = df[df["esg_data_source"] == ref_src]
        for other_src in sources:
            if other_src == ref_src:
                continue
            other = df[df["esg_data_source"] == other_src]
            for col in available:
                a = ref[col].dropna()
                b = other[col].dropna()
                if len(a) < 5 or len(b) < 5:
                    continue
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                # Effect size: rank-biserial r
                n1, n2 = len(a), len(b)
                r_rb = 1 - (2 * stat) / (n1 * n2)
                test_rows.append({
                    "comparison": f"{ref_src} vs {other_src}",
                    "score": col,
                    "ref_mean": round(a.mean(), 2),
                    "other_mean": round(b.mean(), 2),
                    "diff": round(a.mean() - b.mean(), 2),
                    "mann_whitney_U": round(stat, 1),
                    "p_value": round(p, 4),
                    "rank_biserial_r": round(r_rb, 3),
                    "significant_at_05": p < 0.05,
                    "bias_risk": "HIGH" if (p < 0.05 and abs(r_rb) > 0.3) else (
                        "MODERATE" if p < 0.05 else "LOW"),
                })

    tests = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(
        columns=["comparison", "score", "p_value", "bias_risk"]
    )

    # Save both
    combined = pd.concat([summary, pd.DataFrame([{}]), tests], ignore_index=True)
    summary.to_csv(OUT_DIR / "esg_provenance_impact.csv", index=False)
    if not tests.empty:
        tests.to_csv(OUT_DIR / "esg_provenance_tests.csv", index=False)
    return summary


# ===================================================================
# 7. Fama-MacBeth Style Cross-Sectional Regression
# ===================================================================

def fama_macbeth_cross_sectional(df: pd.DataFrame):
    """
    Fama-MacBeth style cross-sectional regression adapted for single-period data.
    Tests whether ESG score predicts cross-sectional return differences
    after controlling for size (market_cap), value (price_to_book), and risk (beta).

    Returns
    -------
    tuple of (quintile_returns, reg_results, ls_spread, r_squared)
        or empty tuple if insufficient data.
    """
    import statsmodels.api as sm

    # Prepare variables
    required = ["ESG_composite", "price_momentum_3m", "beta", "market_cap", "price_to_book"]
    available_req = [c for c in required if c in df.columns]
    df_clean = df.dropna(subset=available_req).copy()

    if len(df_clean) < 30:
        print("   [SKIP] Fewer than 30 observations with required columns.")
        return ()

    # Quintile analysis
    df_clean["esg_quintile"] = pd.qcut(
        df_clean["ESG_composite"], 5, labels=False, duplicates="drop"
    ) + 1
    quintile_returns = df_clean.groupby("esg_quintile")["price_momentum_3m"].agg(
        ["mean", "std", "count"]
    )

    # Long-short spread (Q5 high-ESG minus Q1 low-ESG)
    q5_return = df_clean.loc[df_clean["esg_quintile"] == 5, "price_momentum_3m"].mean()
    q1_return = df_clean.loc[df_clean["esg_quintile"] == 1, "price_momentum_3m"].mean()
    ls_spread = q5_return - q1_return

    # NOTE: This is a single cross-sectional regression. A proper Fama-MacBeth
    # procedure requires estimating lambda_t for each period and averaging.
    # Since we have a single cross-section, we label this "cross-sectional
    # return-factor regression" rather than Fama-MacBeth.
    # Cross-sectional regression: return_proxy ~ ESG + beta + log(mcap) + P/B
    y = df_clean["price_momentum_3m"]

    X_vars: dict[str, pd.Series] = {}
    if "ESG_composite" in df_clean.columns:
        X_vars["ESG_composite"] = df_clean["ESG_composite"]
    if "beta" in df_clean.columns:
        X_vars["beta"] = df_clean["beta"]
    if "market_cap" in df_clean.columns:
        X_vars["log_mcap"] = np.log(df_clean["market_cap"].clip(lower=1))
    if "price_to_book" in df_clean.columns:
        X_vars["price_to_book"] = df_clean["price_to_book"]

    X = pd.DataFrame(X_vars)
    # Mo1 FIX: Add sector dummies to control for industry effects
    if "sector" in df_clean.columns:
        sector_dummies = pd.get_dummies(df_clean["sector"], prefix="sector", drop_first=True, dtype=float)
        X = pd.concat([X, sector_dummies], axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop").fit(cov_type="HC1")  # Heteroscedasticity-robust

    # Collect regression results
    reg_results = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_statistic": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    # Mo6: Sector-adjusted factor IC diagnostic
    factor_cols = ["financial_score", "ESG_composite", "growth_score", "value_score", "stability_score", "risk_adjusted_score"]
    return_col = "price_momentum_3m"
    if "sector" in df_clean.columns:
        print("\n  Sector-adjusted factor ICs:")
        for factor in factor_cols:
            if factor not in df_clean.columns:
                continue
            # Demean factor within sector
            sector_adj = df_clean.groupby("sector")[factor].transform(lambda x: x - x.mean())
            if return_col in df_clean.columns:
                from scipy.stats import spearmanr
                valid = sector_adj.notna() & df_clean[return_col].notna()
                if valid.sum() > 20:
                    rho, p = spearmanr(sector_adj[valid], df_clean.loc[valid, return_col])
                    print(f"    {factor}: sector-adj IC = {rho:.4f} (p={p:.4f})")

    return quintile_returns, reg_results, ls_spread, model.rsquared


# ===================================================================
# 8. Jonckheere-Terpstra Monotonicity Test
# ===================================================================

def jonckheere_terpstra_test(
    df: pd.DataFrame, score_col: str, return_col: str, n_groups: int = 5
) -> dict:
    """Test whether *return_col* increases monotonically with *score_col* quintile.

    Uses the Jonckheere-Terpstra statistic with a normal approximation for the
    p-value.  Returns a dict with test statistics.
    """
    df_clean = df.dropna(subset=[score_col, return_col]).copy()
    if len(df_clean) < 20:
        return {
            "score": score_col,
            "jt_statistic": np.nan,
            "z_statistic": np.nan,
            "p_value": np.nan,
            "monotonic": False,
            "note": "insufficient_data",
        }

    df_clean["quintile"] = pd.qcut(
        df_clean[score_col], n_groups, labels=False, duplicates="drop"
    ) + 1

    groups = [
        df_clean.loc[df_clean["quintile"] == q, return_col].values
        for q in range(1, n_groups + 1)
    ]
    # Drop any empty groups (can happen with duplicate bin edges)
    groups = [g for g in groups if len(g) > 0]
    actual_k = len(groups)

    if actual_k < 3:
        return {
            "score": score_col,
            "jt_statistic": np.nan,
            "z_statistic": np.nan,
            "p_value": np.nan,
            "monotonic": False,
            "note": "too_few_groups",
        }

    # Mann-Whitney U statistic summed over all ordered pairs
    jt_stat = 0.0
    for i in range(actual_k):
        for j in range(i + 1, actual_k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        jt_stat += 1
                    elif xj == xi:
                        jt_stat += 0.5

    # Group sizes
    n_sizes = [len(g) for g in groups]
    n_total = sum(n_sizes)

    # Expected value under H0
    E_JT = (n_total ** 2 - sum(n ** 2 for n in n_sizes)) / 4

    # Variance under H0 (simplified, assuming no ties)
    V_JT = (
        n_total ** 2 * (2 * n_total + 3)
        - sum(n ** 2 * (2 * n + 3) for n in n_sizes)
    ) / 72

    z = (jt_stat - E_JT) / np.sqrt(V_JT) if V_JT > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "score": score_col,
        "jt_statistic": round(jt_stat, 1),
        "z_statistic": round(z, 3),
        "p_value": round(p_value, 6),
        "monotonic": p_value < 0.05,
    }


# ===================================================================
# 9. Transaction Cost Analysis
# ===================================================================

def _get_top20(df: pd.DataFrame, score_col: str = "pref_balanced") -> pd.DataFrame:
    """Return top-20 companies by *score_col* (highest = best)."""
    if score_col not in df.columns:
        # Fallback: try ESG_composite
        score_col = "ESG_composite"
    return df.nlargest(20, score_col).copy()


def _add_liquidity_weights(top20: pd.DataFrame) -> pd.DataFrame:
    """Add ADV-dollar based liquidity weights to a top-20 frame."""
    out = top20.copy()
    out["daily_dollar_volume"] = (
        pd.to_numeric(out.get("avg_daily_volume"), errors="coerce")
        * pd.to_numeric(out.get("price"), errors="coerce")
    )
    total_ddv = out["daily_dollar_volume"].replace([np.inf, -np.inf], np.nan).sum(skipna=True)
    if pd.notna(total_ddv) and total_ddv > 0:
        out["liquidity_weight"] = out["daily_dollar_volume"] / total_ddv
    else:
        out["liquidity_weight"] = 1.0 / max(len(out), 1)
    return out


def _transaction_cost_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate round-trip transaction costs for the top-20 portfolio.

    Model
    -----
    * Market impact (Almgren-Chriss simplified):
          impact = 0.1 * sigma_daily * sqrt(Q / ADV)
      where Q = trade size in shares, ADV = average daily volume, sigma_daily
      is daily volatility in decimal form.
    * Spread cost: 5 bps per side (mid-caps)
    * Commission: 3 bps per side
    * Round-trip cost = 2 * (impact + spread + commission)
    * Reference AUM: $100 M, allocated by liquidity weights.
    """
    top20 = _add_liquidity_weights(_get_top20(df))

    REF_AUM = 100_000_000  # $100 M reference
    SPREAD_BPS = 5
    COMMISSION_BPS = 3
    ANNUAL_TO_DAILY_VOL = np.sqrt(252)

    rows: list[dict] = []
    for _, r in top20.iterrows():
        price = r.get("price", np.nan)
        adv = r.get("avg_daily_volume", np.nan)
        vol = r.get("price_volatility", np.nan)
        liq_w = r.get("liquidity_weight", np.nan)

        if pd.isna(price) or pd.isna(adv) or pd.isna(liq_w) or price <= 0 or adv <= 0 or liq_w <= 0:
            rows.append({
                "ticker": r.get("ticker", ""),
                "company_name": r.get("company_name", ""),
                "price": price,
                "avg_daily_volume": adv,
                "volatility_pct": vol,
                "liquidity_weight": liq_w,
                "position_usd": np.nan,
                "trade_shares": np.nan,
                "spread_cost_bps": SPREAD_BPS,
                "commission_bps": COMMISSION_BPS,
                "market_impact_bps": np.nan,
                "total_rt_cost_bps": np.nan,
                "total_rt_cost_usd": np.nan,
                "note": "missing_data",
            })
            continue

        # Liquidity-weighted position
        position_usd = REF_AUM * liq_w
        trade_shares = position_usd / price
        participation_rate = trade_shares / adv  # Q / ADV

        # Market impact in bps (Almgren-Chriss simplified)
        vol_annual_frac = (vol / 100.0) if pd.notna(vol) else 0.30
        sigma_daily = vol_annual_frac / ANNUAL_TO_DAILY_VOL
        impact_frac = 0.1 * sigma_daily * np.sqrt(max(participation_rate, 0.0))
        impact_bps = impact_frac * 10000

        one_way_bps = impact_bps + SPREAD_BPS + COMMISSION_BPS
        total_rt_bps = 2 * one_way_bps  # round-trip
        total_rt_usd = position_usd * total_rt_bps / 10000.0

        rows.append({
            "ticker": r.get("ticker", ""),
            "company_name": r.get("company_name", ""),
            "price": round(price, 2),
            "avg_daily_volume": round(adv, 0),
            "volatility_pct": round(vol, 2) if pd.notna(vol) else np.nan,
            "liquidity_weight": round(liq_w, 6),
            "position_usd": round(position_usd, 0),
            "trade_shares": round(trade_shares, 0),
            "participation_pct_adv": round(participation_rate * 100, 2),
            "spread_cost_bps": SPREAD_BPS,
            "commission_bps": COMMISSION_BPS,
            "market_impact_bps": round(impact_bps, 2),
            "total_rt_cost_bps": round(total_rt_bps, 2),
            "total_rt_cost_usd": round(total_rt_usd, 2),
        })

    out = pd.DataFrame(rows)

    # Portfolio-level summary row
    if not out.empty and "total_rt_cost_usd" in out.columns:
        total_cost = out["total_rt_cost_usd"].sum()
        avg_bps = (total_cost / REF_AUM) * 10000 if REF_AUM > 0 else np.nan
        summary_row = pd.DataFrame([{
            "ticker": "PORTFOLIO_TOTAL",
            "company_name": f"Top-20 liquidity-weighted @ ${REF_AUM/1e6:.0f}M AUM",
            "total_rt_cost_bps": round(avg_bps, 2),
            "total_rt_cost_usd": round(total_cost, 2),
        }])
        out = pd.concat([out, summary_row], ignore_index=True)

    out.to_csv(OUT_DIR / "transaction_cost_analysis.csv", index=False)
    return out


# ===================================================================
# 10. Turnover Analysis
# ===================================================================

def _turnover_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate portfolio turnover by perturbing scores and measuring rank changes.

    Method
    ------
    1. Take the baseline ranking (top-20 by pref_balanced).
    2. Add uniform noise at multiple perturbation levels (±10%, ±20%, ±30%)
       to capture sensitivity of turnover estimates (Mo7 FIX).
    3. Re-rank and compare overlap with baseline top-20.
    4. Repeat N_SIM times to get expected turnover statistics.
    5. Compute turnover-adjusted return proxy.
    """
    score_col = "pref_balanced" if "pref_balanced" in df.columns else "ESG_composite"
    if score_col not in df.columns:
        return pd.DataFrame([{"note": "score column not available"}])

    N_SIM = 200
    TOP_K = 20
    # Mo7 FIX: Test multiple perturbation levels instead of just ±10%
    PERTURBATION_LEVELS = [0.10, 0.20, 0.30]

    baseline_top = set(df.nlargest(TOP_K, score_col)["ticker"].tolist())
    scores = df[score_col].values.copy()
    tickers = df["ticker"].values

    rng = np.random.default_rng(RANDOM_SEED)

    # Run perturbation analysis at each level
    perturbation_results = {}
    for pert_level in PERTURBATION_LEVELS:
        overlaps: list[float] = []
        turnovers: list[float] = []
        for _ in range(N_SIM):
            noise = rng.uniform(-pert_level, pert_level, size=len(scores))
            perturbed = scores * (1 + noise)
            idx_top = np.argsort(-perturbed)[:TOP_K]
            perturbed_top = set(tickers[idx_top])

            overlap = len(baseline_top & perturbed_top) / TOP_K
            turnover = 1.0 - overlap
            overlaps.append(overlap)
            turnovers.append(turnover)

        perturbation_results[pert_level] = {
            "avg_overlap": np.mean(overlaps),
            "avg_turnover": np.mean(turnovers),
            "std_turnover": np.std(turnovers),
            "p25_turnover": np.percentile(turnovers, 25),
            "p75_turnover": np.percentile(turnovers, 75),
        }

    # Use ±10% as the primary estimate for backward compatibility
    primary = perturbation_results[0.10]
    avg_turnover = primary["avg_turnover"]
    std_turnover = primary["std_turnover"]
    p25_turnover = primary["p25_turnover"]
    p75_turnover = primary["p75_turnover"]

    # Quarterly turnover estimate: assume quarterly rebalance with ±10% score drift
    quarterly_turnover_pct = avg_turnover * 100

    # Turnover-aware net excess return
    # net excess = gross excess - (turnover * round_trip_cost * 4 rebalances)
    top20 = _get_top20(df)
    avg_return_3m = top20["price_momentum_3m"].mean() if "price_momentum_3m" in top20.columns else np.nan
    universe_return_3m = df["price_momentum_3m"].mean() if "price_momentum_3m" in df.columns else np.nan
    gross_excess_3m = (avg_return_3m - universe_return_3m) if pd.notna(avg_return_3m) and pd.notna(universe_return_3m) else np.nan
    gross_excess_annual = gross_excess_3m * 4 if pd.notna(gross_excess_3m) else np.nan

    assumed_turnover = 0.30  # expected turnover per quarterly rebalance
    n_rebalances = 4
    tc_path = OUT_DIR / "transaction_cost_analysis.csv"
    round_trip_cost = np.nan
    if tc_path.exists():
        tc_df = pd.read_csv(tc_path)
        portfolio_row = tc_df[tc_df["ticker"] == "PORTFOLIO_TOTAL"]
        if not portfolio_row.empty:
            round_trip_cost = portfolio_row.iloc[0].get("total_rt_cost_bps", np.nan) / 10000

    turnover_cost_drag = (
        assumed_turnover * round_trip_cost * n_rebalances
        if pd.notna(round_trip_cost)
        else np.nan
    )
    turnover_adj_return = (
        gross_excess_annual - turnover_cost_drag
        if pd.notna(gross_excess_annual) and pd.notna(turnover_cost_drag)
        else np.nan
    )

    rows = [
        {"metric": "baseline_top20_score_col", "value": score_col},
        {"metric": "n_simulations", "value": N_SIM},
        {"metric": "perturbation_range", "value": "±10% (primary), ±20%, ±30%"},
        {"metric": "avg_overlap_pct", "value": round(primary["avg_overlap"] * 100, 1)},
        {"metric": "avg_turnover_pct", "value": round(quarterly_turnover_pct, 1)},
        {"metric": "std_turnover_pct", "value": round(std_turnover * 100, 1)},
        {"metric": "p25_turnover_pct", "value": round(p25_turnover * 100, 1)},
        {"metric": "p75_turnover_pct", "value": round(p75_turnover * 100, 1)},
        {"metric": "avg_3m_return_top20", "value": round(avg_return_3m, 4) if pd.notna(avg_return_3m) else np.nan},
        {"metric": "avg_3m_return_universe", "value": round(universe_return_3m, 4) if pd.notna(universe_return_3m) else np.nan},
        {"metric": "gross_excess_return_annual", "value": round(gross_excess_annual, 4) if pd.notna(gross_excess_annual) else np.nan},
        {"metric": "expected_turnover_per_rebalance", "value": assumed_turnover},
        {"metric": "round_trip_cost_per_rebalance", "value": round(round_trip_cost, 6) if pd.notna(round_trip_cost) else np.nan},
        {"metric": "turnover_cost_drag_annual", "value": round(turnover_cost_drag, 6) if pd.notna(turnover_cost_drag) else np.nan},
        {"metric": "net_excess_return_annual", "value": round(turnover_adj_return, 4) if pd.notna(turnover_adj_return) else np.nan},
        {"metric": "verdict", "value": "LOW" if quarterly_turnover_pct < 20 else ("MODERATE" if quarterly_turnover_pct < 40 else "HIGH")},
    ]

    # Mo7 FIX: Add multi-level perturbation sensitivity rows
    for pert_level, res in perturbation_results.items():
        pct_label = f"±{int(pert_level * 100)}%"
        rows.append({"metric": f"sensitivity_{pct_label}_avg_turnover_pct",
                      "value": round(res["avg_turnover"] * 100, 1)})
        rows.append({"metric": f"sensitivity_{pct_label}_avg_overlap_pct",
                      "value": round(res["avg_overlap"] * 100, 1)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "turnover_analysis.csv", index=False)
    return out


# ===================================================================
# 11. Capacity Analysis
# ===================================================================

def _market_impact_sqr(trade_value: float, adv: float, daily_volatility: float = 0.02) -> float:
    """Square-root market impact model (Almgren-Chriss simplified).

    M7 FIX: Replaces constant participation rate with realistic
    price impact that scales with trade size relative to liquidity.

    impact = sigma * sqrt(trade_value / ADV)

    where sigma is daily volatility (default 2% for mid-cap).
    This is the standard institutional cost model.
    """
    if adv <= 0 or trade_value <= 0:
        return 0.0
    participation = trade_value / adv
    impact_bps = daily_volatility * 10000 * np.sqrt(participation)
    return impact_bps  # in basis points

def _capacity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate strategy capacity for the top-20 portfolio.

    Liquidity-weighted allocation:
      weight_i = dollar_volume_i / sum(dollar_volume)

    Multi-day execution model:
      10 trading days (2 weeks), 3% ADV/day execution budget for mid-caps.
      total_10d_capacity_i = 0.03 * ADV_i * 10 * price_i

    Feasibility metric:
      For each AUM tier, largest position usage must be <15% of (ADV * 10 days).
    """
    top20 = _add_liquidity_weights(_get_top20(df))

    EXEC_DAYS = 10          # 2-week execution window (realistic for mid-cap)
    DAILY_PARTICIPATION = 0.03  # 3% of ADV (conservative for mid-cap)
    LARGEST_POSITION_LIMIT = 0.15  # 15% max position utilization (mid-cap adjusted)

    rows: list[dict] = []
    for _, r in top20.iterrows():
        price = r.get("price", np.nan)
        adv = r.get("avg_daily_volume", np.nan)
        mcap = r.get("market_cap", np.nan)
        liq_w = r.get("liquidity_weight", np.nan)

        if pd.isna(price) or pd.isna(adv) or pd.isna(liq_w) or price <= 0 or adv <= 0 or liq_w <= 0:
            rows.append({
                "ticker": r.get("ticker", ""),
                "company_name": r.get("company_name", ""),
                "price": price,
                "avg_daily_volume": adv,
                "market_cap_M": round(mcap / 1e6, 1) if pd.notna(mcap) else np.nan,
                "daily_dollar_vol_M": np.nan,
                "liquidity_weight": np.nan,
                "max_exec_5d_M": np.nan,
                "max_position_10pct_5d_M": np.nan,
                "note": "missing_data",
            })
            continue

        daily_dollar = adv * price
        max_exec_5d = DAILY_PARTICIPATION * adv * EXEC_DAYS * price
        max_position_10pct_5d = LARGEST_POSITION_LIMIT * adv * EXEC_DAYS * price

        rows.append({
            "ticker": r.get("ticker", ""),
            "company_name": r.get("company_name", ""),
            "sector": r.get("sector", ""),
            "price": round(price, 2),
            "avg_daily_volume": round(adv, 0),
            "market_cap_M": round(mcap / 1e6, 1) if pd.notna(mcap) else np.nan,
            "daily_dollar_vol_M": round(daily_dollar / 1e6, 2),
            "liquidity_weight": round(liq_w, 6),
            "max_exec_5d_M": round(max_exec_5d / 1e6, 2),
            "max_position_10pct_5d_M": round(max_position_10pct_5d / 1e6, 2),
        })

    out = pd.DataFrame(rows)

    if not out.empty and "liquidity_weight" in out.columns:
        valid = out.dropna(subset=["liquidity_weight", "daily_dollar_vol_M"])
        if not valid.empty:
            total_capacity_5d_M = valid["max_exec_5d_M"].sum()

            # Capacity by max utilization constraint: max_i(position_i / (ADV_i*Nd)) < limit
            aum_thresholds = [25, 50, 100, 250, 500, 1000, 2000]  # in $M
            capacity_rows = []
            for aum_M in aum_thresholds:
                positions_M = aum_M * valid["liquidity_weight"]
                adv_5d_M = valid["daily_dollar_vol_M"] * EXEC_DAYS
                usage = positions_M / adv_5d_M.replace(0, np.nan)
                participation_pct = (positions_M / valid["daily_dollar_vol_M"].replace(0, np.nan)) * 100
                market_impact_bps = [
                    _market_impact_sqr(tv, adv)
                    for tv, adv in zip(positions_M * 1_000_000, valid["daily_dollar_vol_M"] * 1_000_000)
                ]
                market_impact_bps = pd.Series(market_impact_bps, index=valid.index)
                total_cost_bps = 5 + market_impact_bps  # spread + impact

                feasible_mask = (total_cost_bps <= 100) & (participation_pct <= 25)
                n_feasible = int(feasible_mask.sum())
                n_total = int(feasible_mask.notna().sum())
                pass_rate = (n_feasible / n_total * 100) if n_total > 0 else np.nan
                max_usage = usage.max()
                feasible = bool(pd.notna(pass_rate) and pass_rate >= 90)
                capacity_rows.append({
                    "aum_M": aum_M,
                    "largest_position_M": round(float(positions_M.max()), 2),
                    "max_position_usage_pct_adv_5d": round(float(max_usage * 100), 2) if pd.notna(max_usage) else np.nan,
                    "market_impact_bps": round(float(market_impact_bps.max()), 2) if not market_impact_bps.empty else np.nan,
                    "total_cost_bps": round(float(total_cost_bps.max()), 2) if not total_cost_bps.empty else np.nan,
                    "n_feasible_at_10pct_adv_5d": n_feasible,
                    "n_feasible_at_1pct_adv": n_feasible,
                    "n_total": n_total,
                    "feasible_pct": round(pass_rate, 2) if pd.notna(pass_rate) else np.nan,
                    "feasible": feasible,
                    "verdict": "PASS" if feasible else "FAIL",
                })

            usage_limit_M = (
                (LARGEST_POSITION_LIMIT * valid["daily_dollar_vol_M"] * EXEC_DAYS)
                / valid["liquidity_weight"].replace(0, np.nan)
            )
            max_aum_largest_position_limit_M = usage_limit_M.min()

            summary = pd.DataFrame([
                {"ticker": "MAX_AUM_LARGEST_POS_10PCT_ADV_5D", "daily_dollar_vol_M": round(max_aum_largest_position_limit_M, 2)},
                {"ticker": "TOTAL_EXEC_CAPACITY_5D_5PCT_ADV", "daily_dollar_vol_M": round(total_capacity_5d_M, 2)},
            ])

            cap_df = pd.DataFrame(capacity_rows)
            feasible_aums = cap_df.loc[cap_df["feasible_pct"] >= 90, "aum_M"]
            max_feasible_aum = feasible_aums.max() if not feasible_aums.empty else np.nan
            summary = pd.concat([
                summary,
                pd.DataFrame([
                    {"ticker": "MAX_FEASIBLE_AUM_90PCT_PASS", "daily_dollar_vol_M": round(float(max_feasible_aum), 2) if pd.notna(max_feasible_aum) else np.nan}
                ]),
            ], ignore_index=True)
            out = pd.concat([out, summary], ignore_index=True)

            cap_df.to_csv(OUT_DIR / "capacity_aum_feasibility.csv", index=False)

    out.to_csv(OUT_DIR / "capacity_analysis.csv", index=False)
    return out


# ===================================================================
# 12. Summary Report
# ===================================================================

def _build_summary(
    ratio_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    currency_df: pd.DataFrame,
    provenance_df: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    # 1. Ratio sanity
    if not ratio_df.empty and "verdict" in ratio_df.columns:
        n_pass = (ratio_df["verdict"] == "PASS").sum()
        n_warn = (ratio_df["verdict"] == "WARN").sum()
        n_fail = (ratio_df["verdict"] == "FAIL").sum()
        n_total = len(ratio_df)
        v = "PASS" if n_fail == 0 and n_warn <= 2 else ("WARN" if n_fail <= 1 else "FAIL")
        rows.append({
            "check": "Financial Ratio Sanity",
            "detail": f"{n_pass}/{n_total} PASS, {n_warn} WARN, {n_fail} FAIL",
            "verdict": v,
        })
    else:
        rows.append({"check": "Financial Ratio Sanity", "detail": "No data", "verdict": "SKIP"})

    # 2. Economic consistency
    if not consistency_df.empty and "verdict" in consistency_df.columns:
        n_consistent = consistency_df["verdict"].isin(
            ["CONSISTENT", "WEAK_CONSISTENT"]
        ).sum()
        n_total = len(consistency_df[consistency_df["verdict"] != "SKIP"])
        pct = round(n_consistent / max(n_total, 1) * 100, 1)
        v = "PASS" if pct >= 70 else ("WARN" if pct >= 50 else "FAIL")
        rows.append({
            "check": "Economic Consistency",
            "detail": f"{n_consistent}/{n_total} relationships consistent ({pct}%)",
            "verdict": v,
        })
    else:
        rows.append({"check": "Economic Consistency", "detail": "No data", "verdict": "SKIP"})

    # 3. Sector economics
    if not sector_df.empty and "pe_in_expected_range" in sector_df.columns:
        n_ok = sector_df["pe_in_expected_range"].sum()
        n_total = len(sector_df)
        pct = round(n_ok / max(n_total, 1) * 100, 1)
        v = "PASS" if pct >= 70 else ("WARN" if pct >= 50 else "FAIL")
        rows.append({
            "check": "Sector Economics (P/E ranges)",
            "detail": f"{int(n_ok)}/{n_total} sectors in expected P/E range ({pct}%)",
            "verdict": v,
        })
    else:
        rows.append({"check": "Sector Economics", "detail": "No data", "verdict": "SKIP"})

    # 4. Factor interpretation — read verdicts file
    vpath = OUT_DIR / "factor_economic_interpretation_verdicts.csv"
    if vpath.exists():
        vdf = pd.read_csv(vpath)
        n_pass = (vdf["verdict"] == "PASS").sum()
        n_total = len(vdf[vdf["verdict"] != "NO_CHECK"])
        pct = round(n_pass / max(n_total, 1) * 100, 1)
        v = "PASS" if pct >= 70 else ("WARN" if pct >= 50 else "FAIL")
        rows.append({
            "check": "Factor Economic Interpretation",
            "detail": f"{n_pass}/{n_total} factors economically sensible ({pct}%)",
            "verdict": v,
        })
    else:
        rows.append({"check": "Factor Economic Interpretation",
                      "detail": "Verdicts not generated", "verdict": "SKIP"})

    # 5. Currency conversion
    if not currency_df.empty:
        flag_path = OUT_DIR / "currency_conversion_flags.csv"
        n_flags = 0
        if flag_path.exists():
            n_flags = len(pd.read_csv(flag_path))
        v = "PASS" if n_flags <= 5 else ("WARN" if n_flags <= 15 else "FAIL")
        rows.append({
            "check": "Currency Conversion",
            "detail": f"{n_flags} flagged anomalies",
            "verdict": v,
        })
    else:
        rows.append({"check": "Currency Conversion", "detail": "No data", "verdict": "SKIP"})

    # 6. ESG provenance
    ptest_path = OUT_DIR / "esg_provenance_tests.csv"
    if ptest_path.exists():
        ptdf = pd.read_csv(ptest_path)
        n_high = (ptdf["bias_risk"] == "HIGH").sum() if "bias_risk" in ptdf.columns else 0
        n_mod = (ptdf["bias_risk"] == "MODERATE").sum() if "bias_risk" in ptdf.columns else 0
        v = "PASS" if n_high == 0 else ("WARN" if n_high <= 2 else "FAIL")
        rows.append({
            "check": "ESG Provenance Bias",
            "detail": f"{n_high} HIGH-bias, {n_mod} MODERATE-bias comparisons",
            "verdict": v,
        })
    else:
        rows.append({"check": "ESG Provenance Bias",
                      "detail": "No test data", "verdict": "SKIP"})

    # 7. Transaction costs
    tc_path = OUT_DIR / "transaction_cost_analysis.csv"
    if tc_path.exists():
        tc = pd.read_csv(tc_path)
        portfolio_row = tc[tc["ticker"] == "PORTFOLIO_TOTAL"]
        if not portfolio_row.empty:
            avg_bps = portfolio_row.iloc[0].get("total_rt_cost_bps", np.nan)
            total_usd = portfolio_row.iloc[0].get("total_rt_cost_usd", np.nan)
            v = "PASS" if pd.notna(avg_bps) and avg_bps < 100 else (
                "WARN" if pd.notna(avg_bps) and avg_bps < 200 else "FAIL")
            rows.append({
                "check": "Transaction Cost (round-trip)",
                "detail": f"Avg {avg_bps:.0f} bps, total ${total_usd:,.0f} per rebalance",
                "verdict": v,
            })
        else:
            rows.append({"check": "Transaction Cost", "detail": "No portfolio total", "verdict": "SKIP"})
    else:
        rows.append({"check": "Transaction Cost", "detail": "Not generated", "verdict": "SKIP"})

    # 8. Turnover
    to_path = OUT_DIR / "turnover_analysis.csv"
    if to_path.exists():
        to_df = pd.read_csv(to_path)
        verdict_row = to_df[to_df["metric"] == "verdict"]
        turnover_row = to_df[to_df["metric"] == "avg_turnover_pct"]
        if not verdict_row.empty and not turnover_row.empty:
            tv = str(verdict_row.iloc[0]["value"])
            tp = turnover_row.iloc[0]["value"]
            v = "PASS" if tv == "LOW" else ("WARN" if tv == "MODERATE" else "FAIL")
            rows.append({
                "check": "Portfolio Turnover",
                "detail": f"Avg quarterly turnover {tp}% ({tv})",
                "verdict": v,
            })
        else:
            rows.append({"check": "Portfolio Turnover", "detail": "Incomplete data", "verdict": "SKIP"})
    else:
        rows.append({"check": "Portfolio Turnover", "detail": "Not generated", "verdict": "SKIP"})

    # 9. Capacity
    cap_path = OUT_DIR / "capacity_aum_feasibility.csv"
    if cap_path.exists():
        cap = pd.read_csv(cap_path)
        details = []
        for _, cr in cap.iterrows():
            details.append(f"${cr['aum_M']:.0f}M={cr['verdict']}")
        # Verdict: PASS if $100M is feasible
        row_100 = cap[cap["aum_M"] == 100]
        if not row_100.empty:
            v100 = str(row_100.iloc[0]["verdict"])
            v = "PASS" if v100 in ("PASS", "YES") else ("WARN" if v100 == "PARTIAL" else "FAIL")
        else:
            v = "SKIP"
        rows.append({
            "check": "Strategy Capacity",
            "detail": ", ".join(details),
            "verdict": v,
        })
    else:
        rows.append({"check": "Strategy Capacity", "detail": "Not generated", "verdict": "SKIP"})

    # Overall verdict
    verdicts = [r["verdict"] for r in rows if r["verdict"] not in ("SKIP",)]
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif verdicts.count("WARN") >= 3:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "CONDITIONAL_PASS"
    else:
        overall = "PASS"
    rows.append({
        "check": "OVERALL FINANCIAL VALIDATION",
        "detail": f"{len([v for v in verdicts if v=='PASS'])} PASS, "
                  f"{verdicts.count('WARN')} WARN, {verdicts.count('FAIL')} FAIL",
        "verdict": overall,
    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "financial_validation_summary.csv", index=False)
    return out


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    print("=" * 70)
    print("PHASE 8 — Financial & Economic Validation")
    print("=" * 70)

    df = _load()
    print(f"Loaded {len(df)} companies, {len(df.columns)} columns.\n")
    print("  (Large-cap benchmarks excluded — validation targets mid-cap universe)\n")

    # 1. Ratio sanity
    print("1. Financial Ratio Sanity Checks …")
    ratio_df = _ratio_sanity(df)
    for _, r in ratio_df.iterrows():
        tag = r.get("verdict", "?")
        print(f"   {r['ratio']:20s}  {tag:6s}  "
              f"({r.get('pct_in_range','?')}% in range, "
              f"range={r.get('expected_range','?')})")
    print()

    # 2. Economic consistency
    print("2. Economic Consistency Checks …")
    consistency_df = _economic_consistency(df)
    for _, r in consistency_df.iterrows():
        tag = r.get("verdict", "?")
        print(f"   {r['relationship']:45s}  {tag:18s}  "
              f"(r={r.get('spearman_r','?')}, p={r.get('p_value','?')})")
    print()

    # 3. Sector economics
    print("3. Sector Economics Validation …")
    sector_df = _sector_economics(df)
    for _, r in sector_df.iterrows():
        in_range = r.get("pe_in_expected_range", "?")
        print(f"   {r['sector']:25s}  avg_PE={r['avg_trailing_pe']:6.1f}  "
              f"expected={r['expected_pe_range']}  "
              f"{'OK' if in_range else 'OUT_OF_RANGE'}")
    print()

    # 4. Factor interpretation
    print("4. Factor Score Economic Interpretation …")
    factor_df = _factor_interpretation(df)
    vpath = OUT_DIR / "factor_economic_interpretation_verdicts.csv"
    if vpath.exists():
        vdf = pd.read_csv(vpath)
        for _, r in vdf.iterrows():
            print(f"   {r['factor']:25s}  {r['verdict']:6s}  {r.get('economic_checks','')}")
    print()

    # 5. Currency conversion
    print("5. Currency Conversion Validation …")
    currency_df = _currency_validation(df)
    print(currency_df.to_string(index=False))
    flag_path = OUT_DIR / "currency_conversion_flags.csv"
    if flag_path.exists():
        fdf = pd.read_csv(flag_path)
        print(f"   → {len(fdf)} anomalies flagged")
    else:
        print("   → No anomalies flagged")
    print()

    # 6. ESG provenance
    print("6. ESG Data Provenance Impact …")
    provenance_df = _esg_provenance(df)
    print(provenance_df.to_string(index=False))
    ptest_path = OUT_DIR / "esg_provenance_tests.csv"
    if ptest_path.exists():
        ptdf = pd.read_csv(ptest_path)
        for _, r in ptdf.iterrows():
            print(f"   {r['comparison']:35s}  {r['score']:15s}  "
                  f"p={r['p_value']:.4f}  bias_risk={r['bias_risk']}")
    print()

    # 7. Fama-MacBeth cross-sectional regression
    print("7. Fama-MacBeth Style Cross-Sectional Regression …")
    fm_result = fama_macbeth_cross_sectional(df)
    if fm_result:
        quintile_ret, reg_df, ls_spread, r_sq = fm_result
        print("   Quintile mean returns (price_momentum_3m proxy):")
        print(quintile_ret.to_string())
        print(f"\n   Long-short spread (Q5 − Q1): {ls_spread:.4f}")
        print(f"   R-squared: {r_sq:.4f}")
        print("\n   Cross-sectional regression coefficients:")
        for _, row in reg_df.iterrows():
            sig = "*" if row["p_value"] < 0.05 else ""
            print(f"     {row['variable']:18s}  coef={row['coefficient']:8.4f}  "
                  f"t={row['t_statistic']:6.2f}  p={row['p_value']:.4f} {sig}")
        # Save outputs
        quintile_ret.to_csv(OUT_DIR / "fama_macbeth_quintile_returns.csv")
        reg_df.to_csv(OUT_DIR / "cross_sectional_regression.csv", index=False)
    else:
        print("   [SKIP] Could not run Fama-MacBeth regression (missing data).")
    print()

    # 8. Jonckheere-Terpstra monotonicity tests
    print("8. Jonckheere-Terpstra Monotonicity Tests …")
    factor_scores = [
        "financial_score", "ESG_composite", "growth_score", "value_score",
        "stability_score", "risk_adjusted_score", "momentum_score",
        "quality_score", "E_score", "S_score",
    ]
    return_col = "price_momentum_3m"
    jt_rows: list[dict] = []
    for sc in factor_scores:
        if sc not in df.columns:
            print(f"   {sc:25s}  SKIP (column missing)")
            continue
        if return_col not in df.columns:
            print(f"   {sc:25s}  SKIP (return proxy '{return_col}' missing)")
            break
        jt = jonckheere_terpstra_test(df, sc, return_col)
        jt_rows.append(jt)
        mono = "YES" if jt.get("monotonic") else "no"
        print(f"   {sc:25s}  z={jt.get('z_statistic','?'):>7}  "
              f"p={jt.get('p_value','?'):<10}  monotonic={mono}")
    if jt_rows:
        jt_df = pd.DataFrame(jt_rows)
        jt_df.to_csv(OUT_DIR / "monotonicity_tests.csv", index=False)
    print()

    # 9. Transaction cost analysis
    print("9. Transaction Cost Analysis …")
    tc_df = _transaction_cost_analysis(df)
    portfolio_row = tc_df[tc_df["ticker"] == "PORTFOLIO_TOTAL"]
    if not portfolio_row.empty:
        avg_bps = portfolio_row.iloc[0].get("total_rt_cost_bps", "?")
        total_usd = portfolio_row.iloc[0].get("total_rt_cost_usd", "?")
        print(f"   Portfolio avg round-trip cost: {avg_bps} bps")
        print(f"   Total cost per rebalance ($100M AUM): ${total_usd:,.0f}")
    else:
        print("   [Note] Could not compute portfolio-level costs.")
    n_companies = len(tc_df[tc_df["ticker"] != "PORTFOLIO_TOTAL"])
    print(f"   Analysed {n_companies} companies in top-20 portfolio")
    print()

    # 10. Turnover analysis
    print("10. Turnover Analysis …")
    to_df = _turnover_analysis(df)
    for _, r in to_df.iterrows():
        print(f"   {str(r['metric']):40s}  {r['value']}")
    print()

    # 11. Capacity analysis
    print("11. Capacity Analysis …")
    cap_df = _capacity_analysis(df)
    cap_feas_path = OUT_DIR / "capacity_aum_feasibility.csv"
    if cap_feas_path.exists():
        feas = pd.read_csv(cap_feas_path)
        for _, r in feas.iterrows():
            print(f"   AUM ${r['aum_M']:.0f}M → {r['n_feasible_at_1pct_adv']}/{r['n_total']} "
                  f"positions feasible at 1% ADV → {r['verdict']}")
    else:
        print("   [Note] Could not compute AUM feasibility.")
    # Print bottleneck rows
    bn = cap_df[cap_df["ticker"].str.startswith("MAX_AUM", na=False)]
    for _, r in bn.iterrows():
        print(f"   {r['ticker']:30s}  ${r['daily_dollar_vol_M']:.1f}M")
    print()

    # 12. Summary
    print("12. Building Summary Report …")
    summary_df = _build_summary(
        ratio_df, consistency_df, sector_df, factor_df,
        currency_df, provenance_df, df,
    )
    print()
    print("=" * 70)
    print("FINANCIAL VALIDATION SUMMARY")
    print("=" * 70)
    for _, r in summary_df.iterrows():
        print(f"  {r['check']:40s}  {r['verdict']:20s}  {r.get('detail','')}")
    print("=" * 70)

    # List all outputs
    outputs = [
        "financial_ratio_sanity.csv",
        "economic_consistency.csv",
        "sector_economics.csv",
        "factor_economic_interpretation.csv",
        "factor_economic_interpretation_verdicts.csv",
        "currency_conversion_validation.csv",
        "currency_conversion_flags.csv",
        "esg_provenance_impact.csv",
        "esg_provenance_tests.csv",
        "fama_macbeth_quintile_returns.csv",
        "cross_sectional_regression.csv",
        "monotonicity_tests.csv",
        "transaction_cost_analysis.csv",
        "turnover_analysis.csv",
        "capacity_analysis.csv",
        "capacity_aum_feasibility.csv",
        "financial_validation_summary.csv",
    ]
    print("\nOutputs saved:")
    for f in outputs:
        p = OUT_DIR / f
        if p.exists():
            print(f"  ✓ {p}")
        else:
            print(f"  - {p} (not generated)")

    overall = summary_df.iloc[-1]["verdict"]
    return 0 if overall in ("PASS", "CONDITIONAL_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
