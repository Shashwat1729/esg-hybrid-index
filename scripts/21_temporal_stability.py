"""
Step 21: Temporal Out-of-Sample Validation
============================================
Tests whether the multi-factor index has **predictive power for future
returns** using actual forward price data downloaded via yfinance.

This goes beyond the cross-sectional momentum proxy used in 07b by
correlating *current* factor scores (computed from the snapshot at time T)
with *actual realised forward returns* (T → T+k months).

**Key difference from 07b:**
  07b uses trailing momentum (price_momentum_1m/3m/6m) as a *return proxy*,
  which is backward-looking.  This script downloads *fresh* price data and
  computes genuinely forward-looking returns relative to the index snapshot
  date.  This is true out-of-sample validation.

**Circularity guard:**
  Forward returns are computed from actual price changes, NOT from the
  momentum variables used in scoring.  We use pref_balanced_ex_market as
  the primary preference score (excludes market_score which contains
  trailing momentum sub-factors).

Validation tests:

  1. **True Out-of-Sample IC**
     Spearman rank correlation between each factor score (at T) and actual
     forward return (T → T+k months, k ∈ {1, 3, 6}).

  2. **Profile-Level IC**
     IC for each investor-profile preference score (ex_market variants).

  3. **Bootstrap Confidence Intervals**
     1000-resample bootstrap for IC, yielding 95% CIs.

  4. **Rank Persistence**
     Split universe into formation/validation halves, measure Kendall τ
     between current factor-score rank and forward-return rank.

  5. **Quintile Portfolio Analysis**
     Form quintile portfolios by preference score; compute average forward
     returns per quintile; test monotonicity (Jonckheere–Terpstra-style).

Input:  data/processed/indexed_data.csv  (factor scores)
        Yahoo Finance daily prices      (downloaded at runtime)
Outputs:
  reports/tables/temporal_oos_ic.csv
  reports/tables/temporal_quintile_returns.csv
  reports/tables/temporal_rank_persistence.csv
  reports/tables/temporal_validation_summary.csv
"""

import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, norm as _norm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, ensure_dir
from src.constants import RANDOM_SEED

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
TABLES = ensure_dir(PROJECT_ROOT / "reports" / "tables")
FIGURES = ensure_dir(PROJECT_ROOT / "reports" / "figures")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = RANDOM_SEED
N_BOOTSTRAP = 1000

# Factor scores to test IC against forward returns
FACTOR_SCORES = [
    "ESG_composite",
    "financial_score",
    "market_score",
    "operational_score",
    "risk_adjusted_score",
    "growth_score",
    "value_score",
    "stability_score",
]

# Preference scores (ex_market = circularity-corrected, PRIMARY)
PREFERENCE_SCORES = [
    "pref_balanced_ex_market",
    "pref_esg_first_ex_market",
    "pref_financial_first_ex_market",
]

# Original preference scores (includes market_score, shown for comparison)
PREFERENCE_SCORES_BIASED = [
    "pref_balanced",
    "pref_esg_first",
    "pref_financial_first",
]

ALL_SCORES = FACTOR_SCORES + PREFERENCE_SCORES + PREFERENCE_SCORES_BIASED

# Forward-return horizons in trading days
FORWARD_HORIZONS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
}

ROLLING_WINDOWS = [
    ("window_1", "price_momentum_1m"),
    ("window_2", "price_momentum_3m"),
    ("window_3", "price_momentum_6m"),
    ("window_4", "price_momentum_12m"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FORWARD-RETURN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_yf_ticker(ticker: str) -> str:
    """Normalise ticker for yfinance.

    Indian tickers should already have .NS suffix; US tickers are plain.
    Handle edge cases where suffix might be missing or duplicated.
    """
    t = str(ticker).strip()
    # Already has a suffix — leave as-is
    if "." in t:
        return t
    # Plain US ticker
    return t


def download_forward_returns(
    tickers: list[str],
    period: str = "2y",
    max_retries: int = 2,
    batch_size: int = 40,
) -> pd.DataFrame:
    """Download daily prices via yfinance and compute forward returns.

    Returns a DataFrame indexed by ticker with columns:
        fwd_return_1m, fwd_return_3m, fwd_return_6m
    computed from the *most recent available* close at the time of index
    snapshot (assumed ≈ latest available date) forward.

    Strategy:
      - Download 2 years of daily adjusted close prices.
      - Identify the snapshot date T as the date the indexed data was built
        (estimated as the most recent common trading date across tickers).
      - Forward returns are percentage returns from T to T + k trading days.
      - If T + k is beyond available data, forward returns are computed from
        the available window and flagged.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[FATAL] yfinance is required: pip install yfinance")
        sys.exit(1)

    print(f"\n  Downloading price data for {len(tickers)} tickers (period={period})...")

    # ── batch download ─────────────────────────────────────────────────
    yf_tickers = [_ensure_yf_ticker(t) for t in tickers]

    all_prices = {}
    failed = []

    # Download in batches to avoid API throttling
    for batch_start in range(0, len(yf_tickers), batch_size):
        batch = yf_tickers[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(yf_tickers) - 1) // batch_size + 1
        print(f"    Batch {batch_num}/{total_batches} "
              f"({len(batch)} tickers: {batch[0]}..{batch[-1]})")

        for attempt in range(max_retries):
            try:
                # Use yf.download for batch efficiency
                data = yf.download(
                    batch,
                    period=period,
                    progress=False,
                    auto_adjust=True,
                    threads=True,
                )
                if data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    failed.extend(batch)
                    break

                # Extract Close prices
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-ticker download returns MultiIndex columns
                    close_df = data["Close"]
                else:
                    # Single ticker returns flat columns
                    close_df = data[["Close"]].rename(columns={"Close": batch[0]})

                for tk in batch:
                    if tk in close_df.columns:
                        series = close_df[tk].dropna()
                        if len(series) >= 30:  # need at least 30 days
                            all_prices[tk] = series
                        else:
                            failed.append(tk)
                    else:
                        failed.append(tk)
                break  # success
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      [RETRY] batch {batch_num}: {e}")
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"      [FAIL]  batch {batch_num}: {e}")
                    failed.extend(batch)

        # Rate-limit courtesy
        time.sleep(1.0)

    print(f"    Prices obtained: {len(all_prices)} / {len(yf_tickers)}")
    if failed:
        unique_failed = sorted(set(failed))
        print(f"    Failed ({len(unique_failed)}): "
              f"{unique_failed[:10]}{'...' if len(unique_failed) > 10 else ''}")

    if len(all_prices) < 10:
        print("[ERROR] Too few tickers with price data. Aborting.")
        return pd.DataFrame()

    # ── Determine snapshot date T ──────────────────────────────────────
    # T = latest date common to at least 50% of tickers
    date_sets = [set(s.index) for s in all_prices.values()]
    # Find dates present in at least 50% of tickers
    from collections import Counter
    date_counts = Counter()
    for ds in date_sets:
        date_counts.update(ds)
    threshold = len(all_prices) * 0.5
    common_dates = sorted(d for d, c in date_counts.items() if c >= threshold)

    if not common_dates:
        print("[ERROR] No common trading dates found. Aborting.")
        return pd.DataFrame()

    # Use the latest common date as snapshot date, but leave room for
    # forward returns.  We want T such that T + 126 days is ≤ max date.
    max_date = common_dates[-1]
    # Pick T as the date ~6 months (126 trading days) before the latest date
    # so forward returns actually span into the future window.
    # If we don't have 126 days forward, we'll use whatever is available.

    # Find a good T: we want to maximise the number of tickers with data
    # at T *and* at T + {21, 63, 126} days.  Use the date ~6 months back.
    target_idx = max(0, len(common_dates) - 1 - 126)
    T = common_dates[target_idx]
    print(f"    Snapshot date T = {T.strftime('%Y-%m-%d')}  "
          f"(latest date: {max_date.strftime('%Y-%m-%d')})")

    # ── Compute forward returns from T ─────────────────────────────────
    rows = []
    for tk, series in all_prices.items():
        # Find the closest trading date to T in this ticker's data
        dates = series.index
        valid = dates[dates >= T]
        if len(valid) == 0:
            continue
        t0_date = valid[0]
        t0_price = series.loc[t0_date]

        row = {"ticker": tk}
        for horizon_label, horizon_days in FORWARD_HORIZONS.items():
            # Find date at T + horizon_days (approximately)
            future = dates[dates >= t0_date]
            if len(future) > horizon_days:
                tk_date = future[horizon_days]
                tk_price = series.loc[tk_date]
                fwd_ret = (tk_price / t0_price - 1) * 100  # percentage
                row[f"fwd_return_{horizon_label}"] = float(fwd_ret)
            elif len(future) > 5:
                # Use whatever is available (shorter horizon)
                tk_date = future[-1]
                tk_price = series.loc[tk_date]
                fwd_ret = (tk_price / t0_price - 1) * 100
                row[f"fwd_return_{horizon_label}"] = float(fwd_ret)
                row[f"fwd_return_{horizon_label}_partial"] = True
            else:
                row[f"fwd_return_{horizon_label}"] = np.nan

        rows.append(row)

    fwd_df = pd.DataFrame(rows)
    print(f"    Forward returns computed for {len(fwd_df)} tickers")
    for h in FORWARD_HORIZONS:
        col = f"fwd_return_{h}"
        n_valid = fwd_df[col].notna().sum()
        if n_valid > 0:
            print(f"      {h}: n={n_valid}, "
                  f"mean={fwd_df[col].mean():.2f}%, "
                  f"std={fwd_df[col].std():.2f}%")

    return fwd_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. INFORMATION COEFFICIENT (IC) WITH BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════════════

def _spearman_ci_fisher(r: float, n: int, alpha: float = 0.05):
    """95% CI for Spearman r via Fisher z-transform."""
    if n <= 3 or not np.isfinite(r):
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = _norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se)))


def _ic_power(r: float, n: int, alpha: float = 0.05) -> float:
    """Statistical power to detect IC = r with sample size n."""
    if n <= 3 or not np.isfinite(r):
        return np.nan
    z_crit = _norm.ppf(1 - alpha / 2)
    ncp = abs(np.arctanh(np.clip(r, -0.9999, 0.9999))) * np.sqrt(n - 3)
    return float(1 - _norm.cdf(z_crit - ncp) + _norm.cdf(-z_crit - ncp))


def bootstrap_ic(
    scores: np.ndarray,
    returns: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap Spearman IC: point estimate + percentile CI."""
    rng = np.random.default_rng(seed)
    n = len(scores)

    # Point estimate
    rho, p = spearmanr(scores, returns)

    # Bootstrap
    boot_ics = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_ics[b], _ = spearmanr(scores[idx], returns[idx])
        except Exception:
            pass

    valid_boots = boot_ics[np.isfinite(boot_ics)]
    if len(valid_boots) < 50:
        return {
            "ic": float(rho),
            "ic_pvalue": float(p),
            "boot_ci_lo": np.nan,
            "boot_ci_hi": np.nan,
            "boot_mean": np.nan,
            "boot_std": np.nan,
            "n_boot_valid": len(valid_boots),
        }

    lo = float(np.percentile(valid_boots, 100 * alpha / 2))
    hi = float(np.percentile(valid_boots, 100 * (1 - alpha / 2)))

    return {
        "ic": float(rho),
        "ic_pvalue": float(p),
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "boot_mean": float(valid_boots.mean()),
        "boot_std": float(valid_boots.std()),
        "n_boot_valid": len(valid_boots),
    }


def compute_oos_ic(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """Compute out-of-sample IC for each score × forward-return horizon.

    Parameters
    ----------
    df : DataFrame
        Must contain score columns and fwd_return_{1m,3m,6m}.
    score_cols : list[str]
        Factor / preference score columns to evaluate.

    Returns
    -------
    DataFrame with one row per score × horizon.
    """
    rows = []
    for score_col in score_cols:
        if score_col not in df.columns:
            continue
        for h_label in FORWARD_HORIZONS:
            ret_col = f"fwd_return_{h_label}"
            if ret_col not in df.columns:
                continue

            valid = df[[score_col, ret_col]].dropna()
            if len(valid) < 15:
                continue

            scores = valid[score_col].values
            returns = valid[ret_col].values

            # Bootstrap IC
            result = bootstrap_ic(scores, returns)

            # Fisher z CI (analytic)
            fisher_lo, fisher_hi = _spearman_ci_fisher(result["ic"], len(valid))

            # Power
            power = _ic_power(result["ic"], len(valid))

            # Determine score type label
            if "_ex_market" in score_col:
                score_type = "preference_ex_market"
            elif score_col.startswith("pref_"):
                score_type = "preference_biased"
            else:
                score_type = "factor"

            rows.append({
                "score": score_col,
                "score_type": score_type,
                "horizon": h_label,
                "ic_spearman": result["ic"],
                "ic_pvalue": result["ic_pvalue"],
                "ic_significant": result["ic_pvalue"] < 0.05,
                "boot_ci_lo": result["boot_ci_lo"],
                "boot_ci_hi": result["boot_ci_hi"],
                "boot_mean": result["boot_mean"],
                "boot_std": result["boot_std"],
                "fisher_ci_lo": fisher_lo,
                "fisher_ci_hi": fisher_hi,
                "power": power,
                "n": len(valid),
                "n_boot_valid": result["n_boot_valid"],
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 3. RANK PERSISTENCE (FORMATION → VALIDATION)
# ═══════════════════════════════════════════════════════════════════════════

def rank_persistence(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """Measure Kendall τ between factor-score ranking and forward-return ranking.

    This captures whether the rank ordering implied by the index at time T
    persists in actual forward returns.
    """
    rows = []
    for score_col in score_cols:
        if score_col not in df.columns:
            continue
        for h_label in FORWARD_HORIZONS:
            ret_col = f"fwd_return_{h_label}"
            if ret_col not in df.columns:
                continue

            valid = df[[score_col, ret_col]].dropna()
            if len(valid) < 15:
                continue

            # Rank by factor score (higher = better) vs rank by forward return
            score_rank = valid[score_col].rank(ascending=False)
            return_rank = valid[ret_col].rank(ascending=False)

            tau, tau_p = kendalltau(score_rank, return_rank)

            rows.append({
                "score": score_col,
                "horizon": h_label,
                "kendall_tau": float(tau),
                "tau_pvalue": float(tau_p),
                "tau_significant": tau_p < 0.05,
                "n": len(valid),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 4. QUINTILE PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _jonckheere_terpstra_approx(quintile_means: np.ndarray) -> dict:
    """Approximate Jonckheere-Terpstra test for ordered alternatives.

    Tests H0: no ordering among quintile means vs H1: monotone ordering.
    Uses the Mann-Kendall S statistic as the JT test statistic with
    normal approximation for p-value.

    Parameters
    ----------
    quintile_means : array of 5 quintile means (Q1 to Q5).

    Returns
    -------
    dict with S, z_score, p_value, monotonic_direction.
    """
    k = len(quintile_means)
    S = 0
    for i in range(k):
        for j in range(i + 1, k):
            if quintile_means[j] > quintile_means[i]:
                S += 1
            elif quintile_means[j] < quintile_means[i]:
                S -= 1

    # Under H0, E[S] = 0 and Var[S] = k(k-1)(2k+5)/18
    var_S = k * (k - 1) * (2 * k + 5) / 18
    if var_S <= 0:
        return {"S": S, "z_score": np.nan, "p_value": np.nan,
                "direction": "unknown"}

    z = S / np.sqrt(var_S)
    # Two-sided p-value
    p_value = 2 * (1 - _norm.cdf(abs(z)))

    if S > 0:
        direction = "increasing"
    elif S < 0:
        direction = "decreasing"
    else:
        direction = "flat"

    return {
        "S": int(S),
        "z_score": float(z),
        "p_value": float(p_value),
        "direction": direction,
    }


def quintile_analysis(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """Form quintile portfolios by each score and compute average forward returns.

    Returns one row per score × horizon × quintile, plus monotonicity test.
    """
    rows = []
    for score_col in score_cols:
        if score_col not in df.columns:
            continue

        valid = df[df[score_col].notna()].copy()
        if len(valid) < 25:  # need at least 5 per quintile
            continue

        # Form quintiles on the score
        try:
            valid["quintile"] = pd.qcut(
                valid[score_col], 5,
                labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"],
            )
        except ValueError:
            # Handle ties by ranking first
            valid["quintile"] = pd.qcut(
                valid[score_col].rank(method="first"), 5,
                labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"],
            )

        q_labels = ["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"]

        for h_label in FORWARD_HORIZONS:
            ret_col = f"fwd_return_{h_label}"
            if ret_col not in valid.columns:
                continue

            # Quintile-level statistics
            q_means = []
            for q in q_labels:
                q_data = valid.loc[valid["quintile"] == q, ret_col].dropna()
                mean_ret = q_data.mean() if len(q_data) > 0 else np.nan
                q_means.append(mean_ret)

                rows.append({
                    "score": score_col,
                    "horizon": h_label,
                    "quintile": q,
                    "mean_fwd_return": float(mean_ret) if np.isfinite(mean_ret) else np.nan,
                    "median_fwd_return": float(q_data.median()) if len(q_data) > 0 else np.nan,
                    "std_fwd_return": float(q_data.std()) if len(q_data) > 1 else np.nan,
                    "n": len(q_data),
                })

            # Monotonicity test (Jonckheere-Terpstra)
            q_arr = np.array(q_means)
            if np.all(np.isfinite(q_arr)):
                jt = _jonckheere_terpstra_approx(q_arr)
                # Spread: Q5 - Q1
                spread = q_arr[-1] - q_arr[0]
                # Add a summary row for the full quintile spread
                rows.append({
                    "score": score_col,
                    "horizon": h_label,
                    "quintile": "Q5-Q1_spread",
                    "mean_fwd_return": float(spread),
                    "median_fwd_return": np.nan,
                    "std_fwd_return": np.nan,
                    "n": int(valid[ret_col].notna().sum()),
                    "jt_S": jt["S"],
                    "jt_z": jt["z_score"],
                    "jt_pvalue": jt["p_value"],
                    "jt_direction": jt["direction"],
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

def build_summary(
    ic_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    quintile_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact summary table suitable for the IEEE thesis.

    One row per score × horizon with: IC, bootstrap CI, Kendall τ,
    Q5-Q1 spread, JT monotonicity p-value.
    """
    rows = []

    if ic_df.empty:
        return pd.DataFrame()

    for _, ic_row in ic_df.iterrows():
        score = ic_row["score"]
        horizon = ic_row["horizon"]

        row = {
            "score": score,
            "score_type": ic_row.get("score_type", ""),
            "horizon": horizon,
            "ic_spearman": ic_row["ic_spearman"],
            "ic_pvalue": ic_row["ic_pvalue"],
            "ic_significant": ic_row["ic_significant"],
            "boot_ci_lo": ic_row["boot_ci_lo"],
            "boot_ci_hi": ic_row["boot_ci_hi"],
            "power": ic_row["power"],
            "n": ic_row["n"],
        }

        # Merge Kendall τ
        if not rank_df.empty:
            match = rank_df[
                (rank_df["score"] == score) & (rank_df["horizon"] == horizon)
            ]
            if len(match) > 0:
                row["kendall_tau"] = match.iloc[0]["kendall_tau"]
                row["tau_pvalue"] = match.iloc[0]["tau_pvalue"]
                row["tau_significant"] = match.iloc[0]["tau_significant"]

        # Merge quintile spread & JT test
        if not quintile_df.empty:
            spread_row = quintile_df[
                (quintile_df["score"] == score)
                & (quintile_df["horizon"] == horizon)
                & (quintile_df["quintile"] == "Q5-Q1_spread")
            ]
            if len(spread_row) > 0:
                row["q5_q1_spread"] = spread_row.iloc[0]["mean_fwd_return"]
                row["jt_pvalue"] = spread_row.iloc[0].get("jt_pvalue", np.nan)
                row["jt_direction"] = spread_row.iloc[0].get("jt_direction", "")

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════

def print_ic_table(ic_df: pd.DataFrame) -> None:
    """Print IC results in a compact table."""
    if ic_df.empty:
        print("  (no IC results)")
        return

    print("\n" + "=" * 90)
    print("  OUT-OF-SAMPLE INFORMATION COEFFICIENT (Spearman IC)")
    print("=" * 90)
    print(f"  {'Score':<35} {'Horizon':>7} {'IC':>7} {'p':>8} "
          f"{'Boot CI':>16} {'Sig':>4} {'n':>4}")
    print("  " + "-" * 85)

    for _, r in ic_df.iterrows():
        sig = "***" if r["ic_pvalue"] < 0.01 else (
            "**" if r["ic_pvalue"] < 0.05 else (
                "*" if r["ic_pvalue"] < 0.05 else ""))
        ci_str = f"[{r['boot_ci_lo']:+.3f}, {r['boot_ci_hi']:+.3f}]"
        print(f"  {r['score']:<35} {r['horizon']:>7} {r['ic_spearman']:+.4f} "
              f"{r['ic_pvalue']:>8.4f} {ci_str:>16} {sig:>4} {r['n']:>4}")


def print_quintile_table(quintile_df: pd.DataFrame) -> None:
    """Print quintile returns for preference scores."""
    if quintile_df.empty:
        print("  (no quintile results)")
        return

    # Only show preference score results in detail
    pref_scores = PREFERENCE_SCORES + PREFERENCE_SCORES_BIASED
    q_pref = quintile_df[
        quintile_df["score"].isin(pref_scores)
        & (quintile_df["quintile"] != "Q5-Q1_spread")
    ]
    if q_pref.empty:
        return

    print("\n" + "=" * 90)
    print("  QUINTILE PORTFOLIO FORWARD RETURNS (Preference Scores)")
    print("=" * 90)

    for score in pref_scores:
        sub = q_pref[q_pref["score"] == score]
        if sub.empty:
            continue
        print(f"\n  {score}:")
        print(f"    {'Horizon':>7} {'Q1(Low)':>10} {'Q2':>10} {'Q3':>10} "
              f"{'Q4':>10} {'Q5(High)':>10} {'Spread':>10}")
        print("    " + "-" * 67)

        for h in FORWARD_HORIZONS:
            h_data = sub[sub["horizon"] == h]
            if h_data.empty:
                continue
            vals = []
            for q in ["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"]:
                q_row = h_data[h_data["quintile"] == q]
                if len(q_row) > 0 and np.isfinite(q_row.iloc[0]["mean_fwd_return"]):
                    vals.append(f"{q_row.iloc[0]['mean_fwd_return']:+.2f}%")
                else:
                    vals.append("    N/A")

            # Get spread
            sp = quintile_df[
                (quintile_df["score"] == score)
                & (quintile_df["horizon"] == h)
                & (quintile_df["quintile"] == "Q5-Q1_spread")
            ]
            sp_str = (f"{sp.iloc[0]['mean_fwd_return']:+.2f}%"
                      if len(sp) > 0 and np.isfinite(sp.iloc[0]["mean_fwd_return"])
                      else "    N/A")

            print(f"    {h:>7} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} "
                  f"{vals[3]:>10} {vals[4]:>10} {sp_str:>10}")


def print_rank_persistence(rank_df: pd.DataFrame) -> None:
    """Print rank persistence results."""
    if rank_df.empty:
        print("  (no rank persistence results)")
        return

    print("\n" + "=" * 90)
    print("  RANK PERSISTENCE (Kendall tau: factor-score rank vs forward-return rank)")
    print("=" * 90)
    print(f"  {'Score':<35} {'Horizon':>7} {'tau':>7} {'p':>8} {'Sig':>4} {'n':>4}")
    print("  " + "-" * 68)

    for _, r in rank_df.iterrows():
        sig = "***" if r["tau_pvalue"] < 0.01 else (
            "**" if r["tau_pvalue"] < 0.05 else (
                "*" if r["tau_pvalue"] < 0.05 else ""))
        print(f"  {r['score']:<35} {r['horizon']:>7} {r['kendall_tau']:+.4f} "
              f"{r['tau_pvalue']:>8.4f} {sig:>4} {r['n']:>4}")


def print_summary_verdict(summary_df: pd.DataFrame) -> None:
    """Print a concise verdict for the thesis."""
    if summary_df.empty:
        return

    print("\n" + "=" * 90)
    print("  VERDICT: TEMPORAL OUT-OF-SAMPLE VALIDATION")
    print("=" * 90)

    # Count significant ICs
    n_total = len(summary_df)
    n_sig_ic = summary_df["ic_significant"].sum() if "ic_significant" in summary_df.columns else 0
    pct_sig = n_sig_ic / n_total * 100 if n_total > 0 else 0

    print(f"  Total score x horizon tests:  {n_total}")
    print(f"  Significant IC (p<0.05):      {n_sig_ic} ({pct_sig:.1f}%)")

    # Focus on ex_market preference scores
    ex_market = summary_df[summary_df["score_type"] == "preference_ex_market"]
    if len(ex_market) > 0:
        n_sig_ex = ex_market["ic_significant"].sum()
        mean_ic_ex = ex_market["ic_spearman"].mean()
        print(f"\n  Ex-market preference scores (circularity-free):")
        print(f"    Tests: {len(ex_market)}, Significant: {n_sig_ex}")
        print(f"    Mean IC: {mean_ic_ex:+.4f}")

        if "q5_q1_spread" in ex_market.columns:
            mean_spread = ex_market["q5_q1_spread"].mean()
            print(f"    Mean Q5-Q1 spread: {mean_spread:+.2f}%")

    # Factor-level summary
    factors = summary_df[summary_df["score_type"] == "factor"]
    if len(factors) > 0:
        n_sig_f = factors["ic_significant"].sum()
        mean_ic_f = factors["ic_spearman"].mean()
        print(f"\n  Factor-level scores:")
        print(f"    Tests: {len(factors)}, Significant: {n_sig_f}")
        print(f"    Mean IC: {mean_ic_f:+.4f}")

    # Interpretation
    if pct_sig >= 30:
        verdict = "STRONG: substantial out-of-sample predictive signal"
    elif pct_sig >= 15:
        verdict = "MODERATE: some out-of-sample predictive signal detected"
    elif pct_sig >= 5:
        verdict = "WEAK: limited out-of-sample signal (above chance)"
    else:
        verdict = "NONE: no out-of-sample predictive signal detected"

    print(f"\n  Overall assessment: {verdict}")
    print(f"  (Chance level at alpha=0.05: ~5% of tests significant)")


# ═══════════════════════════════════════════════════════════════════════════
# 7. ROLLING TEMPORAL STABILITY ANALYSES (CSV-BASED)
# ═══════════════════════════════════════════════════════════════════════════

def load_temporal_inputs() -> pd.DataFrame:
    """Load indexed_data and market_data with robust error handling."""
    try:
        indexed = load_indexed_data(PROJECT_ROOT)
        print(f"  [OK] Loaded indexed_data: {len(indexed)} rows")
    except Exception as e:
        print(f"[FATAL] Failed to load indexed data: {e}")
        return pd.DataFrame()

    market_path = PROJECT_ROOT / "data" / "raw" / "market_data.csv"
    try:
        market = pd.read_csv(market_path)
        print(f"  [OK] Loaded market_data: {len(market)} rows")
    except Exception as e:
        print(f"[FATAL] Failed to load market_data.csv: {e}")
        return pd.DataFrame()

    if "ticker" not in indexed.columns or "ticker" not in market.columns:
        print("[FATAL] Missing 'ticker' column in indexed or market data")
        return pd.DataFrame()

    merged = indexed.merge(market, on="ticker", how="inner", suffixes=("", "_mkt"))

    # Prefer explicit market_data momentum columns when present
    for col in [
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_12m",
    ]:
        mkt_col = f"{col}_mkt"
        if mkt_col in merged.columns:
            merged[col] = merged[mkt_col]

    print(f"  [OK] Merged indexed + market data: {len(merged)} rows")
    return merged


def compute_temporal_rolling_ic(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute 4-window rolling IC summary for available factor scores."""
    score_cols = [c for c in FACTOR_SCORES if c in merged.columns]
    if not score_cols:
        print("  [WARN] No factor score columns available for rolling IC")
        return pd.DataFrame()

    rows = []
    for score_col in score_cols:
        window_ics = {}
        n_points = {}

        for window_name, ret_col in ROLLING_WINDOWS:
            if ret_col not in merged.columns:
                print(f"  [WARN] Missing momentum column: {ret_col}")
                window_ics[window_name] = np.nan
                n_points[window_name] = 0
                continue

            valid = merged[[score_col, ret_col]].dropna()
            n_points[window_name] = len(valid)
            if len(valid) < 15:
                window_ics[window_name] = np.nan
                continue

            rho, _ = spearmanr(valid[score_col], valid[ret_col])
            window_ics[window_name] = float(rho) if np.isfinite(rho) else np.nan

        ic_values = np.array([window_ics[w] for w, _ in ROLLING_WINDOWS], dtype=float)
        valid_ics = ic_values[np.isfinite(ic_values)]

        mean_ic = float(np.mean(valid_ics)) if len(valid_ics) > 0 else np.nan
        ic_variability = float(np.std(valid_ics)) if len(valid_ics) > 1 else np.nan

        if len(valid_ics) > 0:
            signs = np.sign(valid_ics)
            sign_stability = float(abs(np.mean(signs)))
        else:
            sign_stability = np.nan

        rows.append({
            "score": score_col,
            "ic_window_1": window_ics.get("window_1", np.nan),
            "ic_window_2": window_ics.get("window_2", np.nan),
            "ic_window_3": window_ics.get("window_3", np.nan),
            "ic_window_4": window_ics.get("window_4", np.nan),
            "n_window_1": n_points.get("window_1", 0),
            "n_window_2": n_points.get("window_2", 0),
            "n_window_3": n_points.get("window_3", 0),
            "n_window_4": n_points.get("window_4", 0),
            "mean_ic": mean_ic,
            "ic_sign_stability": sign_stability,
            "ic_variability": ic_variability,
        })

    return pd.DataFrame(rows)


def simulate_rolling_portfolio(merged: pd.DataFrame) -> pd.DataFrame:
    """Simulate 6m rolling rebalance with top-20 by composite score."""
    composite_candidates = [
        "pref_balanced_ex_market",
        "pref_balanced",
        "ESG_composite",
    ]
    composite_col = next((c for c in composite_candidates if c in merged.columns), None)
    if composite_col is None:
        print("  [WARN] No composite score available for portfolio simulation")
        return pd.DataFrame()

    rows = []
    for period_name, ret_col in ROLLING_WINDOWS:
        if ret_col not in merged.columns:
            continue

        valid = merged[["ticker", composite_col, ret_col]].dropna()
        if valid.empty:
            continue

        ranked = valid.sort_values(composite_col, ascending=False)
        top20 = ranked.head(20)["ticker"].tolist()
        if len(top20) < 20:
            print(f"  [WARN] {period_name}: fewer than 20 tickers available")

        portfolio_slice = valid[valid["ticker"].isin(top20)]
        if portfolio_slice.empty:
            continue

        portfolio_return = float(portfolio_slice[ret_col].mean())
        universe_return = float(valid[ret_col].mean())
        excess_return = float(portfolio_return - universe_return)

        rows.append({
            "period": period_name,
            "return_proxy_col": ret_col,
            "portfolio_return": portfolio_return,
            "universe_return": universe_return,
            "excess_return": excess_return,
            "beat_universe": int(excess_return > 0),
            "portfolio_size": int(len(portfolio_slice)),
            "composite_col": composite_col,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    beat_count = int(result["beat_universe"].sum())
    n_periods = int(len(result))
    avg_excess = float(result["excess_return"].mean())
    consistency = float(beat_count / n_periods) if n_periods > 0 else np.nan

    summary = pd.DataFrame([
        {
            "period": "summary",
            "return_proxy_col": "-",
            "portfolio_return": np.nan,
            "universe_return": np.nan,
            "excess_return": avg_excess,
            "beat_universe": beat_count,
            "portfolio_size": int(result["portfolio_size"].max()),
            "composite_col": composite_col,
            "n_periods": n_periods,
            "periods_beat_universe": beat_count,
            "average_excess_return": avg_excess,
            "consistency": consistency,
        }
    ])

    for col in ["n_periods", "periods_beat_universe", "average_excess_return", "consistency"]:
        if col not in result.columns:
            result[col] = np.nan

    return pd.concat([result, summary], ignore_index=True)


def compute_factor_ic_decay(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute IC decay curves across 1m/3m/6m/12m momentum horizons."""
    horizon_cols = {
        "1m": "price_momentum_1m",
        "3m": "price_momentum_3m",
        "6m": "price_momentum_6m",
        "12m": "price_momentum_12m",
    }

    rows = []
    for factor_col in FACTOR_SCORES:
        if factor_col not in merged.columns:
            continue

        factor_ic = {}
        for horizon, momentum_col in horizon_cols.items():
            if momentum_col not in merged.columns:
                continue

            valid = merged[[factor_col, momentum_col]].dropna()
            if len(valid) < 15:
                factor_ic[horizon] = np.nan
                continue

            rho, pval = spearmanr(valid[factor_col], valid[momentum_col])
            ic_val = float(rho) if np.isfinite(rho) else np.nan
            factor_ic[horizon] = ic_val

            rows.append({
                "factor": factor_col,
                "horizon": horizon,
                "ic_spearman": ic_val,
                "ic_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "n": len(valid),
            })

        ic_1m = abs(factor_ic.get("1m", np.nan))
        ic_12m = abs(factor_ic.get("12m", np.nan))
        if np.isfinite(ic_1m) and ic_1m > 0 and np.isfinite(ic_12m):
            retention_12m = float(ic_12m / ic_1m)
        else:
            retention_12m = np.nan

        if np.isfinite(retention_12m):
            if retention_12m >= 0.7:
                decay_class = "slow_decay"
            elif retention_12m >= 0.4:
                decay_class = "moderate_decay"
            else:
                decay_class = "fast_decay"
        else:
            decay_class = "unknown"

        for row in rows:
            if row["factor"] == factor_col:
                row["retention_12m_vs_1m"] = retention_12m
                row["decay_class"] = decay_class

    return pd.DataFrame(rows)


def plot_temporal_rolling_performance(portfolio_df: pd.DataFrame) -> None:
    """Generate rolling performance figure for temporal validation."""
    if portfolio_df.empty:
        return

    plot_df = portfolio_df[portfolio_df["period"] != "summary"].copy()
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(x, plot_df["portfolio_return"], marker="o", label="Top-20 Portfolio")
    axes[0].plot(x, plot_df["universe_return"], marker="o", label="Universe")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["period"].tolist())
    axes[0].set_ylabel("Return proxy (%)")
    axes[0].set_title("Temporal Rolling Portfolio vs Universe")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in plot_df["excess_return"]]
    axes[1].bar(x, plot_df["excess_return"], color=colors)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["period"].tolist())
    axes[1].set_ylabel("Excess return (%)")
    axes[1].set_title("Excess Return (Portfolio - Universe)")
    axes[1].grid(True, axis="y", alpha=0.3)

    outpath = FIGURES / "fig_temporal_rolling_performance.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved figure: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("STEP 21: TEMPORAL OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    print("  Correlates current factor scores with actual forward returns")
    print("  downloaded via yfinance (true out-of-sample test).\n")

    # ── Load temporal inputs ───────────────────────────────────────────
    print("\n── Phase 0: Loading Input Data ──")
    merged = load_temporal_inputs()
    if merged.empty:
        print("[FATAL] Failed to load required input data. Exiting.")
        return

    print("\n── Phase A: Multi-period Rolling IC (market_data.csv) ──")
    rolling_ic_df = compute_temporal_rolling_ic(merged)
    if not rolling_ic_df.empty:
        rolling_ic_path = TABLES / "temporal_rolling_ic.csv"
        rolling_ic_df.to_csv(rolling_ic_path, index=False, encoding="utf-8")
        print(f"  [OK] Saved temporal_rolling_ic.csv ({len(rolling_ic_df)} rows)")
    else:
        print("  [WARN] Rolling IC table is empty")

    print("\n── Phase B: Rolling Portfolio Rebalance Simulation ──")
    rolling_portfolio_df = simulate_rolling_portfolio(merged)
    if not rolling_portfolio_df.empty:
        rolling_portfolio_path = TABLES / "temporal_rolling_portfolio.csv"
        rolling_portfolio_df.to_csv(
            rolling_portfolio_path, index=False, encoding="utf-8"
        )
        print(f"  [OK] Saved temporal_rolling_portfolio.csv ({len(rolling_portfolio_df)} rows)")
        plot_temporal_rolling_performance(rolling_portfolio_df)
    else:
        print("  [WARN] Rolling portfolio table is empty")

    print("\n── Phase C: Factor IC Decay Analysis ──")
    ic_decay_df = compute_factor_ic_decay(merged)
    if not ic_decay_df.empty:
        ic_decay_path = TABLES / "temporal_ic_decay.csv"
        ic_decay_df.to_csv(ic_decay_path, index=False, encoding="utf-8")
        print(f"  [OK] Saved temporal_ic_decay.csv ({len(ic_decay_df)} rows)")
    else:
        print("  [WARN] IC decay table is empty")

    # ── Keep existing temporal OOS analysis ────────────────────────────
    df = merged.copy()
    print(f"\n  [OK] Using {len(df)} merged companies for OOS validation")

    # Determine which score columns are available
    available_scores = [c for c in ALL_SCORES if c in df.columns]
    print(f"  [OK] {len(available_scores)} score columns available for validation")

    if "ticker" not in df.columns:
        print("[FATAL] No 'ticker' column in indexed data. Cannot proceed.")
        return

    tickers = df["ticker"].dropna().unique().tolist()
    print(f"  [OK] {len(tickers)} unique tickers")

    # Build forward-return aliases from market_data momentum columns
    for h in FORWARD_HORIZONS:
        src = f"price_momentum_{h}"
        dst = f"fwd_return_{h}"
        if src in merged.columns and dst not in merged.columns:
            merged[dst] = merged[src]

    if not any(f"fwd_return_{h}" in merged.columns for h in FORWARD_HORIZONS):
        print("[FATAL] Missing forward return proxy columns (price_momentum_1m/3m/6m).")
        return

    # ── Compute IC ─────────────────────────────────────────────────────
    print("\n── Phase 3: Computing Out-of-Sample IC ──")
    ic_df = compute_oos_ic(merged, available_scores)
    if not ic_df.empty:
        ic_df.to_csv(TABLES / "temporal_oos_ic.csv", index=False, encoding="utf-8")
        print(f"  [OK] Saved temporal_oos_ic.csv ({len(ic_df)} rows)")
        print_ic_table(ic_df)
    else:
        print("  [WARN] No IC results computed")

    # ── Rank persistence ───────────────────────────────────────────────
    print("\n── Phase 4: Rank Persistence ──")
    rank_df = rank_persistence(merged, available_scores)
    if not rank_df.empty:
        rank_df.to_csv(
            TABLES / "temporal_rank_persistence.csv",
            index=False, encoding="utf-8",
        )
        print(f"  [OK] Saved temporal_rank_persistence.csv ({len(rank_df)} rows)")
        print_rank_persistence(rank_df)
    else:
        print("  [WARN] No rank persistence results")

    # ── Quintile analysis ──────────────────────────────────────────────
    print("\n── Phase 5: Quintile Portfolio Analysis ──")
    quintile_df = quintile_analysis(merged, available_scores)
    if not quintile_df.empty:
        quintile_df.to_csv(
            TABLES / "temporal_quintile_returns.csv",
            index=False, encoding="utf-8",
        )
        print(f"  [OK] Saved temporal_quintile_returns.csv ({len(quintile_df)} rows)")
        print_quintile_table(quintile_df)
    else:
        print("  [WARN] No quintile results")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n── Phase 6: Building Summary Table ──")
    summary_df = build_summary(ic_df, rank_df, quintile_df)
    if not summary_df.empty:
        summary_df.to_csv(
            TABLES / "temporal_validation_summary.csv",
            index=False, encoding="utf-8",
        )
        print(f"  [OK] Saved temporal_validation_summary.csv ({len(summary_df)} rows)")
        print_summary_verdict(summary_df)
    else:
        print("  [WARN] Empty summary")

    # ── Copy key tables to Thesis_report ───────────────────────────────
    thesis_tables = PROJECT_ROOT / "Thesis_report" / "Tables"
    if thesis_tables.exists():
        import shutil
        for fname in [
            "temporal_oos_ic.csv",
            "temporal_quintile_returns.csv",
            "temporal_validation_summary.csv",
        ]:
            src = TABLES / fname
            if src.exists():
                shutil.copy2(src, thesis_tables / fname)
        print(f"\n  [OK] Copied key tables to {thesis_tables}")

    print("\n" + "=" * 70)
    print("STEP 21: COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
