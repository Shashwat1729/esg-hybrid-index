"""
Step 25: Genuine Out-of-Sample Evaluation
=========================================
Evaluates the index on returns realised strictly AFTER the data snapshot.

All factor scores in data/processed/indexed_data.csv are functions of
fundamentals and trailing market data observed at the close of the snapshot
date (config: universe.snapshot_date = 2026-04-02; verified by matching the
cached price_latest to Yahoo closes).  Every outcome used here is measured
over (snapshot_date, oos.end_date], so no input can have seen it.

PRE-DECLARED HYPOTHESES (written into this file before any post-snapshot
return was downloaded; alpha = 0.05, Holm-adjusted across H1-H3):

  H1  The deployed balanced composite (pref_balanced) has a positive
      Spearman rank IC with country-neutral forward returns over the full
      out-of-sample window.
  H2  ESG_composite is negatively associated with realised out-of-sample
      volatility AFTER controlling for trailing (pre-snapshot) volatility
      (partial Spearman correlation) -- the "risk filter" claim.
  H3  The equal-weighted top-N balanced portfolio (N = get_portfolio_top_n)
      earns a positive country-neutral excess return over the equal-weighted
      universe over the full window.

Everything else (per-factor ICs at 1m / 3m / full horizons, quintile
spreads, alternative weightings, the pre-audit pipeline's scores, regression
with sector/country effects) is SECONDARY / EXPLORATORY and is reported with
Holm adjustment within its family.

Returns are measured in local currency and made country-neutral by
demeaning within country (US vs India), so the test is not a bet on
USD/INR or on one market's beta.  A sector-and-country-neutral variant is
reported as a robustness check.

Data: Yahoo Finance v8 chart API (adjusted close), cached to
data/raw/forward_prices.csv so that the evaluation is reproducible offline.

Usage:
    python scripts/25_out_of_sample_evaluation.py            # use cache if present
    python scripts/25_out_of_sample_evaluation.py --refresh  # re-download
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.constants import DEFAULT_WEIGHTS, RANDOM_SEED
from src.utils import get_portfolio_top_n

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
PRICE_CACHE = RAW / "forward_prices.csv"

N_BOOT = 5000

FACTORS = [
    "ESG_composite", "E_score", "S_score", "G_score",
    "financial_score", "market_score", "operational_score",
    "risk_adjusted_score", "value_score", "growth_score",
    "stability_score", "sector_position",
]

# Original Feb-2026 balanced weights (git f77385e), ex-market as deployed.
FEB2026_BALANCED = {
    "ESG_composite": 0.15, "financial_score": 0.25, "operational_score": 0.10,
    "risk_adjusted_score": 0.08, "growth_score": 0.12, "value_score": 0.08,
    "stability_score": 0.05, "similarity_rank": 0.04, "sector_position": 0.03,
}

COUNTRY_INDEX = {"US": "^GSPC", "India": "^NSEI"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_oos_config():
    with open(PROJECT_ROOT / "config" / "index_config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    snapshot = pd.Timestamp(cfg["universe"]["snapshot_date"])
    oos = cfg.get("oos_evaluation", {})
    end = pd.Timestamp(oos.get("end_date", "2026-09-23"))
    horizons = oos.get("horizons_months", [1, 3])
    return snapshot, end, horizons


# ---------------------------------------------------------------------------
# Price download (Yahoo v8 chart API; no yfinance dependency)
# ---------------------------------------------------------------------------
def _fetch_chart(ticker, start, end, session, retries=3):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(start.replace(tzinfo=timezone.utc).timestamp()),
        "period2": int(end.replace(tzinfo=timezone.utc).timestamp()),
        "interval": "1d",
        "events": "div,split",
    }
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            res = r.json()["chart"]["result"]
            if not res:
                return None
            res = res[0]
            ts = res.get("timestamp") or []
            if not ts:
                return None
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            close = res["indicators"]["quote"][0]["close"]
            vals = adj if adj is not None else close
            # Exchange-local calendar date of each bar
            offset = res.get("meta", {}).get("gmtoffset", 0) or 0
            dates = [
                datetime.fromtimestamp(t + offset, tz=timezone.utc).date() for t in ts
            ]
            s = pd.Series(vals, index=pd.to_datetime(dates), dtype=float).dropna()
            return s[~s.index.duplicated(keep="last")]
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def download_prices(tickers, snapshot, end):
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research; multi-factor-esg)"})
    start = (snapshot - pd.Timedelta(days=10)).to_pydatetime()
    stop = (end + pd.Timedelta(days=2)).to_pydatetime()
    frames, failed = [], []
    for i, t in enumerate(tickers, 1):
        s = _fetch_chart(t, start, stop, session)
        if s is None or s.empty:
            failed.append(t)
        else:
            frames.append(pd.DataFrame({"date": s.index, "ticker": t, "adj_close": s.values}))
        if i % 25 == 0:
            print(f"    downloaded {i}/{len(tickers)}")
        time.sleep(0.25)
    prices = pd.concat(frames, ignore_index=True)
    prices = prices[prices["date"] <= end]
    print(f"  Prices: {prices['ticker'].nunique()} tickers, failed: {failed}")
    return prices, failed


def load_prices(tickers, snapshot, end, refresh):
    if PRICE_CACHE.exists() and not refresh:
        prices = pd.read_csv(PRICE_CACHE, parse_dates=["date"])
        print(f"  Using cached prices: {PRICE_CACHE} ({prices['ticker'].nunique()} tickers)")
        return prices
    print(f"  Downloading daily prices {snapshot.date()} -> {end.date()} for {len(tickers)} tickers...")
    prices, _ = download_prices(tickers, snapshot, end)
    prices.to_csv(PRICE_CACHE, index=False, encoding="utf-8")
    return prices


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def _price_on_or_before(series, date):
    s = series[series.index <= date]
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, np.nan)


def compute_outcomes(prices, snapshot, end, horizons):
    """Per-ticker forward returns and realised risk over (snapshot, end]."""
    idx_prices = {k: prices[prices["ticker"] == v].set_index("date")["adj_close"].sort_index()
                  for k, v in COUNTRY_INDEX.items()}
    targets = {f"ret_{h}m": snapshot + pd.DateOffset(months=h) for h in horizons}
    targets["ret_full"] = end
    rows = []
    for t, g in prices.groupby("ticker"):
        s = g.set_index("date")["adj_close"].sort_index()
        d0, p0 = _price_on_or_before(s, snapshot)
        if d0 is None or (snapshot - d0).days > 5:
            continue  # no valid anchor price at the snapshot
        row = {"ticker": t, "anchor_date": d0}
        for name, tgt in targets.items():
            d1, p1 = _price_on_or_before(s, tgt)
            row[name] = (p1 / p0 - 1.0) * 100.0 if d1 is not None and d1 > d0 else np.nan
        window = s[(s.index >= d0) & (s.index <= end)]
        daily = window.pct_change().dropna()
        row["last_date"] = window.index[-1]
        row["n_days_oos"] = len(daily)
        row["realized_vol_oos"] = daily.std(ddof=1) * np.sqrt(252) * 100 if len(daily) > 20 else np.nan
        path = window / window.iloc[0]
        row["max_drawdown_oos"] = ((path / path.cummax()) - 1.0).min() * 100 if len(window) > 20 else np.nan
        row["_daily"] = daily
        rows.append(row)
    out = pd.DataFrame(rows)
    return out, idx_prices


def add_realized_beta(out, universe, idx_prices):
    country = universe.set_index("ticker")["country"]
    betas = []
    for _, r in out.iterrows():
        c = country.get(r["ticker"], "US")
        ip = idx_prices.get(c)
        if ip is None or ip.empty or r["_daily"] is None or len(r["_daily"]) < 20:
            betas.append(np.nan)
            continue
        mret = ip.pct_change().dropna()
        j = pd.concat([r["_daily"], mret], axis=1, join="inner").dropna()
        if len(j) < 20 or j.iloc[:, 1].var() <= 0:
            betas.append(np.nan)
            continue
        betas.append(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    out["realized_beta_oos"] = betas
    return out.drop(columns=["_daily"])


def neutralize(df, col, by):
    return df[col] - df.groupby(by)[col].transform("mean")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def spearman_with_ci(x, y, rng, n_boot=N_BOOT):
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(), y[m].to_numpy()
    n = len(x)
    rho, p = stats.spearmanr(x, y)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([np.corrcoef(stats.rankdata(rx[i]), stats.rankdata(ry[i]))[0, 1] for i in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, p, lo, hi, n


def partial_spearman(x, y, z):
    """Spearman partial correlation of x and y given z (rank-residual method)."""
    m = x.notna() & y.notna() & z.notna()
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    n = int(m.sum())
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r**2))
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return r, p, n


def partial_spearman_ci(x, y, z, rng, n_boot=N_BOOT):
    """Firm-level bootstrap 95% CI for the partial Spearman correlation."""
    m = x.notna() & y.notna() & z.notna()
    xs, ys, zs = (v[m].reset_index(drop=True) for v in (x, y, z))
    n = len(xs)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        draws.append(partial_spearman(xs.iloc[i].reset_index(drop=True),
                                      ys.iloc[i].reset_index(drop=True),
                                      zs.iloc[i].reset_index(drop=True))[0])
    return tuple(np.nanpercentile(draws, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def portfolio_excess(score, ret, top_n, rng, n_boot=N_BOOT):
    """Equal-weight top-N minus equal-weight universe, with bootstrap CI.

    The bootstrap resamples firms (with replacement) and re-forms the
    top-N portfolio in each draw, so the CI reflects cross-sectional
    sampling uncertainty of the one-period excess return.
    """
    m = score.notna() & ret.notna()
    s, r = score[m].to_numpy(), ret[m].to_numpy()
    n = len(s)
    top = np.argsort(-s)[:top_n]
    excess = r[top].mean() - r.mean()
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        k = min(top_n, n)
        t = i[np.argsort(-s[i])[:k]]
        draws.append(r[t].mean() - r[i].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p_one_sided = float(np.mean(np.asarray(draws) <= 0))
    hit = float(np.mean(r[top] > np.median(r)))
    return excess, lo, hi, p_one_sided, hit, n


def quintile_spread(score, ret):
    m = score.notna() & ret.notna()
    q = pd.qcut(score[m].rank(method="first"), 5, labels=False)
    means = ret[m].groupby(q).mean()
    return means.iloc[-1] - means.iloc[0], means.to_numpy()


def rank_composite(df, weights):
    """Same aggregation as the deployed pref_* scores (rank mode)."""
    w = {k: v for k, v in weights.items() if v > 0 and k in df.columns}
    tot = sum(w.values())
    s = pd.Series(0.0, index=df.index)
    for k, v in w.items():
        vals = df[k].fillna(df[k].median())
        s += (v / tot) * vals.rank(method="average") / len(vals) * 100
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    refresh = "--refresh" in sys.argv
    snapshot, end, horizons = load_oos_config()
    print("=" * 70)
    print("STEP 25: GENUINE OUT-OF-SAMPLE EVALUATION")
    print(f"  Snapshot (scores observed at close): {snapshot.date()}")
    print(f"  Out-of-sample window: ({snapshot.date()}, {end.date()}]")
    print("=" * 70)

    df = pd.read_csv(PROCESSED / "indexed_data.csv")
    universe = df[~df["is_large_cap_benchmark"].fillna(False).astype(bool)].copy()
    n_universe = len(universe)
    top_n = get_portfolio_top_n(n_universe)

    tickers = sorted(set(df["ticker"]) | set(COUNTRY_INDEX.values()))
    prices = load_prices(tickers, snapshot, end, refresh)
    outcomes, idx_prices = compute_outcomes(prices, snapshot, end, horizons)
    outcomes = add_realized_beta(outcomes, df, idx_prices)

    u = universe.merge(outcomes, on="ticker", how="left")
    ret_cols = [f"ret_{h}m" for h in horizons] + ["ret_full"]
    n_missing = int(u["ret_full"].isna().sum())
    early_stop = u[(u["last_date"].notna()) & (u["last_date"] < end - pd.Timedelta(days=10))]
    print(f"  Mid-cap universe: {n_universe}; with full-window return: {n_universe - n_missing}; "
          f"stopped trading early: {len(early_stop)} {early_stop['ticker'].tolist()}")

    for c in ret_cols:
        u[f"{c}_cn"] = neutralize(u, c, "country")
        u[f"{c}_scn"] = neutralize(u, c, ["country", "sector"])

    # Market-model (beta-adjusted) full-window return using the *trailing*
    # (pre-snapshot) beta, so no OOS information enters the adjustment:
    # r_i - beta_i * r_index(country).  Separates low-beta tilts from
    # stock selection in a window where both indices rose strongly.
    idx_ret = {}
    for c, ip in idx_prices.items():
        _, p0 = _price_on_or_before(ip, snapshot)
        _, p1 = _price_on_or_before(ip, end)
        idx_ret[c] = (p1 / p0 - 1.0) * 100.0
    u["index_ret_full"] = u["country"].map(idx_ret)
    u["ret_full_ba"] = u["ret_full"] - u["beta"].fillna(u["beta"].median()) * u["index_ret_full"]
    u["ret_full_ba_cn"] = neutralize(u, "ret_full_ba", "country")

    # Alternative composites (robustness)
    u["pref_equal_weight"] = rank_composite(u, {k: 1.0 for k in DEFAULT_WEIGHTS if DEFAULT_WEIGHTS[k] > 0})
    u["pref_balanced_feb2026"] = rank_composite(u, FEB2026_BALANCED)
    pre = PROJECT_ROOT / "reports" / "pre_audit" / "indexed_data.csv"
    if pre.exists():
        pa = pd.read_csv(pre)[["ticker", "pref_balanced", "ESG_composite"]].rename(
            columns={"pref_balanced": "pref_balanced_pre_audit", "ESG_composite": "ESG_composite_pre_audit"})
        u = u.merge(pa, on="ticker", how="left")
    # ESG orthogonal to the other factors (incremental ESG information)
    others = [c for c in ["financial_score", "operational_score", "risk_adjusted_score",
                          "value_score", "growth_score", "stability_score", "market_score"] if c in u]
    X = np.column_stack([np.ones(len(u))] + [u[c].fillna(u[c].median()) for c in others])
    y = u["ESG_composite"].fillna(u["ESG_composite"].median()).to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u["ESG_orthogonal"] = y - X @ beta
    esg_r2_on_factors = 1 - np.var(u["ESG_orthogonal"]) / np.var(y)

    rng = np.random.default_rng(RANDOM_SEED)
    target = "ret_full_cn"

    # ---------------- Primary hypotheses ----------------
    rho1, p1_two, lo1, hi1, n1 = spearman_with_ci(u["pref_balanced"], u[target], rng)
    p1 = p1_two / 2 if rho1 > 0 else 1 - p1_two / 2
    r2, p2_two, n2 = partial_spearman(u["ESG_composite"], u["realized_vol_oos"], u["price_volatility"])
    p2 = p2_two / 2 if r2 < 0 else 1 - p2_two / 2
    lo2, hi2 = partial_spearman_ci(u["ESG_composite"], u["realized_vol_oos"], u["price_volatility"], rng)
    ex3, lo3, hi3, p3, hit3, n3 = portfolio_excess(u["pref_balanced"], u[target], top_n, rng)
    raw_esg_vol = stats.spearmanr(u["ESG_composite"], u["realized_vol_oos"], nan_policy="omit")
    primary = pd.DataFrame([
        {"hypothesis": "H1", "test": "Spearman IC(pref_balanced, country-neutral full-window return)",
         "estimate": rho1, "ci_low": lo1, "ci_high": hi1, "p_one_sided": p1, "n": n1},
        {"hypothesis": "H2", "test": "Partial Spearman(ESG_composite, realised OOS vol | trailing vol)",
         "estimate": r2, "ci_low": lo2, "ci_high": hi2, "p_one_sided": p2, "n": n2},
        {"hypothesis": "H3", "test": f"Top-{top_n} minus universe, country-neutral full-window return (pp)",
         "estimate": ex3, "ci_low": lo3, "ci_high": hi3, "p_one_sided": p3, "n": n3},
    ])
    primary["p_holm"] = holm(primary["p_one_sided"])
    primary["reject_h0_at_5pct"] = primary["p_holm"] < 0.05

    # Survivorship bounds: firms that traded at the snapshot but left the
    # market during the window (M&A exits) have no observable end price.
    # Assign them the 5th / 95th percentile of the country-neutral return
    # distribution and report the resulting range for H1 and H3.
    exited = u["ret_full"].isna()
    bound_rows = []
    for label, q in (("worst_case_p05", 0.05), ("best_case_p95", 0.95)):
        r = u[target].copy()
        r[exited] = u[target].quantile(q)
        rho_b = stats.spearmanr(u["pref_balanced"], r)[0]
        top = u["pref_balanced"].rank(ascending=False, method="first") <= top_n
        bound_rows.append({"scenario": label, "n_exited_imputed": int(exited.sum()),
                           "exited_tickers": ";".join(u.loc[exited, "ticker"]),
                           "exited_in_top_n": int((exited & top).sum()),
                           "H1_ic": rho_b, "H3_excess_pp": r[top].mean() - r.mean()})
    pd.DataFrame(bound_rows).to_csv(TABLES / "oos_survivorship_bounds.csv", index=False, encoding="utf-8")
    primary.to_csv(TABLES / "oos_primary_hypotheses.csv", index=False, encoding="utf-8")

    # ---------------- Secondary: IC table ----------------
    signals = FACTORS + ["ESG_orthogonal", "pref_balanced", "pref_esg_first", "pref_financial_first",
                         "pref_equal_weight", "pref_balanced_feb2026", "pref_balanced_pre_audit",
                         "ESG_composite_pre_audit"]
    signals = [s for s in signals if s in u.columns]
    ic_rows = []
    for sig in signals:
        for rc in ret_cols:
            for neut in ("cn", "scn"):
                col = f"{rc}_{neut}"
                rho, p, lo, hi, n = spearman_with_ci(u[sig], u[col], rng, n_boot=1000)
                spread, _ = quintile_spread(u[sig], u[col])
                ic_rows.append({"signal": sig, "horizon": rc.replace("ret_", ""),
                                "neutralization": {"cn": "country", "scn": "sector+country"}[neut],
                                "ic": rho, "ci_low": lo, "ci_high": hi, "p_two_sided": p,
                                "q5_minus_q1_pp": spread, "n": n})
    ic = pd.DataFrame(ic_rows)
    ic["p_holm_within_horizon"] = np.nan
    for (h, neut), g in ic.groupby(["horizon", "neutralization"]):
        ic.loc[g.index, "p_holm_within_horizon"] = holm(g["p_two_sided"])
    ic.to_csv(TABLES / "oos_information_coefficients.csv", index=False, encoding="utf-8")

    # ---------------- Portfolios ----------------
    port_rows = []
    for sig in ["pref_balanced", "pref_esg_first", "pref_financial_first", "pref_equal_weight",
                "pref_balanced_feb2026", "pref_balanced_pre_audit", "ESG_composite"]:
        if sig not in u:
            continue
        for rc in ret_cols:
            e, lo, hi, p, hit, n = portfolio_excess(u[sig], u[f"{rc}_cn"], top_n, rng, n_boot=2000)
            port_rows.append({"signal": sig, "horizon": rc.replace("ret_", ""), "top_n": top_n,
                              "excess_pp": e, "ci_low": lo, "ci_high": hi,
                              "p_one_sided": p, "hit_rate_vs_median": hit, "n": n})
        # Risk-adjusted variants of the full-window excess: market-model
        # (trailing beta) and sector+country neutral (removes sector tilts).
        for variant, col in (("full_beta_adj", "ret_full_ba_cn"), ("full_sector_neutral", "ret_full_scn")):
            e, lo, hi, p, hit, n = portfolio_excess(u[sig], u[col], top_n, rng, n_boot=2000)
            port_rows.append({"signal": sig, "horizon": variant, "top_n": top_n,
                              "excess_pp": e, "ci_low": lo, "ci_high": hi,
                              "p_one_sided": p, "hit_rate_vs_median": hit, "n": n})
    ports = pd.DataFrame(port_rows)
    ports.to_csv(TABLES / "oos_portfolio_excess.csv", index=False, encoding="utf-8")

    # ---------------- ESG as a risk filter (forward) ----------------
    risk_rows = []
    controls = {"realized_vol_oos": "price_volatility", "max_drawdown_oos": "max_drawdown_1y",
                "realized_beta_oos": "beta"}
    for esg_col in ["ESG_composite", "E_score", "S_score", "G_score", "ESG_orthogonal"]:
        for out_col, ctrl in controls.items():
            if esg_col not in u or out_col not in u:
                continue
            rho, p, lo, hi, n = spearman_with_ci(u[esg_col], u[out_col], rng, n_boot=1000)
            row = {"esg_measure": esg_col, "outcome": out_col, "spearman": rho, "ci_low": lo,
                   "ci_high": hi, "p_two_sided": p, "n": n, "control": ctrl}
            if ctrl in u:
                pr, pp, pn = partial_spearman(u[esg_col], u[out_col], u[ctrl])
                row.update({"partial_spearman": pr, "partial_p_two_sided": pp})
                row["trailing_vs_oos_spearman"] = stats.spearmanr(
                    u[ctrl], u[out_col], nan_policy="omit")[0]
            risk_rows.append(row)
    risk = pd.DataFrame(risk_rows)
    risk["partial_p_holm"] = holm(risk["partial_p_two_sided"].fillna(1.0))
    risk.to_csv(TABLES / "oos_esg_risk_filter.csv", index=False, encoding="utf-8")

    risk_regression_table(u).to_csv(TABLES / "oos_esg_risk_regression.csv", index=False, encoding="utf-8")

    # ---------------- Cross-sectional regression (HC3) ----------------
    reg = regression_table(u, target)
    reg.to_csv(TABLES / "oos_regression.csv", index=False, encoding="utf-8")

    # ---------------- Summary / metadata ----------------
    universe_ret = u.groupby("country")["ret_full"].mean()
    meta = pd.DataFrame([{
        "snapshot_date": str(snapshot.date()), "end_date": str(end.date()),
        "n_universe": n_universe, "n_with_full_return": int(u["ret_full"].notna().sum()),
        "n_stopped_early": len(early_stop), "top_n": top_n,
        "trading_days_us": int(u.loc[u["country"] == "US", "n_days_oos"].median()),
        "trading_days_india": int(u.loc[u["country"] == "India", "n_days_oos"].median()),
        "universe_mean_return_us_pct": universe_ret.get("US", np.nan),
        "universe_mean_return_india_pct": universe_ret.get("India", np.nan),
        "index_return_us_pct": idx_ret.get("US", np.nan),
        "index_return_india_pct": idx_ret.get("India", np.nan),
        "esg_r2_on_other_factors": esg_r2_on_factors,
        "esg_raw_spearman_with_oos_vol": raw_esg_vol.statistic if hasattr(raw_esg_vol, "statistic") else raw_esg_vol[0],
        "trailing_vs_oos_vol_spearman": stats.spearmanr(u["price_volatility"], u["realized_vol_oos"], nan_policy="omit")[0],
        "min_detectable_rho_80pct_power": min_detectable_rho(int(u["ret_full"].notna().sum())),
    }])
    meta.to_csv(TABLES / "oos_metadata.csv", index=False, encoding="utf-8")
    u[["ticker", "country", "sector"] + ret_cols + ["realized_vol_oos", "max_drawdown_oos",
       "realized_beta_oos", "n_days_oos"]].to_csv(TABLES / "oos_firm_outcomes.csv", index=False, encoding="utf-8")

    make_figure(u, ic, top_n)

    print("\n  PRIMARY HYPOTHESES (Holm-adjusted, one-sided):")
    for _, r in primary.iterrows():
        ci = "" if pd.isna(r["ci_low"]) else f" [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        print(f"    {r['hypothesis']}: {r['estimate']:+.3f}{ci}  p={r['p_one_sided']:.4f}  "
              f"p_holm={r['p_holm']:.4f}  n={r['n']}")
    print(f"\n  ESG R^2 on the other factors: {esg_r2_on_factors:.3f}")
    print("  [OK] Saved oos_primary_hypotheses.csv, oos_information_coefficients.csv, "
          "oos_portfolio_excess.csv, oos_esg_risk_filter.csv, oos_regression.csv, oos_metadata.csv")


def min_detectable_rho(n, alpha=0.05, power=0.80):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.tanh(z / np.sqrt(n - 3)))


def regression_table(u, target):
    """OLS of country-neutral return on standardized factors + sector/country FE, HC3 SE."""
    # statsmodels.api pulls in compiled tsa modules that some locked-down
    # Windows environments block; the OLS submodule is sufficient here.
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    cols = ["ESG_composite", "financial_score", "operational_score", "risk_adjusted_score",
            "value_score", "growth_score", "stability_score", "market_score"]
    d = u[[target, "sector", "country", "market_cap"] + cols].dropna()
    Xf = (d[cols] - d[cols].mean()) / d[cols].std()
    Xf["log_market_cap"] = np.log(d["market_cap"].clip(lower=1))
    Xf["log_market_cap"] = (Xf["log_market_cap"] - Xf["log_market_cap"].mean()) / Xf["log_market_cap"].std()
    fe = pd.get_dummies(d[["sector", "country"]], drop_first=True, dtype=float)
    rows = []
    for spec, use in [("with_esg", cols), ("without_esg", [c for c in cols if c != "ESG_composite"])]:
        X = add_constant(pd.concat([Xf[use + ["log_market_cap"]], fe], axis=1))
        fit = OLS(d[target], X).fit(cov_type="HC3")
        for name in use + ["log_market_cap"]:
            rows.append({"spec": spec, "variable": name, "coef_pp_per_sd": fit.params[name],
                         "se_hc3": fit.bse[name], "p_value": fit.pvalues[name],
                         "r2": fit.rsquared, "adj_r2": fit.rsquared_adj, "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def risk_regression_table(u):
    """Rank regression of realised OOS volatility on each ESG measure.

    Controls: trailing (pre-snapshot) volatility, trailing beta, log market
    cap (all as cross-sectional percentile ranks) plus sector and country
    fixed effects; HC3 standard errors.  Tests whether ESG carries forward
    risk information beyond what past risk and size already reveal.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    d = u.dropna(subset=["realized_vol_oos", "price_volatility", "market_cap"]).copy()
    pct = lambda s: s.rank() / len(s)
    fe = pd.get_dummies(d[["sector", "country"]], drop_first=True, dtype=float)
    rows = []
    for esg in ["ESG_composite", "E_score", "S_score", "G_score", "ESG_orthogonal"]:
        if esg not in d:
            continue
        X = pd.DataFrame({
            "esg": pct(d[esg]),
            "trailing_vol": pct(d["price_volatility"]),
            "trailing_beta": pct(d["beta"].fillna(d["beta"].median())),
            "log_size": pct(np.log(d["market_cap"].clip(lower=1))),
        }, index=d.index)
        fit = OLS(pct(d["realized_vol_oos"]), add_constant(pd.concat([X, fe], axis=1))).fit(cov_type="HC3")
        rows.append({"esg_measure": esg, "coef": fit.params["esg"], "se_hc3": fit.bse["esg"],
                     "p_value": fit.pvalues["esg"], "coef_trailing_vol": fit.params["trailing_vol"],
                     "coef_size": fit.params["log_size"], "r2": fit.rsquared, "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def make_figure(u, ic, top_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = ic[(ic["horizon"] == "full") & (ic["neutralization"] == "country")].copy()
    order = ["pref_balanced", "pref_esg_first", "pref_financial_first", "pref_equal_weight",
             "pref_balanced_feb2026", "pref_balanced_pre_audit", "ESG_composite", "ESG_orthogonal",
             "E_score", "S_score", "G_score", "financial_score", "operational_score",
             "risk_adjusted_score", "value_score", "growth_score", "stability_score",
             "market_score", "sector_position"]
    sub = sub.set_index("signal").reindex([o for o in order if o in set(sub["signal"])]).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    y = np.arange(len(sub))[::-1]
    colors = ["#1f5a96" if s.startswith("pref_") else "#7a7a7a" for s in sub["signal"]]
    ax.errorbar(sub["ic"], y, xerr=[sub["ic"] - sub["ci_low"], sub["ci_high"] - sub["ic"]],
                fmt="none", ecolor="#b0b0b0", elinewidth=1.4, capsize=2)
    ax.scatter(sub["ic"], y, c=colors, s=28, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["signal"], fontsize=8)
    ax.set_xlabel("Out-of-sample rank IC with country-neutral return (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_oos_ic.png", dpi=200)
    fig.savefig(FIGURES / "fig_oos_ic.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
