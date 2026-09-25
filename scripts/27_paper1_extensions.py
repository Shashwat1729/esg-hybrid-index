"""
Step 27: Paper 1 Extensions (secondary analyses on the out-of-sample window)
============================================================================
Extends the pre-declared out-of-sample test of step 25 with the analyses a
referee will ask for next.  Everything here uses the same frozen snapshot
scores (2026-04-02 close) and the same post-snapshot prices; nothing is
re-fitted on the outcome window except where a cross-validated split is
explicitly reported.  All results are SECONDARY / EXPLORATORY.

  P2  Is the ESG risk signal just a known factor?
      H4: ESG_composite is negatively related to realised out-of-sample
      volatility after controlling for trailing volatility, beta, size,
      quality (profitability), leverage, liquidity, momentum, value and
      sector + country fixed effects (rank regression, HC3).  Also a
      cross-validated forecast test (does adding ESG improve out-of-fold
      R^2 of realised volatility?) and a trailing-volatility x ESG double
      sort.
  P4  Measured vs proxy ESG.  Each firm's ESG cells are split by provenance
      (measured: ISS-via-Yahoo, SEC, public registry; proxy: derived from
      financial statements; imputed: sector/global medians) and the risk
      test is re-run on each sub-score.
  P6  Weight uncertainty.  Out-of-sample IC, top-N excess return and top-N
      realised volatility under (a) uniform Dirichlet weights over the
      simplex and (b) Dirichlet(kappa * w_deployed), plus an ESG-weight
      sweep (the return / risk trade-off of adding ESG).
  P8  Costs.  Top-N excess returns net of round-trip trading costs by
      country (base and 3x stress), against the universe and against the
      country index.
  T   Temporal stability.  Month-by-month ICs and ESG-volatility partial
      correlations inside the window.

Inputs:  data/processed/indexed_data.csv, data/raw/forward_prices.csv,
         reports/tables/oos_firm_outcomes.csv (step 25),
         reports/tables/esg_data_provenance.csv
Outputs: reports/tables/ext_*.csv, reports/figures/fig_ext_*.{png,pdf}
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import DEFAULT_WEIGHTS, RANDOM_SEED
from src.utils import get_portfolio_top_n

_spec = importlib.util.spec_from_file_location(
    "oos25", PROJECT_ROOT / "scripts" / "25_out_of_sample_evaluation.py")
oos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oos)

TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"

N_BOOT = 2000
N_DIRICHLET = 2000
KAPPA = 100

# One-way trading cost assumptions (bps).  US: 5 bps half-spread + 3 bps
# commission (as in step 16).  India: 10 bps securities transaction tax on
# delivery + ~10 bps half-spread for mid-caps + ~5 bps exchange/broker fees.
ONE_WAY_COST_BPS = {"US": 8.0, "India": 25.0}
COST_STRESS = 3.0

MEASURED = {"real_yahoo", "real_sec", "real_public_registry", "real_epa"}
PROXY = {"financial_proxy"}
IMPUTED = {"sector_imputed", "cross_sector_imputed"}
# Used indicators with no provenance column are financial by construction.
FINANCIAL_BY_CONSTRUCTION = {"revenue_per_employee"}

# Controls for H4 (all converted to cross-sectional percentile ranks).
CONTROLS = {
    "trailing_vol": "price_volatility",
    "trailing_beta": "beta",
    "size": "market_cap",
    "quality": "profitability_score",
    "leverage": "debt_to_equity",
    "liquidity": "log_dollar_volume",
    "momentum_6m": "price_momentum_6m",
    "value_pb": "price_to_book",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_frame():
    df = pd.read_csv(PROCESSED / "indexed_data.csv")
    uni = df[~df["is_large_cap_benchmark"].fillna(False).astype(bool)].copy()
    out = pd.read_csv(TABLES / "oos_firm_outcomes.csv").drop(columns=["country", "sector"])
    u = uni.merge(out, on="ticker", how="left")
    u["ret_full_cn"] = oos.neutralize(u, "ret_full", "country")
    u["ret_full_scn"] = oos.neutralize(u, "ret_full", ["country", "sector"])
    return df, u


def provenance_subscores(df_all, u):
    """Per-firm ESG sub-scores from measured / proxy / imputed cells only.

    Each used indicator's *_norm column (direction-corrected, as it enters
    the pillar scores) is converted to a percentile rank over all scored
    firms; a firm's sub-score is the mean over the cells of that class.
    """
    prov = pd.read_csv(TABLES / "esg_data_provenance.csv").set_index("ticker")
    norm_cols = [c for c in df_all.columns if c.endswith("_norm")
                 and c[:-5] in set(prov.columns) | FINANCIAL_BY_CONSTRUCTION]
    ranks = df_all.set_index("ticker")[norm_cols].rank(pct=True) * 100
    rows = {}
    for t in u["ticker"]:
        acc = {"measured": [], "proxy": [], "imputed": []}
        for c in norm_cols:
            ind = c[:-5]
            v = ranks.at[t, c] if t in ranks.index else np.nan
            if pd.isna(v):
                continue
            src = "financial_proxy" if ind in FINANCIAL_BY_CONSTRUCTION else (
                prov.at[t, ind] if t in prov.index else "missing")
            if src in MEASURED:
                acc["measured"].append(v)
            elif src in PROXY:
                acc["proxy"].append(v)
            elif src in IMPUTED:
                acc["imputed"].append(v)
        rows[t] = {f"ESG_{k}": (np.mean(v) if v else np.nan) for k, v in acc.items()}
        rows[t].update({f"n_{k}_cells": len(v) for k, v in acc.items()})
    sub = pd.DataFrame.from_dict(rows, orient="index").rename_axis("ticker").reset_index()
    return u.merge(sub, on="ticker", how="left"), len(norm_cols)


def pct(s):
    s = s.fillna(s.median())
    return s.rank() / len(s)


# ---------------------------------------------------------------------------
# P2: controlled risk tests
# ---------------------------------------------------------------------------
def controlled_risk_regressions(u, measures, outcomes):
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    rows = []
    for out_col, trailing in outcomes.items():
        d = u.dropna(subset=[out_col]).copy()
        fe = pd.get_dummies(d[["sector", "country"]], drop_first=True, dtype=float)
        base = pd.DataFrame({k: pct(d[v]) for k, v in CONTROLS.items()}, index=d.index)
        own = {v: k for k, v in CONTROLS.items()}.get(trailing)
        if own is None:
            own = "trailing_outcome"
            base[own] = pct(d[trailing])
        for spec in ("trailing_only", "full_controls"):
            X0 = base[[own]] if spec == "trailing_only" else base
            for m in measures:
                if m not in d or d[m].notna().sum() < 50:
                    continue
                X = pd.concat([pd.DataFrame({"esg": pct(d[m])}, index=d.index), X0, fe], axis=1)
                fit = OLS(pct(d[out_col]), add_constant(X)).fit(cov_type="HC3")
                rows.append({"outcome": out_col, "spec": spec, "esg_measure": m,
                             "coef": fit.params["esg"], "se_hc3": fit.bse["esg"],
                             "ci_low": fit.conf_int().loc["esg", 0], "ci_high": fit.conf_int().loc["esg", 1],
                             "p_two_sided": fit.pvalues["esg"],
                             "p_one_sided_neg": stats.norm.cdf(fit.params["esg"] / fit.bse["esg"]),
                             "r2": fit.rsquared, "n": int(fit.nobs)})
    res = pd.DataFrame(rows)
    res["p_holm_within_outcome_spec"] = np.nan
    for _, g in res.groupby(["outcome", "spec"]):
        res.loc[g.index, "p_holm_within_outcome_spec"] = oos.holm(g["p_two_sided"])
    return res


def joint_provenance_regression(u):
    """Measured and proxy ESG entered together with full controls."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    d = u.dropna(subset=["realized_vol_oos"]).copy()
    fe = pd.get_dummies(d[["sector", "country"]], drop_first=True, dtype=float)
    X = pd.DataFrame({"esg_measured": pct(d["ESG_measured"]), "esg_proxy": pct(d["ESG_proxy"])},
                     index=d.index)
    X = pd.concat([X, pd.DataFrame({k: pct(d[v]) for k, v in CONTROLS.items()}, index=d.index), fe], axis=1)
    fit = OLS(pct(d["realized_vol_oos"]), add_constant(X)).fit(cov_type="HC3")
    return pd.DataFrame([{"variable": v, "coef": fit.params[v], "se_hc3": fit.bse[v],
                          "p_two_sided": fit.pvalues[v], "r2": fit.rsquared, "n": int(fit.nobs)}
                         for v in ["esg_measured", "esg_proxy"] + list(CONTROLS)])


def cv_vol_forecast(u, rng, n_repeats=50, k=10):
    """Out-of-fold R^2 for realised-volatility ranks, with and without ESG."""
    d = u.dropna(subset=["realized_vol_oos"]).reset_index(drop=True)
    y = pct(d["realized_vol_oos"]).to_numpy()
    fe = pd.get_dummies(d[["sector", "country"]], drop_first=True, dtype=float).to_numpy()
    ctl_small = np.column_stack([pct(d[c]) for c in ("price_volatility", "beta", "market_cap")])
    ctl_full = np.column_stack([pct(d[v]) for v in CONTROLS.values()])
    specs = {
        "baseline_small": ctl_small,
        "baseline_small+ESG": np.column_stack([ctl_small, pct(d["ESG_composite"])]),
        "baseline_full": ctl_full,
        "baseline_full+ESG": np.column_stack([ctl_full, pct(d["ESG_composite"])]),
        "baseline_full+ESG_measured": np.column_stack([ctl_full, pct(d["ESG_measured"])]),
        "baseline_full+ESG_proxy": np.column_stack([ctl_full, pct(d["ESG_proxy"])]),
    }
    n = len(y)
    res = {s: [] for s in specs}
    for _ in range(n_repeats):
        folds = rng.permutation(n) % k
        preds = {s: np.empty(n) for s in specs}
        for f in range(k):
            tr, te = folds != f, folds == f
            for s, Xs in specs.items():
                X = np.column_stack([np.ones(n), Xs, fe])
                b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                preds[s][te] = X[te] @ b
        for s in specs:
            res[s].append(1 - np.sum((y - preds[s]) ** 2) / np.sum((y - y.mean()) ** 2))
    rows = []
    for s, v in res.items():
        v = np.asarray(v)
        base = "baseline_small" if s.startswith("baseline_small") else "baseline_full"
        dv = v - np.asarray(res[base])
        rows.append({"spec": s, "oof_r2_mean": v.mean(), "oof_r2_sd": v.std(ddof=1),
                     "delta_vs_baseline_mean": dv.mean(), "delta_p05": np.percentile(dv, 5),
                     "delta_p95": np.percentile(dv, 95), "share_repeats_improved": float(np.mean(dv > 0)),
                     "n": n, "repeats": n_repeats, "folds": k})
    return pd.DataFrame(rows)


def double_sort(u, rng, n_boot=N_BOOT):
    """Within trailing-vol terciles (per country): high- minus low-ESG realised vol."""
    d = u.dropna(subset=["realized_vol_oos", "price_volatility", "ESG_composite"]).copy()
    d["vol_t"] = d.groupby("country")["price_volatility"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    d["esg_hi"] = d.groupby(["country", "vol_t"])["ESG_composite"].transform(lambda s: s > s.median())

    def stat(x):
        g = x.groupby(["vol_t", "esg_hi"])["realized_vol_oos"].mean().unstack()
        return (g[True] - g[False])

    point = stat(d)
    draws = []
    for _ in range(n_boot):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        draws.append(stat(s).mean())
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    rows = [{"trailing_vol_tercile": int(t) + 1, "high_minus_low_esg_vol_pp": v,
             "mean_trailing_vol": d.loc[d["vol_t"] == t, "price_volatility"].mean(),
             "n": int((d["vol_t"] == t).sum())} for t, v in point.items()]
    rows.append({"trailing_vol_tercile": "average", "high_minus_low_esg_vol_pp": point.mean(),
                 "ci_low": lo, "ci_high": hi, "p_one_sided": float(np.mean(np.asarray(draws) >= 0)),
                 "n": len(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# P6: weight uncertainty
# ---------------------------------------------------------------------------
def weight_uncertainty(u, top_n, rng):
    factors = [k for k, v in DEFAULT_WEIGHTS.items() if v > 0 and k in u]
    w0 = np.array([DEFAULT_WEIGHTS[k] for k in factors])
    w0 = w0 / w0.sum()
    d = u.dropna(subset=["ret_full_cn"]).reset_index(drop=True)
    R = np.column_stack([pct(d[f]).to_numpy() for f in factors])
    ret = d["ret_full_cn"].to_numpy()
    vol = d["realized_vol_oos"].to_numpy()
    ret_rank = stats.rankdata(ret)

    def evaluate(W):
        comp = R @ W.T                                  # n x draws
        ranks = np.apply_along_axis(stats.rankdata, 0, comp)
        ic = np.array([np.corrcoef(ranks[:, j], ret_rank)[0, 1] for j in range(W.shape[0])])
        top = np.argsort(-comp, axis=0)[:top_n]
        ex = ret[top].mean(axis=0) - ret.mean()
        vx = np.nanmean(vol[top], axis=0) - np.nanmean(vol)
        return ic, ex, vx

    esg_i = factors.index("ESG_composite")
    draws = {"uniform_simplex": rng.dirichlet(np.ones(len(factors)), N_DIRICHLET),
             f"dirichlet_kappa{KAPPA}": rng.dirichlet(KAPPA * w0, N_DIRICHLET)}
    summ, per_draw = [], []
    ic0, ex0, vx0 = evaluate(w0[None, :])
    for name, W in draws.items():
        ic, ex, vx = evaluate(W)
        summ.append({"scheme": name, "draws": len(W), "ic_deployed": ic0[0],
                     "ic_median": np.median(ic), "ic_p025": np.percentile(ic, 2.5),
                     "ic_p975": np.percentile(ic, 97.5), "share_ic_positive": float(np.mean(ic > 0)),
                     "excess_median_pp": np.median(ex), "excess_p025": np.percentile(ex, 2.5),
                     "excess_p975": np.percentile(ex, 97.5), "share_excess_positive": float(np.mean(ex > 0)),
                     "topn_vol_diff_median_pp": np.median(vx), "share_topn_lower_vol": float(np.mean(vx < 0)),
                     "corr_esg_weight_ic": stats.spearmanr(W[:, esg_i], ic)[0],
                     "corr_esg_weight_topn_vol": stats.spearmanr(W[:, esg_i], vx)[0]})
        per_draw.append(pd.DataFrame({"scheme": name, "w_esg": W[:, esg_i], "ic": ic,
                                      "excess_pp": ex, "topn_vol_diff_pp": vx}))
    summary = pd.DataFrame(summ)

    # ESG weight sweep: others keep their deployed proportions.
    others = np.delete(w0, esg_i) / np.delete(w0, esg_i).sum()
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    W = np.array([np.insert(others * (1 - g), esg_i, g) for g in grid])
    ic, ex, vx = evaluate(W)
    boot_ic = []
    comp_all = R @ W.T
    for _ in range(500):
        i = rng.integers(0, len(d), len(d))
        rr = stats.rankdata(ret[i])
        boot_ic.append([np.corrcoef(stats.rankdata(comp_all[i, j]), rr)[0, 1] for j in range(len(grid))])
    boot_ic = np.asarray(boot_ic)
    sweep = pd.DataFrame({"w_esg": grid, "ic": ic, "ic_p025": np.percentile(boot_ic, 2.5, axis=0),
                          "ic_p975": np.percentile(boot_ic, 97.5, axis=0),
                          "topn_excess_pp": ex, "topn_vol_diff_pp": vx,
                          "is_deployed": np.isclose(grid, round(w0[esg_i] / 0.05) * 0.05)})
    return summary, sweep, pd.concat(per_draw, ignore_index=True), w0[esg_i]


# ---------------------------------------------------------------------------
# P8: costs
# ---------------------------------------------------------------------------
def cost_adjusted(u, top_n):
    d = u.dropna(subset=["ret_full"]).copy()
    d["rt_cost_pct"] = d["country"].map(ONE_WAY_COST_BPS).fillna(ONE_WAY_COST_BPS["US"]) * 2 / 100
    # Excess over own country index (investable passive alternative).
    idx_meta = pd.read_csv(TABLES / "oos_metadata.csv").iloc[0]
    idx = {"US": idx_meta["index_return_us_pct"], "India": idx_meta["index_return_india_pct"]}
    d["ret_vs_index"] = d["ret_full"] - d["country"].map(idx)
    rows = []
    for sig in ["pref_balanced", "pref_esg_first", "pref_financial_first", "ESG_composite"]:
        top = d[sig].rank(ascending=False, method="first") <= top_n
        gross_u = d.loc[top, "ret_full_cn"].mean() - d["ret_full_cn"].mean()
        gross_i = d.loc[top, "ret_vs_index"].mean()
        cost = d.loc[top, "rt_cost_pct"].mean()
        for scen, mult in (("base", 1.0), ("stress_3x", COST_STRESS)):
            rows.append({"signal": sig, "top_n": top_n, "cost_scenario": scen,
                         "round_trip_cost_pp": cost * mult,
                         "gross_excess_vs_universe_pp": gross_u, "net_excess_vs_universe_pp": gross_u - cost * mult,
                         "gross_excess_vs_country_index_pp": gross_i,
                         "net_excess_vs_country_index_pp": gross_i - cost * mult,
                         "n_us": int((top & (d["country"] == "US")).sum()),
                         "n_india": int((top & (d["country"] == "India")).sum()),
                         "one_way_bps_us": ONE_WAY_COST_BPS["US"] * mult,
                         "one_way_bps_india": ONE_WAY_COST_BPS["India"] * mult})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Temporal: month-by-month
# ---------------------------------------------------------------------------
def monthly_stability(u, snapshot, end):
    prices = pd.read_csv(oos.PRICE_CACHE, parse_dates=["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    bounds = sorted({min(snapshot + pd.DateOffset(months=k), end) for k in range(0, 7)})
    tick = u.set_index("ticker")
    rows = []
    for k in range(1, len(bounds)):
        a, b = bounds[k - 1], bounds[k]
        pa = wide[wide.index <= a].ffill().iloc[-1]
        pb = wide[wide.index <= b].ffill().iloc[-1]
        seg = wide[(wide.index > a) & (wide.index <= b)]
        r = ((pb / pa - 1) * 100).reindex(tick.index)
        daily = wide[(wide.index >= a) & (wide.index <= b)].pct_change(fill_method=None).iloc[1:]
        mvol = (daily.std() * np.sqrt(252) * 100).reindex(tick.index)
        few = (daily.notna().sum() < 10).reindex(tick.index, fill_value=True).astype(bool)
        mvol[few] = np.nan
        m = pd.DataFrame({"ret": r, "vol": mvol, "country": tick["country"]})
        m["ret_cn"] = m["ret"] - m.groupby("country")["ret"].transform("mean")
        ic_bal = stats.spearmanr(tick["pref_balanced"], m["ret_cn"], nan_policy="omit")[0]
        ic_esg = stats.spearmanr(tick["ESG_composite"], m["ret_cn"], nan_policy="omit")[0]
        pr, pp, pn = oos.partial_spearman(tick["ESG_composite"], m["vol"], tick["price_volatility"])
        rows.append({"month": k, "start": a.date(), "end": b.date(), "trading_days": len(seg),
                     "ic_pref_balanced": ic_bal, "ic_esg_composite": ic_esg,
                     "partial_esg_vol_given_trailing": pr, "partial_p_two_sided": pp, "n": pn})
    res = pd.DataFrame(rows)
    agg = {"month": "mean_tstat"}
    for c in ("ic_pref_balanced", "ic_esg_composite", "partial_esg_vol_given_trailing"):
        v = res[c].dropna()
        agg[c] = v.mean()
        agg[c + "_t"] = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        agg[c + "_n_negative"] = int((v < 0).sum())
    return res, pd.DataFrame([agg])


def recent_vol_control(u, snapshot, end, rng):
    """Is ESG just a slow-moving volatility proxy?

    Splits the window at its midpoint and asks whether ESG predicts
    second-half realised volatility once first-half realised volatility
    (information the scores never saw, but an investor would have) is
    controlled for, alongside the pre-snapshot trailing volatility.
    """
    prices = pd.read_csv(oos.PRICE_CACHE, parse_dates=["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    mid = snapshot + (end - snapshot) / 2
    tick = u.set_index("ticker")

    def vol(a, b):
        daily = wide[(wide.index >= a) & (wide.index <= b)].pct_change(fill_method=None).iloc[1:]
        v = daily.std() * np.sqrt(252) * 100
        v[daily.notna().sum() < 40] = np.nan
        return v.reindex(tick.index)

    v1, v2 = vol(snapshot, mid), vol(mid, end)
    esg, trail = tick["ESG_composite"], tick["price_volatility"]
    rows = []
    for label, z in (("pre_snapshot_trailing_vol", trail), ("first_half_realised_vol", v1)):
        r, p, n = oos.partial_spearman(esg, v2, z)
        lo, hi = oos.partial_spearman_ci(esg, v2, z, rng, n_boot=1000)
        rows.append({"control": label, "partial_rho_esg_second_half_vol": r, "ci_low": lo,
                     "ci_high": hi, "p_two_sided": p, "n": n,
                     "control_vs_outcome_spearman": stats.spearmanr(z, v2, nan_policy="omit")[0]})
    # Both controls jointly (rank-residualise on the two), with firm bootstrap CI.
    m = (esg.notna() & v2.notna() & v1.notna() & trail.notna()).to_numpy()
    arr = np.column_stack([esg.to_numpy()[m], v2.to_numpy()[m], trail.to_numpy()[m], v1.to_numpy()[m]])

    def partial2(a):
        rk = [stats.rankdata(a[:, j]) for j in range(4)]
        Z = np.column_stack([np.ones(len(a)), rk[2], rk[3]])
        res = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return np.corrcoef(res(rk[0]), res(rk[1]))[0, 1]

    r = partial2(arr)
    n = len(arr)
    t = r * np.sqrt((n - 4) / (1 - r ** 2))
    boots = [partial2(arr[rng.integers(0, n, n)]) for _ in range(1000)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    rows.append({"control": "both", "partial_rho_esg_second_half_vol": r, "ci_low": lo, "ci_high": hi,
                 "p_two_sided": 2 * stats.t.sf(abs(t), n - 4), "n": n})
    out = pd.DataFrame(rows)
    out.insert(0, "split_date", str(mid.date()))
    return out


PINNED_RATIO = 0.25  # OOS vol below this fraction of trailing vol => price pinned by a deal


def subsample_robustness(u, rng):
    """H2 (partial Spearman) and H4 (full-control rank regression) on subsamples."""
    ratio = u["realized_vol_oos"] / u["price_volatility"]
    pinned = ratio < PINNED_RATIO
    lo_w, hi_w = u["realized_vol_oos"].quantile([0.01, 0.99])
    wins = u.assign(realized_vol_oos=u["realized_vol_oos"].clip(lo_w, hi_w))
    samples = {
        "all (pre-declared)": u,
        "excl. deal-pinned": u[~pinned],
        "winsorised vol 1/99": wins,
        "US only": u[u["country"] == "US"],
        "India only": u[u["country"] == "India"],
        "excl. Utilities, Real Estate": u[~u["sector"].isin(["Utilities", "Real Estate"])],
    }
    rows = []
    for name, d in samples.items():
        d = d.copy()
        r, p, n = oos.partial_spearman(d["ESG_composite"], d["realized_vol_oos"], d["price_volatility"])
        lo, hi = oos.partial_spearman_ci(d["ESG_composite"], d["realized_vol_oos"], d["price_volatility"],
                                         rng, n_boot=1000)
        reg = controlled_risk_regressions(d, ["ESG_composite"], {"realized_vol_oos": "price_volatility"})
        h4 = reg[reg["spec"] == "full_controls"].iloc[0]
        rows.append({"sample": name, "n": n, "h2_partial_rho": r, "h2_ci_low": lo, "h2_ci_high": hi,
                     "h2_p_two_sided": p, "h4_coef": h4["coef"], "h4_ci_low": h4["ci_low"],
                     "h4_ci_high": h4["ci_high"], "h4_p_two_sided": h4["p_two_sided"],
                     "excluded_tickers": ";".join(u.loc[pinned, "ticker"]) if name == "excl. deal-pinned" else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_mechanism_figure(ctrl, rv, dry):
    """Headline RQ4 figure: where the ESG risk signal lives and how it fades."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = {"ESG_composite": "ESG composite", "E_score": "E pillar", "S_score": "S pillar",
              "G_score": "G pillar", "ESG_measured": "measured cells", "ESG_proxy": "proxy cells",
              "ESG_imputed": "imputed cells"}
    colors = {"ESG_composite": "#1f5a96", "E_score": "#7a7a7a", "S_score": "#7a7a7a", "G_score": "#7a7a7a",
              "ESG_measured": "#1b7837", "ESG_proxy": "#b35806", "ESG_imputed": "#9e9e9e"}
    sub = ctrl[(ctrl.outcome == "realized_vol_oos") & (ctrl.spec == "full_controls")].set_index("esg_measure")
    order = [m for m in labels if m in sub.index]
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        r = sub.loc[m]
        ax.errorbar(r["coef"], yi, xerr=[[r["coef"] - r["ci_low"]], [r["ci_high"] - r["coef"]]],
                    fmt="o", color=colors[m], ms=4, elinewidth=1.3, capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(y[3] - 0.5, color="#cccccc", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in order], fontsize=7)
    ax.set_xlabel("ESG coefficient on realised-vol rank\n(full controls, 95% CI)", fontsize=8)
    ax.set_title("(a) which ESG data carry the signal", fontsize=8, loc="left")

    ax = axes[1]
    rows = [("pre-snapshot\nvol.", rv.loc["pre_snapshot_trailing_vol"]),
            ("first-half\nvol.", rv.loc["first_half_realised_vol"]),
            ("both", rv.loc["both"])]
    xs = np.arange(len(rows))
    for x, (lab, r) in zip(xs, rows):
        ax.errorbar(x, r["partial_rho_esg_second_half_vol"],
                    yerr=[[r["partial_rho_esg_second_half_vol"] - r["ci_low"]],
                          [r["ci_high"] - r["partial_rho_esg_second_half_vol"]]],
                    fmt="o", color="#1f5a96", ms=4, elinewidth=1.3, capsize=2)
    if dry is not None:
        ax.errorbar(len(rows), dry["estimate"], yerr=[[dry["estimate"] - dry["ci_low"]],
                                                      [dry["ci_high"] - dry["estimate"]]],
                    fmt="s", color="#b35806", ms=4, elinewidth=1.3, capsize=2)
        rows.append(("recent 3m vol.\n(dry run)", None))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("control for earlier volatility", fontsize=8)
    ax.set_ylabel("partial $\\rho$(ESG, later vol.)", fontsize=8)
    ax.set_title("(b) the signal fades given recent volatility", fontsize=8, loc="left")
    for a in axes:
        a.tick_params(labelsize=7)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig_ext_mechanism.png", dpi=200)
    fig.savefig(FIGURES / "fig_ext_mechanism.pdf")
    plt.close(fig)


def make_figure(per_draw, sweep, w_esg_deployed, ic_deployed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    uni = per_draw[per_draw["scheme"] == "uniform_simplex"]
    ax = axes[0]
    ax.hist(uni["ic"], bins=40, color="#9bb7d4", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(ic_deployed, color="#b2182b", lw=1.4, ls="--", label="deployed")
    ax.set_xlabel("OOS IC, random weights")
    ax.set_ylabel("draws")
    ax.set_title("(a)", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ax.fill_between(sweep["w_esg"], sweep["ic_p025"], sweep["ic_p975"], color="#9bb7d4", alpha=0.5, lw=0)
    ax.plot(sweep["w_esg"], sweep["ic"], color="#1f5a96", marker="o", ms=2.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(w_esg_deployed, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("ESG weight")
    ax.set_ylabel("OOS IC (95% CI)")
    ax.set_title("(b)", fontsize=8, loc="left")
    ax = axes[2]
    ax.plot(sweep["w_esg"], sweep["topn_excess_pp"], color="#762a83", marker="s", ms=2.5, label="excess return")
    ax.plot(sweep["w_esg"], sweep["topn_vol_diff_pp"], color="#1b7837", marker="o", ms=2.5, label="realised vol")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(w_esg_deployed, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("ESG weight")
    ax.set_ylabel("top-N minus universe (pp)")
    ax.set_title("(c)", fontsize=8, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_ext_weights.png", dpi=200)
    fig.savefig(FIGURES / "fig_ext_weights.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    snapshot, end, _ = oos.load_oos_config()
    print("=" * 70)
    print("STEP 27: PAPER 1 EXTENSIONS (secondary, out-of-sample window)")
    print("=" * 70)
    rng = np.random.default_rng(RANDOM_SEED)
    df_all, u = load_frame()
    u, n_used = provenance_subscores(df_all, u)
    top_n = get_portfolio_top_n(len(u))

    # ---- P4: provenance sub-scores ----
    others = ["financial_score", "operational_score", "risk_adjusted_score",
              "value_score", "growth_score", "stability_score"]
    X = np.column_stack([np.ones(len(u))] + [u[c].fillna(u[c].median()) for c in others])
    prov_rows = []
    for m in ["ESG_composite", "ESG_measured", "ESG_proxy", "ESG_imputed"]:
        y = u[m]
        ok = y.notna().to_numpy()
        b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        r2 = 1 - np.var(y[ok] - X[ok] @ b) / np.var(y[ok])
        pr, pp, pn = oos.partial_spearman(u[m], u["realized_vol_oos"], u["price_volatility"])
        lo, hi = oos.partial_spearman_ci(u[m], u["realized_vol_oos"], u["price_volatility"], rng, n_boot=1000)
        cells = u.get(f"n_{m.split('_')[1]}_cells") if m != "ESG_composite" else None
        prov_rows.append({"measure": m, "firms_with_score": int(y.notna().sum()),
                          "mean_cells_per_firm": float(cells.mean()) if cells is not None else float(n_used),
                          "r2_on_financial_factors": r2,
                          "spearman_with_esg_composite": stats.spearmanr(y, u["ESG_composite"], nan_policy="omit")[0],
                          "partial_rho_oos_vol": pr, "ci_low": lo, "ci_high": hi,
                          "p_two_sided": pp, "n": pn})
    prov = pd.DataFrame(prov_rows)
    prov["spearman_measured_vs_proxy"] = stats.spearmanr(u["ESG_measured"], u["ESG_proxy"], nan_policy="omit")[0]
    prov.to_csv(TABLES / "ext_provenance_split.csv", index=False, encoding="utf-8")
    joint_provenance_regression(u).to_csv(TABLES / "ext_provenance_joint.csv", index=False, encoding="utf-8")

    # ---- P2: controlled risk tests ----
    measures = ["ESG_composite", "E_score", "S_score", "G_score", "ESG_measured", "ESG_proxy", "ESG_imputed"]
    ctrl = controlled_risk_regressions(
        u, measures, {"realized_vol_oos": "price_volatility", "max_drawdown_oos": "max_drawdown_1y",
                      "realized_beta_oos": "beta"})
    ctrl.to_csv(TABLES / "ext_risk_controls.csv", index=False, encoding="utf-8")
    cv = cv_vol_forecast(u, rng)
    cv.to_csv(TABLES / "ext_vol_forecast_cv.csv", index=False, encoding="utf-8")
    ds = double_sort(u, rng)
    rob = subsample_robustness(u, rng)
    rob.to_csv(TABLES / "ext_subsample_robustness.csv", index=False, encoding="utf-8")
    print(rob.drop(columns="excluded_tickers").round(3).to_string(index=False))
    ds.to_csv(TABLES / "ext_double_sort.csv", index=False, encoding="utf-8")

    # ---- P6: weights ----
    wsum, sweep, per_draw, w_esg = weight_uncertainty(u, top_n, rng)
    wsum.to_csv(TABLES / "ext_weight_uncertainty.csv", index=False, encoding="utf-8")
    sweep.to_csv(TABLES / "ext_esg_weight_sweep.csv", index=False, encoding="utf-8")
    make_figure(per_draw, sweep, w_esg, wsum["ic_deployed"].iloc[0])

    # ---- P8: costs ----
    cost_adjusted(u, top_n).to_csv(TABLES / "ext_costs.csv", index=False, encoding="utf-8")

    # ---- Temporal ----
    rv = recent_vol_control(u, snapshot, end, rng)
    rv.to_csv(TABLES / "ext_recent_vol_control.csv", index=False, encoding="utf-8")
    print(rv.round(3).to_string(index=False))
    dry_path = PROJECT_ROOT / "preregistration" / "window2" / "dry_run_results.csv"
    dry = pd.read_csv(dry_path).set_index("hypothesis").loc["H2"] if dry_path.exists() else None
    make_mechanism_figure(ctrl, rv.set_index("control"), dry)
    monthly, magg = monthly_stability(u, snapshot, end)
    monthly.to_csv(TABLES / "ext_monthly.csv", index=False, encoding="utf-8")
    magg.to_csv(TABLES / "ext_monthly_summary.csv", index=False, encoding="utf-8")

    h4 = ctrl[(ctrl["outcome"] == "realized_vol_oos") & (ctrl["spec"] == "full_controls")
              & (ctrl["esg_measure"] == "ESG_composite")].iloc[0]
    print(f"  H4 (ESG -> OOS vol | full controls): coef={h4['coef']:+.3f} "
          f"[{h4['ci_low']:+.3f}, {h4['ci_high']:+.3f}] p1={h4['p_one_sided_neg']:.4f}")
    print(prov[["measure", "mean_cells_per_firm", "r2_on_financial_factors", "partial_rho_oos_vol", "p_two_sided"]]
          .to_string(index=False))
    print(cv[["spec", "oof_r2_mean", "delta_vs_baseline_mean", "share_repeats_improved"]].to_string(index=False))
    print(wsum[["scheme", "ic_median", "ic_p025", "ic_p975", "share_ic_positive",
                "share_topn_lower_vol"]].to_string(index=False))
    print(monthly.to_string(index=False))
    print("  [OK] Saved ext_*.csv and fig_ext_weights")


if __name__ == "__main__":
    main()
