"""
Step 28: Pre-registered confirmatory test (window 2)
====================================================
Freezes the scores and the analysis plan BEFORE the outcome window opens,
and evaluates them unchanged after it closes.

Window 2: returns and risk over (ANCHOR, END] = (2026-09-28, 2027-03-31].
Scores: the frozen 2026-04-02 snapshot (data/processed/indexed_data.csv),
i.e. a 6-12 month-ahead test of the same index.  Risk controls are
refreshed to what is observable at the anchor: realised volatility, beta and
return over the six months before the anchor.

Registered hypotheses (one-sided, alpha = 0.05, Holm across H1-H4):
  H1  IC(pref_balanced, country-neutral return) > 0.
  H2  Partial Spearman(ESG_composite, realised vol | trailing 6m vol) < 0.
  H3  Top-19 pref_balanced minus universe, country-neutral return > 0.
  H4  In a rank regression of realised vol on ESG_composite plus trailing
      6m vol, trailing 6m beta, size, quality, leverage, liquidity,
      trailing 6m return, value and sector + country fixed effects (HC3),
      the ESG coefficient < 0.

Usage:
  python scripts/28_preregistration.py freeze            # write package + hashes
  python scripts/28_preregistration.py verify            # re-check hashes
  python scripts/28_preregistration.py evaluate --dry-run  # test the code on window 1 data
  python scripts/28_preregistration.py evaluate          # only after END + 3 days
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.constants import RANDOM_SEED

_spec = importlib.util.spec_from_file_location(
    "oos25", PROJECT_ROOT / "scripts" / "25_out_of_sample_evaluation.py")
oos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oos)

PREREG = PROJECT_ROOT / "preregistration" / "window2"
SNAPSHOT = pd.Timestamp("2026-04-02")
ANCHOR = pd.Timestamp("2026-09-28")
END = pd.Timestamp("2027-03-31")
TRAILING_MONTHS = 6
TOP_N = 19
N_BOOT = 5000
ALPHA = 0.05
# Dry run: same code on an already observed sub-window of window 1.
DRY_ANCHOR = pd.Timestamp("2026-07-01")
DRY_END = pd.Timestamp("2026-09-23")

FROZEN_COLS = ["ticker", "company_name", "country", "sector", "pref_balanced", "ESG_composite",
               "E_score", "S_score", "G_score", "market_cap", "profitability_score",
               "debt_to_equity", "log_dollar_volume", "price_to_book"]
HASHED_CODE = ["scripts/25_out_of_sample_evaluation.py", "scripts/28_preregistration.py",
               "src/constants.py", "src/utils.py"]


def sha256(path: Path) -> str:
    """SHA-256 of the file with CRLF normalised to LF.

    Normalising line endings makes the hash identical on Windows and POSIX
    checkouts of the same committed file (git may rewrite line endings).
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty(paths) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--"] + list(paths), text=True)
        return bool(out.strip())
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def freeze():
    today = pd.Timestamp(date.today())
    if today >= ANCHOR:
        sys.exit(f"Refusing to freeze: today ({today.date()}) is not before the anchor {ANCHOR.date()}.")
    PREREG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "indexed_data.csv")
    uni = df[~df["is_large_cap_benchmark"].fillna(False).astype(bool)]
    frozen = uni[FROZEN_COLS].sort_values("ticker").reset_index(drop=True)
    frozen_path = PREREG / "frozen_scores.csv"
    frozen.to_csv(frozen_path, index=False, encoding="utf-8", lineterminator="\n")

    manifest = {
        "frozen_on": str(today.date()),
        "git_head_at_freeze": git_head(),
        "code_uncommitted_at_freeze": git_dirty(HASHED_CODE),
        "snapshot_date": str(SNAPSHOT.date()), "anchor": str(ANCHOR.date()), "end": str(END.date()),
        "top_n": TOP_N, "n_boot": N_BOOT, "seed": RANDOM_SEED, "alpha": ALPHA,
        "n_firms": len(frozen),
        "sha256": {"preregistration/window2/frozen_scores.csv": sha256(frozen_path),
                   **{p: sha256(PROJECT_ROOT / p) for p in HASHED_CODE}},
    }
    (PREREG / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PREREG / "REGISTRATION.md").write_text(registration_text(manifest), encoding="utf-8")
    print(f"  Frozen {len(frozen)} firms -> {PREREG}")
    print("  Next: commit preregistration/ and upload REGISTRATION.md + manifest.json + "
          "frozen_scores.csv to OSF before the anchor close.")


def registration_text(m):
    hashes = "\n".join(f"| `{k}` | `{v}` |" for k, v in m["sha256"].items())
    return f"""# Pre-registration: out-of-sample test of a public-data ESG mid-cap index (window 2)

Frozen on {m['frozen_on']}, before the outcome window opens.  Git HEAD at freeze:
`{m['git_head_at_freeze']}`.

## 1. Background

Paper 1 (\"What Does Public-Data ESG Measure?\") built a multi-factor index for
{m['n_firms']} US and Indian mid-caps from data observable at the {m['snapshot_date']} close and
tested it on returns over (2026-04-02, 2026-09-23].  Result: no return
predictability (H1, H3 not supported); higher ESG predicted lower realised
volatility given trailing volatility (H2 supported, Holm p = 0.013).  The
window-1 test was declared in code before prices were downloaded but was not
registered with a third party.  This registration makes the next window a
confirmatory test.

## 2. Data frozen now

* `frozen_scores.csv`: the {m['n_firms']} mid-cap firms, their frozen composite
  (`pref_balanced`), ESG composite and pillar scores, and the snapshot-date
  fundamentals used as controls.  No score will be recomputed.
* Prices: Yahoo Finance daily adjusted closes, downloaded after {m['end']} by
  `python scripts/28_preregistration.py evaluate`.

## 3. Window and outcomes

* Anchor: close of **{m['anchor']}**.  End: close of **{m['end']}**.
* Return: local-currency total return from the last close on or before the
  anchor to the last close on or before the end; made country-neutral by
  demeaning within country (US, India).
* Realised volatility: annualised s.d. of daily returns in the window
  (at least 20 observations).
* Trailing controls, measured over the {TRAILING_MONTHS} months up to the anchor:
  realised volatility, beta against the country index (S&P 500, NIFTY 50),
  and total return.

## 4. Hypotheses (one-sided, alpha = {m['alpha']}, Holm across H1-H4)

| | Hypothesis | Test |
|---|---|---|
| H1 | The composite ranks returns | Spearman IC(pref_balanced, country-neutral return) > 0; firm bootstrap CI |
| H2 | ESG predicts lower risk | Partial Spearman(ESG_composite, realised vol given trailing vol) < 0 |
| H3 | The top-{m['top_n']} portfolio outperforms | Equal-weight top-{m['top_n']} minus equal-weight universe, country-neutral > 0 |
| H4 | The risk signal survives known factors | Rank-regression coefficient on ESG_composite < 0, controls: trailing vol, trailing beta, size, quality (profitability_score), leverage (debt_to_equity), liquidity (log_dollar_volume), trailing return, value (price_to_book), sector and country fixed effects; HC3 |

What we already know, stated before the window opens.  Window 1 (full
period, pre-snapshot trailing vol as the control): IC = -0.094, H2 partial
rho = -0.163, H4 coefficient = -0.106.  But when the control is *recent*
realised volatility, the ESG association weakens: splitting window 1 at its
midpoint, the partial rho with second-half volatility is -0.066 (95% CI
[-0.19, 0.05]) given first-half volatility; a dry run of this exact script on
(2026-07-01, 2026-09-23] gives H2 = -0.02 and H4 = -0.04 (neither
significant; `dry_run_results.csv`).  Because window 2 controls for the six
months of realised volatility up to the anchor, we regard H2 and H4 as
genuinely uncertain, and we expect H1 and H3 not to be supported.  A null on
H2/H4 would indicate that the ESG risk signal is largely stale volatility
information.

## 5. Sample rules

* Universe: all {m['n_firms']} frozen firms with a valid anchor price (within 5 days).
* Firms that stop trading during the window: excluded from the primary test;
  survivorship bounds reported by imputing the 5th / 95th percentile return.
* No other exclusions, winsorisation or re-weighting.

## 6. Inference

Bootstrap: {m['n_boot']} firm resamples, seed {m['seed']}.  Holm adjustment across H1-H4.
All secondary analyses (pillars, horizons, sector-neutral returns, weight
draws) will be labelled exploratory.

## 7. Integrity

The files below are hashed.  `evaluate` refuses to run if any hash differs
or if the window has not closed.

| File | SHA-256 |
|---|---|
{hashes}
"""


def verify(strict=True):
    manifest = json.loads((PREREG / "manifest.json").read_text(encoding="utf-8"))
    bad = [p for p, h in manifest["sha256"].items() if sha256(PROJECT_ROOT / p) != h]
    if bad:
        msg = f"Hash mismatch (files changed since registration): {bad}"
        if strict:
            sys.exit(msg)
        print("  WARNING:", msg)
    else:
        print("  All registered hashes match.")
    return manifest


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def trailing_stats(prices, idx_prices, country, start, anchor):
    rows = []
    idx_ret = {c: ip.pct_change().dropna() for c, ip in idx_prices.items()}
    for t, g in prices.groupby("ticker"):
        s = g.set_index("date")["adj_close"].sort_index()
        w = s[(s.index > start) & (s.index <= anchor)]
        d = w.pct_change().dropna()
        if len(d) < 60:
            continue
        m = idx_ret.get(country.get(t, "US"))
        beta = np.nan
        if m is not None:
            j = pd.concat([d, m], axis=1, join="inner").dropna()
            if len(j) >= 60 and j.iloc[:, 1].var() > 0:
                beta = j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()
        rows.append({"ticker": t, "trail_vol": d.std(ddof=1) * np.sqrt(252) * 100,
                     "trail_beta": beta, "trail_ret": (w.iloc[-1] / w.iloc[0] - 1) * 100})
    return pd.DataFrame(rows)


def h4_regression(d):
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    d = d.dropna(subset=["realized_vol_oos", "trail_vol"]).copy()
    pct = lambda s: s.fillna(s.median()).rank() / len(s)
    X = pd.DataFrame({"esg": pct(d["ESG_composite"]), "trail_vol": pct(d["trail_vol"]),
                      "trail_beta": pct(d["trail_beta"]), "size": pct(d["market_cap"]),
                      "quality": pct(d["profitability_score"]), "leverage": pct(d["debt_to_equity"]),
                      "liquidity": pct(d["log_dollar_volume"]), "trail_ret": pct(d["trail_ret"]),
                      "value": pct(d["price_to_book"])}, index=d.index)
    fe = pd.get_dummies(d[["sector", "country"]], drop_first=True, dtype=float)
    fit = OLS(pct(d["realized_vol_oos"]), add_constant(pd.concat([X, fe], axis=1))).fit(cov_type="HC3")
    ci = fit.conf_int().loc["esg"]
    from scipy import stats
    return fit.params["esg"], ci[0], ci[1], float(stats.norm.cdf(fit.params["esg"] / fit.bse["esg"])), int(fit.nobs)


def evaluate(dry_run=False):
    anchor, end = (DRY_ANCHOR, DRY_END) if dry_run else (ANCHOR, END)
    manifest = verify(strict=not dry_run)
    today = pd.Timestamp(date.today())
    if not dry_run and today < end + pd.Timedelta(days=3):
        sys.exit(f"Window not closed: evaluation allowed from {(end + pd.Timedelta(days=3)).date()}.")
    frozen = pd.read_csv(PREREG / "frozen_scores.csv")
    country = frozen.set_index("ticker")["country"]
    tickers = sorted(set(frozen["ticker"]) | set(oos.COUNTRY_INDEX.values()))
    start = anchor - pd.DateOffset(months=TRAILING_MONTHS)

    if dry_run:
        prices = pd.read_csv(oos.PRICE_CACHE, parse_dates=["date"])
        prices = prices[prices["date"] <= end]
    else:
        cache = PREREG / "window2_prices.csv"
        if cache.exists():
            prices = pd.read_csv(cache, parse_dates=["date"])
        else:
            prices, _ = oos.download_prices(tickers, start - pd.Timedelta(days=10), end)
            prices.to_csv(cache, index=False, encoding="utf-8")

    outcomes, idx_prices = oos.compute_outcomes(prices, anchor, end, [])
    outcomes = oos.add_realized_beta(outcomes, frozen, idx_prices)
    trail = trailing_stats(prices, idx_prices, country, start, anchor)
    d = frozen.merge(outcomes, on="ticker", how="left").merge(trail, on="ticker", how="left")
    d = d[d["anchor_date"].notna()].copy()
    d["ret_cn"] = oos.neutralize(d, "ret_full", "country")

    rng = np.random.default_rng(RANDOM_SEED)
    n_boot = 500 if dry_run else N_BOOT
    r1, p1_two, lo1, hi1, n1 = oos.spearman_with_ci(d["pref_balanced"], d["ret_cn"], rng, n_boot)
    r2, p2_two, n2 = oos.partial_spearman(d["ESG_composite"], d["realized_vol_oos"], d["trail_vol"])
    lo2, hi2 = oos.partial_spearman_ci(d["ESG_composite"], d["realized_vol_oos"], d["trail_vol"], rng, n_boot)
    e3, lo3, hi3, p3, _, n3 = oos.portfolio_excess(d["pref_balanced"], d["ret_cn"], TOP_N, rng, n_boot)
    b4, lo4, hi4, p4, n4 = h4_regression(d)
    res = pd.DataFrame([
        {"hypothesis": "H1", "estimate": r1, "ci_low": lo1, "ci_high": hi1,
         "p_one_sided": p1_two / 2 if r1 > 0 else 1 - p1_two / 2, "n": n1},
        {"hypothesis": "H2", "estimate": r2, "ci_low": lo2, "ci_high": hi2,
         "p_one_sided": p2_two / 2 if r2 < 0 else 1 - p2_two / 2, "n": n2},
        {"hypothesis": "H3", "estimate": e3, "ci_low": lo3, "ci_high": hi3, "p_one_sided": p3, "n": n3},
        {"hypothesis": "H4", "estimate": b4, "ci_low": lo4, "ci_high": hi4, "p_one_sided": p4, "n": n4},
    ])
    res["p_holm"] = oos.holm(res["p_one_sided"])
    res["supported"] = res["p_holm"] < ALPHA
    res.insert(0, "anchor", str(anchor.date()))
    res.insert(1, "end", str(end.date()))
    res["mode"] = "dry_run_window1_subperiod" if dry_run else "confirmatory"
    out = PREREG / ("dry_run_results.csv" if dry_run else "confirmatory_results.csv")
    res.to_csv(out, index=False, encoding="utf-8")
    print(res.to_string(index=False))
    print(f"  Saved {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "freeze":
        freeze()
    elif cmd == "verify":
        verify()
    elif cmd == "evaluate":
        evaluate(dry_run="--dry-run" in sys.argv)
    else:
        print(__doc__)
