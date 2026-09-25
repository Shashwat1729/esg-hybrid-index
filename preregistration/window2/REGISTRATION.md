# Pre-registration: out-of-sample test of a public-data ESG mid-cap index (window 2)

Frozen on 2026-09-25, before the outcome window opens.  Git HEAD at freeze:
`97a2b43239936fde5dd592987f6fc2e929d46b18`.

## 1. Background

Paper 1 ("What Does Public-Data ESG Measure?") built a multi-factor index for
269 US and Indian mid-caps from data observable at the 2026-04-02 close and
tested it on returns over (2026-04-02, 2026-09-23].  Result: no return
predictability (H1, H3 not supported); higher ESG predicted lower realised
volatility given trailing volatility (H2 supported, Holm p = 0.013).  The
window-1 test was declared in code before prices were downloaded but was not
registered with a third party.  This registration makes the next window a
confirmatory test.

## 2. Data frozen now

* `frozen_scores.csv`: the 269 mid-cap firms, their frozen composite
  (`pref_balanced`), ESG composite and pillar scores, and the snapshot-date
  fundamentals used as controls.  No score will be recomputed.
* Prices: Yahoo Finance daily adjusted closes, downloaded after 2027-03-31 by
  `python scripts/28_preregistration.py evaluate`.

## 3. Window and outcomes

* Anchor: close of **2026-09-28**.  End: close of **2027-03-31**.
* Return: local-currency total return from the last close on or before the
  anchor to the last close on or before the end; made country-neutral by
  demeaning within country (US, India).
* Realised volatility: annualised s.d. of daily returns in the window
  (at least 20 observations).
* Trailing controls, measured over the 6 months up to the anchor:
  realised volatility, beta against the country index (S&P 500, NIFTY 50),
  and total return.

## 4. Hypotheses (one-sided, alpha = 0.05, Holm across H1-H4)

| | Hypothesis | Test |
|---|---|---|
| H1 | The composite ranks returns | Spearman IC(pref_balanced, country-neutral return) > 0; firm bootstrap CI |
| H2 | ESG predicts lower risk | Partial Spearman(ESG_composite, realised vol given trailing vol) < 0 |
| H3 | The top-19 portfolio outperforms | Equal-weight top-19 minus equal-weight universe, country-neutral > 0 |
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

* Universe: all 269 frozen firms with a valid anchor price (within 5 days).
* Firms that stop trading during the window: excluded from the primary test;
  survivorship bounds reported by imputing the 5th / 95th percentile return.
* No other exclusions, winsorisation or re-weighting.

## 6. Inference

Bootstrap: 5000 firm resamples, seed 42.  Holm adjustment across H1-H4.
All secondary analyses (pillars, horizons, sector-neutral returns, weight
draws) will be labelled exploratory.

## 7. Integrity

The files below are hashed.  `evaluate` refuses to run if any hash differs
or if the window has not closed.

| File | SHA-256 |
|---|---|
| `preregistration/window2/frozen_scores.csv` | `c00ebc22269d22c2917ac220f689389ac9ebd858d46428bdd99904f3904bba07` |
| `scripts/25_out_of_sample_evaluation.py` | `7455d173c2764938f1aff61d23529151048a60cf1ebe6ff55077e8c6462786fa` |
| `scripts/28_preregistration.py` | `d8ed6236bbbe0a0257d774c743c3dd97fdfe2418316cf102b8c98b4e46b12f1b` |
| `src/constants.py` | `8339bee13a172dd1acca8c84c6c8bea0f20fef3f72e8bb38769b94837d51f787` |
| `src/utils.py` | `82d0037bf83302fdb0baba8766ce37f268ec0ddc08130dad08a71e8883d46959` |
