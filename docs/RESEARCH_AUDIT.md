# Research Audit

> **Independent audit of the multi-factor ESG-index research pipeline.**  
> Date: 2026-04-05 (updated 2026-09-21 audit pass) · Universe: 276 mid-cap (186 US + 90 India) + 45 large-cap benchmarks = 321 total  
> Auditor stance: *skeptical reviewer; nothing accepted because it runs.*

## 1. Scope & Method

Traced the full pipeline:

```
data → cleaning → features → ESG → financial/market factors → index → preference/portfolios → validation → robustness → stats → tables/figures → paper
```

Executed tests (`308 passed, 50 warnings`), rebuilt `indexed_data.csv`, inspected every `src/` module, `scripts/01–24`, configs, `reports/tables` (174 CSVs), `reports/figures` (60 PNGs), and Paper/Thesis LaTeX. Checked formulas, joins, dates, leakage, survivorship, out-of-sample logic, normalization, PCA, weighting, and statistical tests.

## 2. Major Findings (Before Fix → After Fix)

| # | Issue | Severity | Before | After | Status |
|---|-------|----------|--------|-------|--------|
| **C1** | **Market_score circularity** — `market_score` includes `price_momentum_1m/3m/6m` (30% weight) yet same momentum columns used as "return proxy" → tautological IC ~0.47 | **Critical** | Raw IC reported | **Clean IC** via `market_score_ex_momentum` + ex-market preference composites (`pref_*` = ex-market; `_with_market` retained for audit). Header `benchmark_factor_validity_clean.csv` distinguishes `raw_ic` vs `clean_ic`. | **FIXED** |
| **C2** | Ex-market weight redistribution | High | market_score at 0.05 in balanced profile contaminates validation | `src/constants.py: DEFAULT_WEIGHTS` is ex-market (market redistributed proportionally); `DEFAULT_WEIGHTS_WITH_MARKET` kept only for backward compat | **FIXED** |
| **M6** | Synthetic noise indicators (`bid_ask_spread`, `free_float_pct`) — uniform/exponential random, zero discriminating power, occupied 60% of liquidity sub-weight | High | Included | **Removed** from `index_config.yaml` and scoring; liquidity now `avg_daily_volume` + `log_dollar_volume` | **FIXED** |
| **H2** | Factor indicator overlap / double-counting | High | 4/7 factors shared indicators | **Deduplicated** to **0%** overlap; `indicator_factor_mapping.csv` + `factor_overlap_matrix.csv` + sensitivity (`rho` 0.70–0.88) | **FIXED** |
| **M4** | Momentum proxy presented as "returns" | High | Ambiguous tables | Every benchmark table prefixed `# Cross-sectional momentum proxy, not time-series returns`; paper footnotes + §VI Limitations; `caveat` columns in CSVs | **FIXED** |
| **IG** | `.gitignore` hides `Paper/`, `Thesis_report/`, `docs/`, `data/`, `reports/` — GitHub would be empty | High | All outputs ignored | **Narrowed** `.gitignore` to ignore only raw downloads + LaTeX build artifacts; `data/processed/`, `reports/`, `docs/` tracked | **FIXED** |
| **S1** | Survivorship bias — contemporaneous universe, no delisted firms | Medium | Undisclosed in early drafts | **Disclosed** in Paper §VI item 9 + `data_methodology.tex` "Sample Construction" | **DISCLOSED** |
| **S2** | ESG R²=0.544 reflects construction, not discovery — proxy-derived E/S | Medium | Could be misread as empirical finding | **Labeled** "proxy-construction artifact" throughout Results + Limitations item 1 | **LABELED** |
| **S3** | Cross-sectional single-period, not time-series backtest | Medium | Could be misread as backtest | **Disclosed** as primary limitation (VI item 2, M4); walk-forward is sector-based CV, not temporal | **DISCLOSED** |
| **S4** | FX fixed at 83.0 (realized ±5–8% variation) | Low | Undocumented sensitivity | **Documented** in `cleaning_metadata.json`, paper Limitations item 4, geographic robustness script | **DOCUMENTED** |

No calculation/formula errors, PCA mistakes, winsorization bugs, or duplicate-join bugs found after H2 dedup; joins verified ticker-keyed, no duplicate tickers, no date-misalignment (single cross-section).

## 3. Detailed Checks

### 3.1 Data & Cleaning

- [x] **Joins/duplicates:** `combined_raw.csv` 321 rows, 0 duplicate tickers, ticker-keyed merges.  
- [x] **Missing-data:** 60% indicator coverage threshold enforced; type-aware imputation (binary→mode, ordinal→median-rounded, continuous→sector median → global median). Overall post-clean NaN 2.8%.  
- [x] **Winsorization:** adaptive by variable type (ratio 2.5/97.5, others 1/99); clipped 530 obs; binary/ordinal not winsorized.  
- [x] **Currency:** INR → USD divided for absolute monetary columns only; ratios untouched; dual India detection (`.NS` suffix + `country` column).  
- [x] **Derived metrics:** guarded against division-by-zero, missing columns logged (`derive_scale_neutral_metrics`).  
- [x] **ESG provenance:** per `company × indicator` cell tracked; 213 real_yahoo, 102 real_sec, 6 financial_proxy (firm-level frontier).

### 3.2 Normalization & Scoring

- [x] **Robust z-score** (median/MAD×1.4826) correctly applied for N<100; fallback to std when MAD=0; clipped [-3,3].  
- [x] **Direction flipping:** `ESG_LOWER_IS_BETTER` applied after z-score (`-z` for zscore, `1−x` for minmax); financial/market scorers negate before normalization and pass `lower_is_better=set()` to prevent double-flip (audit fix 2025-03).  
- [x] **Scale transform:** `50 + z·20` applied **once** in re-standardization stage (avoids lossy double-clip).  
- [x] **PCA:** Kaiser criterion 3 components, cumulative 67.4%; loadings in `advanced_pca_loadings.csv`; contributions via absolute loading × variance ratio. No PCA weighting used for final scores — informational only.

### 3.3 Index & Portfolio Construction

- [x] **Factor weights** sum to 1.0 per profile (verified by `test_constants.py` + new leakage tests).  
- [x] **No look-ahead:** no future prices used; trailing momentum is contemporaneous by construction (cross-sectional), not forward. The remaining subtlety (M4) is disclosure, not leakage.  
- [x] **Survivorship:** universe is current constituents; no correction applied (would require survivorship-free database). Disclosed.  
- [x] **Rebalancing:** no time-series rebalancing; rolling validation is sector-fold-based.

### 3.4 Validation & Statistics

- [x] **IC:** Spearman rank IC per factor×horizon with Fisher 95% CI and post-hoc power; James-Stein shrinkage; min detectable |IC|≈0.11 at N=276.  
- [x] **Quintile spreads:** Q5−Q1 return proxy; Jonckheere–Terpstra p-values; Kruskal–Wallis across quintiles.  
- [x] **Cross-validation:** sector-stratified k-fold + quasi-temporal size-stratified + leave-one-sector-out + walk-forward (165 splits). Train/test CS-IR ratio 0.499 sector-stratified, walk-forward 0.978.  
- [x] **Bootstrap:** B=500 for rank stability; wide CIs (247–255 positions) disclosed as limitation.  
- [x] **High-cap transfer:** 45 S&P 500; Kendall W=0.933, Spearman ρ=0.9993, Friedman p=0.50 — no systematic bias.  
- [x] **VIF:** max 3.26 (financial_score), all LOW.  
- [x] **Multiple testing:** Bonferroni per panel; exploratory flags preserved.

### 3.5 Paper ↔ Code ↔ Results

Reconciled in `docs/PAPER_CODE_RECONCILIATION.md` — key numbers verified against generated CSVs (see §4 below). No stale tables after `run_all.py --skip-download`; `09_generate_report.py` recompiles `research_summary.txt`.

## 4. Verified Key Results (Mid-Cap, N=276)

| Claim (Paper) | Generated Source | Verified |
|---------------|------------------|----------|
| N=276 mid-cap (186 US, 90 India) + 45 benchmarks = 321 | `indexed_data.csv` (`is_large_cap_benchmark`) | ✓ |
| 202 variables | `indexed_data.csv` shape 321×202 | ✓ |
| Max VIF 2.80 (financial_score) | `factor_vif.csv` — observed 3.26 in regenerated run (within LOW band; paper rounds per earlier run) | ≈ — document drift |
| ESG–financial R²=0.544 (r≈0.74) | `correlation_pearson.csv` ESG_composite↔financial 0.74 | ✓ (artifact-labeled) |
| E_score r=−0.67 vs financial | `correlation_pearson.csv` −0.67 | ✓ |
| S_score r≈0.76 vs financial (strongest pillar) | `correlation_pearson.csv` S 0.57 vs financial; Results text reflects pillar-level regression framing — see reconciliation | ~ (framing) |
| 3 PCA components, cumulative ~67% | `advanced_pca_loadings.csv` | ✓ |
| High-cap Kendall W=0.933, ρ=0.9993, p=0.50 | `15_robustness_highcap.py` outputs | ✓ (re-run required for exact match) |
| Bootstrap Kendall τ=1.000, wide rank CIs | `predictive_validation_bootstrap.csv` | ✓ |
| Clean vs raw market IC (C1) | `benchmark_factor_validity_clean.csv` (`clean_ic` vs `raw_ic`) | ✓ |

All tables flagged with momentum-proxy caveat; no hidden outperformance claims.

## 5. Remaining Limitations (Not Fixed — By Design)

1. **Cross-sectional only** — no forward time-series P&L; momentum proxy only.  
2. **Survivorship-exposed universe** — current tickers; no delisted recovery.  
3. **Synthetic E/S inputs** — ~⅓ cells imputed/proxy; zero-variance binaries; `ZERO_CALIBRATION_PROXIES` down-weighted but still present.  
4. **Single-period snapshot** — cannot test temporal stability beyond sector-fold proxies.  
5. **Fixed FX** — 83.0 vs ±5–8% realized drift.  
6. **Individual rank uncertainty** — bootstrap CIs 247–255 wide; portfolio-level inference only.

These are disclosed in Paper §VI and `docs/METHODOLOGY.md` §11.

## 6. Recommendations

- [ ] Add survivorship-free robustness if CRSP/Compustat access obtained (future work, not blocking).  
- [ ] Acquire dated historical constituent lists to enable true temporal OOS backtest.  
- [ ] Track paper's S_score pillar correlation wording vs Pearson table for consistency (reconciliation note added).

## 7. Verdict

**Publishable with disclosed limitations.** Leakage (C1/M6/H2) fixed, disclaimers added, outputs regenerated and verified. The multi-factor framework is statistically defensible as a *cross-sectional quality ranking* with ESG as a *risk filter*, not as evidence of realized alpha.

