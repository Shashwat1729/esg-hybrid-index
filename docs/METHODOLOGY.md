# Methodology

> **Canonical reference for the 10-factor (9 deployed) ESG-integrated index.**  
> For implementation details see `config/index_config.yaml`, `src/`, and `scripts/03_build_index.py`.

## 1. Research Question

Can a transparent, multi-factor ESG-integrated index built from public data outperform single-factor approaches for mid-cap equity evaluation (US + India), and does ESG act as a return predictor, a risk filter, or both?

## 2. Universe

| Segment | N | Source |
|---------|---|--------|
| US mid-cap | 181 | S&P MidCap 400 / Russell Midcap constituents (Yahoo Finance) |
| India mid-cap | 88 | NIFTY Midcap 150 constituents (`.NS` suffix, NSE) |
| **Mid-cap analytical universe** | **269** | Traded on the 2026-04-02 snapshot date (7 candidates removed) |
| Large-cap benchmarks | 45 | S&P 500 names (AAPL, MSFT, GOOGL, …) — robustness only, excluded by `load_indexed_data()` default |
| **Total raw universe** | **321** | 314 scored after the investability filter |

Period: single cross-sectional snapshot at the close of 2026-04-02 (`universe.snapshot_date`). Genuine out-of-sample evaluation over (2026-04-02, 2026-09-23] in `scripts/25_out_of_sample_evaluation.py`; in-sample "returns" in older tables are trailing momentum and are not predictive evidence (see `docs/RESEARCH_AUDIT.md`).

## 3. Data Sources

| Family | Fields | Coverage |
|--------|--------|----------|
| **Yahoo Finance** (`yfinance`) | Financial statements, market history, governance risk (`auditRisk`, `boardRisk`, `compensationRisk`, `shareholderRightsRisk`, `overallRisk`) | 321 tickers queried; ISS governance risk populated for 209/269 mid-caps |
| **SEC EDGAR XBRL** | R&D expenditure, share-based comp, repurchases, dividends, segments, employee count, goodwill, debt, tax rate | SEC-sourced cells: 3% of G-pillar cells (mid-caps) |
| **Hybrid ESG pipeline** | 34 declared ESG indicators (11 E, 11 S, 12 G); 21 carry information — see §4 and Paper Table I | 6-tier hierarchy per `company × indicator` cell |
| **Benchmark indices** | Daily OHLCV: S&P 500, S&P MidCap 400, Russell 2000, NIFTY 50 | 739–755 trading days |

Indian monetary fields converted to USD at **INR/USD = 92.97**, the Yahoo `INR=X` close on the 2026-04-02 snapshot date (`config/index_config.yaml`; §2 of `02_clean_data.py`). Stored in `data/processed/cleaning_metadata.json`.

## 4. Hybrid ESG Construction (6-Tier Provenance)

Applied **per `company × indicator` cell** (`esg_data_provenance.csv`):

1. `real_yahoo` — Yahoo governance risk scores  
2. `real_sec` — SEC XBRL fields (R&D, `revenue_per_employee`, governance)  
3. `financial_proxy` — calibrated proxies (12 documented; e.g. R&D intensity → environmental innovation, per Porter & van der Linde 1995)  
4. `sector_imputed` — sector median  
5. `global_imputed` — global median fallback  
6. `missing` — residual NaN  

**Transparency constraint:** E and S are predominantly proxy-derived; G has substantial directly observed data. Three binary ESG indicators (`carbon_reduction_target`, `human_rights_policy`, `anti_corruption_policy`) have zero cross-sectional variance post-imputation (all = 1.0); excluded automatically by zero-variance filter (§6).

Zero-calibration proxies (no `n_real` for validation) are down-weighted 0.3× in pillar scoring (`src/constants.py: ZERO_CALIBRATION_PROXIES`).

Provenance audit: `reports/tables/esg_data_provenance.csv`; held-out proxy validation in `17_proxy_validation.py`.

## 5. Variable Treatment (02_clean_data.py)

Seven variable types drive type-aware cleaning:

| Type | Examples | Treatment |
|------|----------|-----------|
| `binary` | `carbon_reduction_target`, `human_rights_policy` | Keep 0/1; mode-imputed; proportion-encoded downstream |
| `ordinal` | `board_size`, `esg_risk_rating` | Percentile-rank → [0,100] |
| `bounded_pct` | `renewable_energy_pct`, `board_independence_pct` | Clip [0,100] |
| `ratio` | `trailing_pe`, `debt_to_equity`, `ceo_pay_ratio` | Winsorize 2.5/97.5 pctile; robust MAD-z |
| `count` | `market_cap`, `total_revenue`, `employees`, `scope1_emissions` | Winsorize 1/99; log1p if skew > 2 |
| `rate` | `roa`, `roe`, `revenue_growth`, `r_d_intensity` | Winsorize 1/99 |
| `continuous` | `injury_rate`, `beta`, `price_volatility` | Winsorize 1/99 |

Outlier detection: consensus across **IQR (k=1.5/3.0) + MAD (3.5) + z-score (3σ)** plus **Mahalanobis** for multivariate (§2, `comprehensive_outlier_report`). Winsorization clips (does not delete) to preserve sector balance.

Derived scale-neutral metrics: `asset_turnover`, `cash_conversion`, `ebitda_margin`, `cash_to_assets`, `log_dollar_volume`, `sustainable_growth_rate`.

## 6. Scoring Pipeline (03_build_index.py)

### 6.1 Normalization — 3-Stage Design

**Stage 1 — Indicator-level** (`normalize_indicators`, `composite_index.py`):

- Method per `index_config.yaml`: `robust_zscore` (default).  
- `_z_score_sub` auto-selects by N: `percentile_rank` (N<20) → `robust_zscore` median/MAD×1.4826 (N<100) → `zscore` mean/std (N≥100).  
- Non-linear pre-transforms: `log1p` for ratios, `sqrt` for growth rates, `sigmoid` for profitability.  
- `lower_is_better` indicators (e.g. `scope1_emissions`, `injury_rate`, `esg_risk_rating`) flipped: `-z` for zscore, `1−x` for minmax.  
- Winsorization clipped at mapped scale (negated first for inverted indicators); then z-clipped to `[-3, 3]`.

**Stage 2 — Factor aggregation** (9 deployed factors, config-driven):

| # | Factor | Construction | Key config section |
|---|--------|-------------|-------------------|
| 1 | **ESG Composite** | 12 categories → 3 pillars (E/S/G) → SASB sector-materiality weighted composite (`config/sasb_materiality.yaml`) | `esg_index` |
| 2 | **Financial Score** | 5 categories (profitability 0.30, scale 0.20, efficiency 0.20, stability 0.15, valuation 0.15) | `financial_scoring` |
| 3 | **Market Score** | 3 categories (liquidity 0.40, volatility 0.30, momentum 0.30) — uses `price_momentum_1m/3m/6m` | `market_factors` |
| 4 | **Operational Score** | productivity / profitability / scale | `operational_quality` |
| 5 | **Risk-Adjusted Score** | `sharpe_ratio_1y`, `sortino_ratio_1y`, `max_drawdown_1y` + risk metrics | `risk_adjusted_scoring` |
| 6 | **Value Score** | Inverted multiples (`trailing_pe`, `price_to_book`, …) | `value_scoring` |
| 7 | **Growth Score** | `revenue_growth`, `earnings_growth`, `sustainable_growth_rate` | `growth_scoring` |
| 8 | **Stability Score** | liquidity / leverage / volatility | `stability_scoring` |
| 9 | **Sector Position** | Within-sector percentile of raw quality indicators | — |
| — | ~~Similarity Rank~~ | **Removed from deployed composite** (r=0.70 with ESG_composite; near-collinear) | — |

Indicator overlap after dedup: **0%** shared indicators across factors (`indicator_factor_mapping.csv`, `factor_overlap_matrix.csv`). Two-level hierarchical weighting: `category_weight × indicator_weight`, renormalized when indicators missing.

**Stage 3 — Single re-standardization** (Step 7b):

All factor scores → `50 + z·10`, clipped [0,100] exactly once (avoids lossy double-clip). Re-standardization is the only transform to the 0–100 scale.

### 6.2 Post-Processing

- **Sector blend:** 70% cross-sectional + 30% within-sector percentile (prevents sector-neutralization while reducing sector-tilt bias).  
- **Monotonicity correction:** adaptive isotonic regression on decile bins (60–90% weight by pre-correction monotonicity).  
- **E_score residualization:** `E_score ⟂ financial_score` via OLS residual (`E = α + β·financial_score + ε`; use `ε`) to reduce construction circularity.  
- **G harmonization:** source-wise **quantile normalization** across Yahoo vs SEC channels before pillar merge.

## 7. Investor Profiles (Preference Scoring)

Composite per profile `p`:

```
C_i^{(p)} = Σ_k w_k^{(p)} · F_ik ,  Σ_k w_k^{(p)} = 1
```

| Factor | ESG-First | Balanced | Financial-First |
|--------|-----------|----------|-----------------|
| ESG Composite | 0.55 | **0.22** | 0.05 |
| Financial | 0.08 | **0.18** | 0.25 |
| Market | 0.02 | **0.05** | 0.10 |
| Operational | 0.05 | **0.06** | 0.08 |
| Risk-Adjusted | 0.08 | **0.15** | 0.18 |
| Growth | 0.06 | **0.14** | 0.15 |
| Value | 0.03 | **0.04** | 0.07 |
| Stability | 0.06 | **0.08** | 0.07 |
| Sector Position | 0.07 | **0.08** | 0.05 |

(Weights from `config/index_config.yaml: preference_scoring.investor_profiles`; `similarity_rank = 0` excluded.)

Aggregation mode: **`rank`** — percentile-rank transform before weighting (equalizes variance across factors). Verified in `tests/`.

**Circularity-controlled deployment:** `market_score` excluded from primary weights (`DEFAULT_WEIGHTS` in `src/constants.py` = ex-market; `_with_market` retained only for audit). All headline validation uses **ex-market** composites (`pref_*_ex_market`).

## 8. SASB Materiality

Sector-specific E/S/G pillar weights (`config/sasb_materiality.yaml`):

| Sector | E | S | G |
|--------|---|---|---|
| Energy | 0.55 | 0.20 | 0.25 |
| Utilities | 0.50 | 0.20 | 0.30 |
| Basic Materials | 0.50 | 0.25 | 0.25 |
| Industrials | 0.40 | 0.30 | 0.30 |
| Tech / Comm. Services | 0.20 | 0.45 | 0.35 |
| Healthcare / Consumer Cyclical | 0.25 | 0.40 | 0.35 |

## 9. What "Returns" Mean — Critical Caveat (M4)

> **All performance metrics are cross-sectional momentum-proxy differentials, not time-series portfolio returns.**

- Return proxy: **trailing** `price_momentum_1m / 3m / 6m` (point-in-time).  
- Cross-sectional IR: `mean(momentum) / std(momentum)` across N selected stocks.  
- Benchmark "outperformance": selected mean momentum − universe mean.  
- No forward returns, no timestamped rebalancing, no P&L. Each table carries header `# Cross-sectional momentum proxy, not time-series returns`.

Paper §IV footnotes and §VI Limitations disclose this; benchmark tables flagged via `proxy_header=True` in code.

## 10. Statistical Validation (Scripts 04–24)

| Check | Script | Key output |
|-------|--------|------------|
| Descriptive + normality + correlations + VIF + ANOVA | `04_statistical_tests.py` | 25+ tables; max VIF 2.04 |
| Weight grid search (99 combos) + perturbation stability | `05_weight_sensitivity.py` | ≥0.99 Spearman at ±20% |
| Multi-horizon + alpha/beta + regime + weighting methods + **clean IC** | `06_benchmark_comparison.py` | `benchmark_factor_validity_clean.csv` |
| In-sample IC / quintiles / CV (descriptive only — trailing targets) | `07b_cross_sectional_validation.py` | `predictive_validation_*.csv` |
| PCA / clustering / rank uncertainty (universe resampling + Dirichlet weights, B=1000) | `08_advanced_analysis.py` | `advanced_*.csv`, `bootstrap_rank_uncertainty_summary.csv` |
| Large-cap reference comparison (45 names; not a generalisation test) | `15_robustness_highcap.py` | — |
| Turnover / capacity / walk-forward | `16_financial_validation.py` | Pass $25M–$500M |
| Proxy provenance + held-out | `17_proxy_validation.py` | Per-indicator `n_real` |
| Leave-one-sector-out | `18_sector_cv.py` | — |
| Noise injection + factor dropout | `19_synthetic_sensitivity.py` | — |
| Subsampling + rolling | `20_subsampling_stability.py` | — |
| Incremental ESG R² | `22_esg_incremental_value.py` | — |
| PCA weight audit | `23_pca_weight_validation.py` | — |
| US/India normalization bias | `24_geographic_robustness.py` | — |
| **Genuine out-of-sample test (H1–H3, Holm)** | `25_out_of_sample_evaluation.py` | `oos_*.csv` |
| Mechanism & robustness: factor controls (H4), provenance split, CV vol forecast, double sort, recent-vol control, weight draws, costs, monthly | `27_paper1_extensions.py` | `ext_*.csv`, `fig_ext_weights` |
| Pre-registered window 2 (freeze / verify / evaluate) | `28_preregistration.py` | `preregistration/window2/` |
| Paper numbers, tables, figures | `26_paper_artifacts.py` | `Paper/generated/` |

Multiple-testing: Bonferroni per panel (IC tests), Benjamini–Hochberg where noted; 95% CIs via Fisher z or bootstrap (B=500, seed 42).

## 11. Known Limitations

See `docs/RESEARCH_AUDIT.md` §5 and Paper §VII. In brief: one six-month OOS window (MDE ≈ 0.17); E and S pillars are financial proxies; ISS governance data dependence; candidate list not point-in-time; results gross of costs.

## 12. Reproducing

```bash
pip install -r requirements.txt
python scripts/03_build_index.py        # or full: python scripts/run_all.py --skip-download
python app.py                           # demo on :7860
pytest -q
```

Full instructions: `docs/REPRODUCIBILITY.md`.

## 13. References

- Fama & French (1993), Jegadeesh & Titman (1993), Amihud (2002), Efron & Tibshirani (1993), Gompers et al. (2003), Bebchuk et al. (2009), Khan et al. (2016), SASB Materiality Map (2023), Leys et al. (2013), Iglewicz & Hoaglin (1993), Aguinis et al. (2013).

