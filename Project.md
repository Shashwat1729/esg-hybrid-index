# Project: Multi-Factor ESG-Integrated Investment Index

## Purpose
Constructs a transparent, ten-factor ESG-integrated investment index for mid-cap companies across US and Indian markets. This is a B.E. thesis project (BITS F421T) by Shashwat Bajpai (BITS Pilani, Hyderabad Campus) that demonstrates how multi-factor scoring (combining ESG quality, financial strength, market behaviour, and operational metrics) outperforms single-factor approaches in portfolio construction.

## Tech Stack
- **Language:** Python 3.10+ (developed with 3.11/3.12)
- **Data Sources:** Yahoo Finance (`yfinance`), SEC EDGAR XBRL (`requests`), synthetic ESG proxy generation
- **Data Processing:** `pandas 2.2.2`, `numpy 1.26.4`, `openpyxl 3.1.5`
- **Statistics:** `scipy 1.13.1`, `statsmodels 0.14.2`, `scikit-learn 1.5.1`
- **Visualization:** `matplotlib 3.9.2`, `seaborn 0.13.2`, `plotly 5.24.0`, `matplotlib-venn 1.1.1`
- **Web Interface:** `gradio 4.44.0`
- **Config:** YAML (`pyyaml 6.0.1`)
- **Testing:** `pytest 8.3.2` (252 tests across 13 modules)
- **Documentation:** LaTeX (BITS Pilani template + IEEE conference paper)
- **VCS:** Git, GitHub (`https://github.com/Shashwat1729/multi-factor-esg.git`)

## Architecture

### Data Flow Pipeline
```
Raw Data Sources
  |
  v
01_download_data.py ──> data/raw/*.csv (12 files)
  |                     Yahoo Finance (321 tickers), SEC EDGAR, benchmark indices
  v
02_clean_data.py ──> data/processed/clean_data.csv
  |                  Type-aware imputation, currency conversion (INR->USD),
  |                  outlier detection (IQR+MAD+Z consensus), variable classification
  v
03_build_index.py ──> data/processed/indexed_data.csv
  |                   10 factor scores + 3 investor profile preference scores
  |                   + similarity matrix + PCA + company rankings
  v
04-24 Analysis Scripts ──> reports/tables/ (160+ CSV), reports/figures/ (52+ PNG)
  |                        Statistical tests, weight sensitivity, benchmarking,
  |                        robustness checks, cross-validation, regime analysis
  v
09_generate_report.py ──> reports/research_summary.txt
  |
  v
app.py ──> Gradio web interface (localhost:7860)
```

### Ten-Factor Scoring Model
```
                    ┌─────────────────────────────────────────────────┐
                    │           10-Factor Composite Index             │
                    │                                                 │
                    │  1. ESG Composite (E + S + G pillars, SASB)     │
                    │  2. Financial Score (profitability, scale, eff) │
                    │  3. Market Score (liquidity, vol, momentum)     │
                    │  4. Operational Score (productivity, innovation)│
                    │  5. Risk-Adjusted Score (Sharpe, Sortino, DD)   │
                    │  6. Value Score (P/E, P/B, EV multiples)        │
                    │  7. Growth Score (revenue, earnings growth)     │
                    │  8. Stability Score (leverage, liquidity ratios)│
                    │  9. Similarity Rank (cosine sim on ESG vectors) │
                    │ 10. Sector Position (within-sector percentile)  │
                    └─────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              ESG-First (40%)     Balanced (25%)     Financial-First (5%)
              Profile              Profile            Profile
              (ESG weight)         (ESG weight)       (ESG weight)
```

### Three Investor Profiles (weights sum to 1.0)
| Factor | ESG-First | Balanced | Financial-First |
|--------|-----------|----------|-----------------|
| ESG Composite | 0.40 | 0.25 | 0.05 |
| Financial | 0.10 | 0.20 | 0.30 |
| Market | 0.05 | 0.08 | 0.15 |
| Operational | 0.08 | 0.10 | 0.10 |
| Risk-Adjusted | 0.07 | 0.10 | 0.10 |
| Growth | 0.05 | 0.08 | 0.10 |
| Value | 0.05 | 0.07 | 0.10 |
| Stability | 0.07 | 0.05 | 0.05 |
| Similarity | 0.08 | 0.04 | 0.03 |
| Sector Position | 0.05 | 0.03 | 0.02 |

## Directory Structure
```
multi-factor-esg/
├── app.py                          # Gradio web interface (5 tabs, 641 lines)
├── requirements.txt                # Python 3.10+ dependencies (15 packages)
├── pytest.ini                      # Pytest configuration
├── README.md                       # Project overview (290 lines)
├── LICENSE                         # MIT License (2026)
├── .gitignore                      # Python, LaTeX, IDE ignores
│
├── config/
│   ├── index_config.yaml           # Master config (659 lines): weights, normalization, profiles
│   ├── data_sources.yaml           # API config: Yahoo, SEC, commercial providers
│   └── sasb_materiality.yaml       # SASB sector E/S/G weight splits (11 sectors)
│
├── src/                            # Core library (1,916 lines across 14 files)
│   ├── __init__.py
│   ├── constants.py                # Column defs, weight profiles, type metadata (285 lines)
│   ├── utils.py                    # robust_zscore, path helpers, data loaders (220 lines)
│   ├── logging_config.py           # Centralized logging setup (63 lines)
│   ├── data_collection/
│   │   ├── data_pipeline.py        # Yahoo Finance + SEC EDGAR ingestion (254 lines)
│   │   └── data_quality.py         # Coverage checks, imputation, QualityReport (103 lines)
│   ├── financial_scoring/
│   │   └── financial_scorer.py     # FinancialScorer + MarketFactorScorer (359 lines)
│   ├── index_construction/
│   │   └── composite_index.py      # CompositeIndexBuilder, normalize, pillar scores (472 lines)
│   └── similarity/
│       ├── cosine_similarity.py    # Pairwise similarity (cosine/euclidean/jaccard) (206 lines)
│       └── preference_scoring.py   # PreferenceScorer with 3 aggregation modes (224 lines)
│
├── scripts/                        # Analysis pipeline (26 scripts, ~15,000+ lines)
│   ├── run_all.py                  # Master orchestrator (133 lines)
│   ├── 01_download_data.py         # Download Yahoo Finance, SEC EDGAR, benchmarks (1,794 lines)
│   ├── 02_clean_data.py            # Type-aware cleaning + imputation
│   ├── 03_build_index.py           # Build 10-factor composite index (1,585 lines)
│   ├── 04_statistical_tests.py     # Normality, correlation, regression, ANOVA, VIF
│   ├── 05_weight_sensitivity.py    # Weight grid search, rank stability
│   ├── 06_benchmark_comparison.py  # Multi-horizon returns, alpha/beta decomposition
│   ├── 07_visualizations.py        # Generate 30+ research figures
│   ├── 07b_cross_sectional_validation.py  # IC analysis, quintile spreads, circularity
│   ├── 08_advanced_analysis.py     # PCA, clustering, bootstrap CI, factor ablation
│   ├── 09_generate_report.py       # Compile research summary (22 sections)
│   ├── 10_esg_benchmarking.py      # Cross-provider ESG validation
│   ├── 11_profile_justification.py # Statistical justification for 3 profiles
│   ├── 13_compute_summaries.py     # Numeric summaries for Results chapter
│   ├── 14_run_checks.py            # Pipeline health/sanity checks
│   ├── 15_robustness_highcap.py    # Large-cap generalization test
│   ├── 16_financial_validation.py  # Fama-MacBeth, turnover, transaction costs
│   ├── 17_proxy_validation.py      # ESG proxy calibration + held-out validation
│   ├── 18_sector_cv.py             # Leave-one-sector-out cross-validation
│   ├── 19_synthetic_sensitivity.py # Noise injection + factor dropout sensitivity
│   ├── 20_subsampling_stability.py # 500-iteration random subsampling stability
│   ├── 21_temporal_stability.py    # Temporal out-of-sample validation
│   ├── 22_esg_incremental_value.py # ESG incremental R-squared beyond sector dummies
│   ├── 23_pca_weight_validation.py # PCA vs configured weight comparison
│   ├── 24_geographic_robustness.py # US/India normalization bias test
│   └── sector_geography_analysis.py # Sector/geography deep dive
│
├── tests/                          # 252 tests across 13+ modules
│   ├── conftest.py                 # 8 shared fixtures (279 lines)
│   ├── test_build_index.py         # 38 tests for index construction
│   ├── test_clean_data.py          # 42 tests for data cleaning
│   ├── test_composite_index.py     # 22 tests for ESG composite
│   ├── test_constants.py           # 19 tests for constants/config alignment
│   ├── test_cosine_similarity.py   # 23 tests for similarity computation
│   ├── test_data_pipeline.py       # 10 tests for data pipeline
│   ├── test_data_quality.py        # 17 tests for data quality
│   ├── test_financial_scorer.py    # 10 tests for financial scoring
│   ├── test_integration.py         # 7 integration tests (full pipeline)
│   ├── test_preference_scoring.py  # 21 tests for preference scoring
│   ├── test_time_splits.py         # 6 tests for cross-sectional validation
│   ├── test_utils.py               # 8 tests for utilities
│   ├── debug_fragment.py           # Disabled debug stub
│   └── run_time_split_tests.py     # Subprocess test runner
│
├── data/
│   ├── raw/                        # 12 source CSVs
│   │   ├── yahoo_financials.csv    # 321 companies, 43 columns (REAL)
│   │   ├── yahoo_esg.csv           # 321 companies, 11 columns (REAL, mostly governance risk)
│   │   ├── synthetic_esg.csv       # 263 companies, 32 ESG indicators (SYNTHETIC)
│   │   ├── hybrid_esg.csv          # 321 companies, 32 columns (HYBRID: real + proxy)
│   │   ├── market_data.csv         # 314 companies, 17 market metrics (REAL)
│   │   ├── sec_governance.csv      # 77 US companies, employee data (REAL)
│   │   ├── sec_rd_data.csv         # 73 US companies, R&D + revenue (REAL)
│   │   ├── combined_raw.csv        # 321 companies, ~100 merged columns
│   │   ├── sp500_benchmark.csv     # S&P 500 daily prices (754 days, 2023-2026)
│   │   ├── sp400_benchmark.csv     # S&P MidCap 400 daily prices
│   │   ├── russell2000_benchmark.csv # Russell 2000 daily prices
│   │   └── nifty50_benchmark.csv   # NIFTY 50 daily prices (740 days)
│   ├── processed/                  # 5 cleaned/scored datasets
│   │   ├── clean_data.csv          # 321 rows, ~95 columns (cleaned + computed ratios)
│   │   ├── indexed_data.csv        # 321 rows, ~170 columns (all scores + _norm columns)
│   │   ├── cluster_assignments.csv # 276 companies, 3 clusters
│   │   ├── pca_scores.csv          # 276 companies, 4 PCA components
│   │   ├── similarity_matrix.csv   # 321x321 cosine similarity matrix
│   │   └── cleaning_metadata.json  # Exchange rate, Indian ticker list
│   └── benchmarking/
│       ├── company_esg_ratings.csv  # 27 companies, MSCI/Sustainalytics/S&P ratings (REAL)
│       ├── esg_benchmark_data.md    # 617-line ESG provider methodology reference
│       └── academic_references.bib  # Academic citations for benchmarking
│
├── reports/
│   ├── research_summary.txt         # Auto-generated 765-line, 22-section summary
│   ├── key_findings.csv             # Key findings table
│   ├── figures/                     # 52+ PNG figures (+ some PDF duplicates)
│   │   ├── fig01-fig30_*.png        # Core analysis figures
│   │   ├── benchmark_*.png          # 7 benchmark comparison figures
│   │   ├── profile_*.png            # 7 investor profile figures
│   │   ├── robustness_highcap_*.png # 2 robustness figures
│   │   ├── fig_ic_heatmap.*         # Information coefficient heatmap
│   │   ├── fig_quintile_returns.*   # Quintile return visualization
│   │   ├── fig_proxy_calibration.png # ESG proxy calibration
│   │   └── timesplit_sharpe_synthetic.* # Time-split Sharpe analysis
│   ├── tables/                      # 160+ CSV result tables
│   │   ├── descriptive_statistics.csv     # Full descriptive stats
│   │   ├── normality_tests.csv            # Shapiro-Wilk, Jarque-Bera, K-S
│   │   ├── correlation_pearson.csv        # Pearson correlation matrix
│   │   ├── correlation_spearman.csv       # Spearman correlation matrix
│   │   ├── multiple_regression.csv        # ESG -> financial regressions
│   │   ├── factor_vif.csv                 # Variance Inflation Factors (max 1.70)
│   │   ├── company_rankings.csv           # Full 276-company rankings (97.5 KB)
│   │   ├── top20_balanced.csv             # Top 20 by balanced profile
│   │   ├── weight_grid_search.csv         # 99 weight combinations tested
│   │   ├── benchmark_summary.csv          # Multi-factor vs alternatives
│   │   ├── benchmark_multi_horizon.csv    # Returns at 1m, 3m, 6m, 12m
│   │   ├── fama_macbeth_regression.csv    # Cross-sectional regressions
│   │   ├── esg_data_provenance.csv        # Per-indicator data source tracking
│   │   ├── advanced_pca_loadings.csv      # PCA factor loadings
│   │   ├── advanced_cluster_profiles.csv  # K-means cluster profiles
│   │   ├── advanced_bootstrap_ci.csv      # 500-iteration bootstrap CIs
│   │   ├── advanced_factor_ablation.csv   # Leave-one-factor-out analysis
│   │   ├── advanced_efficient_frontier.csv # Mean-variance frontier (132 KB)
│   │   └── ... (140+ more tables)
│   └── analysis/                    # 44 analytical summary files
│       ├── benchmark_comparison_report.md
│       ├── esg_financial_tradeoff.md
│       ├── improved_results_summary.md
│       ├── profile_portfolio_analysis.md
│       ├── robustness_validation_summary.md
│       ├── score_analysis_summary.md
│       ├── sector_geography_analysis.md
│       ├── esg_financial_tradeoff_data.json
│       └── ... (36+ CSV analysis files)
│
├── docs/
│   ├── LITERATURE_REVIEW.txt        # 406-line comprehensive lit review
│   └── data_sources.xlsx            # Data source documentation
│
├── Thesis_report/                   # Full LaTeX thesis (BITS Pilani template)
│   ├── main.tex                     # Master document
│   ├── Thesis.cls                   # Document class
│   ├── variables.tex                # Metadata
│   ├── Bibliography.bib             # References
│   ├── Makefile                     # LaTeX build
│   ├── Chapters/
│   │   ├── Introduction.tex
│   │   ├── Literature_Review.tex
│   │   ├── Data_and_Methods.tex
│   │   ├── Results.tex
│   │   └── FutureWork.tex
│   ├── Appendices/                  # A, B, C (Profile Justification)
│   ├── Sections/                    # Generated table inclusions
│   ├── Tables/                      # 18 CSV + TeX table files
│   ├── Figures/                     # 24 PNG + 2 PDF
│   └── Missing_Packages/           # LaTeX .sty fallbacks (9 files)
│
└── Paper/                           # IEEE conference paper (standalone)
    ├── Thesis.tex                   # Master document
    ├── Thesis.cls                   # IEEE document class
    ├── sections/                    # 10 section .tex files + references.bib
    └── Figures/                     # 28 PNG figures
```

## Key Files

### Source Library (`src/`)
| File | Lines | Purpose |
|------|-------|---------|
| `src/constants.py` | 285 | Column definitions (ESG_ENV/SOC/GOV_COLS, FINANCIAL_COLS, MARKET_COLS), weight profiles (DEFAULT_WEIGHTS, 9 factors ex-market), variable type metadata (BINARY_VARS, ORDINAL_VARS, ESG_LOWER_IS_BETTER), `load_profiles_from_config()` |
| `src/utils.py` | 220 | `robust_zscore()` (MAD-based), `get_project_root()`, `setup_paths()`, `load_indexed_data()` (filters out large-cap benchmarks), `load_profile_weights()` |
| `src/logging_config.py` | 63 | `setup_logging()` — centralized console + optional file logging |
| `src/data_collection/data_pipeline.py` | 254 | `PipelinePaths` dataclass, `fetch_yahoo_financials()` (JSON cache), `fetch_sec_company_facts()` (SEC EDGAR API), `standardize_and_clean()` (coverage filter + group-median imputation), `load_configs()` |
| `src/data_collection/data_quality.py` | 103 | `QualityReport` dataclass, `missingness_report()` (z>4 outlier counts), `enforce_min_coverage()` (60% threshold), `group_median_impute()` (sector-based) |
| `src/financial_scoring/financial_scorer.py` | 359 | `FinancialScorer` (5-category composite: profitability/scale/efficiency/stability/valuation), `MarketFactorScorer` (3-category: liquidity/volatility/momentum). Both invert lower-is-better, winsorize at 1/99%, scale to 50+z*20, clip [0,100] |
| `src/index_construction/composite_index.py` | 472 | `normalize_indicators()` (4 methods: zscore/minmax/percentile/robust_zscore, winsorization, sector grouping, direction flipping), `compute_pillar_scores()` (E/S/G + SASB materiality), `CompositeIndexBuilder` (full ESG index build) |
| `src/similarity/cosine_similarity.py` | 206 | `cosine_similarity()`, `compute_similarity_matrix()` (3 metrics: cosine/euclidean/jaccard, feature weighting), `rank_by_similarity()` (peer ranking) |
| `src/similarity/preference_scoring.py` | 224 | `PreferenceScorer` with 3 aggregation modes (rank/variance_equalized/raw), 10-factor weighted composite, 3 investor profiles, `rank_companies()` |

### Pipeline Scripts (`scripts/`)
| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| `run_all.py` | Master orchestrator — runs all 20 scripts sequentially with `--skip-download` option | Console status report |
| `01_download_data.py` | Downloads Yahoo Finance (321 tickers), SEC EDGAR (R&D, governance), 4 benchmark indices, generates hybrid ESG via 6-tier pipeline | `data/raw/*.csv` (12 files) |
| `02_clean_data.py` | Type-aware cleaning: 7 variable types (binary/ordinal/bounded_pct/ratio/count/rate/continuous), adaptive winsorization, multi-method outlier detection, sector-median imputation, INR->USD conversion (rate=83.0), log transforms for skewed counts | `data/processed/clean_data.csv` |
| `03_build_index.py` | Builds all 10 factor scores using `src/` library, computes 3 preference profiles, sector position, similarity rank, PCA weight rationale | `data/processed/indexed_data.csv`, `reports/tables/company_rankings.csv` |
| `04_statistical_tests.py` | Normality (Shapiro-Wilk, Jarque-Bera, K-S), Pearson/Spearman correlations, multiple regression, heteroscedasticity (Breusch-Pagan), VIF, ANOVA by sector/country, quintile analysis, power analysis, multiple testing correction | 25+ tables |
| `05_weight_sensitivity.py` | Grid search over 99 weight combinations, rank stability under ±20% perturbation, weight interdependence matrix | 5+ tables |
| `06_benchmark_comparison.py` | Multi-horizon returns (1m/3m/6m/12m), alpha/beta decomposition, regime analysis (bull/bear), vs ESG-only/financial-only/growth-only strategies | 10+ tables, benchmark figures |
| `07_visualizations.py` | Generates 30+ publication-quality figures (distributions, radar, heatmaps, scatter, boxplots, PCA biplots, dendrograms, CDFs, dashboards) | `reports/figures/fig01-fig30_*.png` |
| `07b_cross_sectional_validation.py` | Information Coefficient analysis, quintile return spreads, bootstrap rank stability, Kruskal-Wallis, circularity checks | IC heatmap, quintile returns, tables |
| `08_advanced_analysis.py` | PCA (Kaiser criterion), K-means clustering (silhouette), 500-iteration bootstrap CIs, factor ablation, efficient frontier, Gini inequality | 15+ tables, PCA/cluster outputs |
| `09_generate_report.py` | Compiles 22-section research summary from all table outputs | `reports/research_summary.txt` |
| `10_esg_benchmarking.py` | Cross-provider validation (MSCI, Sustainalytics, S&P Global correlations) | Benchmark correlation tables |
| `11_profile_justification.py` | Statistical justification for 3 investor profiles (differentiation, sensitivity, overlap) | Profile analysis tables/figures |
| `13_compute_summaries.py` | Numeric summaries for thesis Results chapter | Summary statistics |
| `14_run_checks.py` | Pipeline health checks: file existence, column counts, score ranges, weight sums | Pass/fail report |
| `15_robustness_highcap.py` | Large-cap generalization: re-runs methodology on 45 large-cap benchmarks | Robustness comparison tables |
| `16_financial_validation.py` | Fama-MacBeth cross-sectional regressions, portfolio turnover, capacity analysis | Financial validation tables |
| `17_proxy_validation.py` | ESG proxy calibration, per-indicator provenance audit, held-out validation | Provenance + proxy tables |
| `18_sector_cv.py` | Leave-one-sector-out cross-validation stability | Sector CV tables |
| `19_synthetic_sensitivity.py` | Noise injection (Gaussian perturbation), factor dropout, rank stability under noise | Sensitivity tables |
| `20_subsampling_stability.py` | 500-iteration random subsampling, rolling window stability | Stability tables |
| `21_temporal_stability.py` | Temporal out-of-sample validation (train/test time splits) | Time-split tables |
| `22_esg_incremental_value.py` | ESG incremental R-squared beyond sector dummies (demonstrating ESG adds information) | Incremental value tables |
| `23_pca_weight_validation.py` | Compares PCA-derived weights vs configured weights | PCA validation tables |
| `24_geographic_robustness.py` | US vs India normalization bias testing | Geographic robustness tables |

### Web Application
| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 641 | Gradio interface with 5 tabs: **Company Explorer** (single company deep-dive with radar chart + ESG rating), **Company Comparison** (side-by-side with weighted recommendation), **Investment Screener** (filter by ESG/financial/risk thresholds), **Portfolio Builder** (equal-weighted aggregate vs universe), **Index Methodology** (interactive weight adjustment with auto-normalization) |

## Company Universe

### Composition
- **Total raw universe:** 321 unique tickers
- **After quality filtering:** 276 companies (60% minimum indicator coverage)
- **US companies:** 186 (67.4%) — mid-caps from S&P MidCap 400 & Russell Midcap
- **Indian companies:** 90 (32.6%) — NSE-listed (.NS suffix) from NIFTY Midcap 150 ESG
- **Large-cap benchmarks:** 45 companies (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, etc.) — included for statistical power, filtered out for mid-cap-only analysis

### Sectors (11 GICS sectors)
| Sector | Count | Notable Tickers |
|--------|-------|-----------------|
| Healthcare | 40 | ILMN, ALGN, INCY, BMRN, DXCM |
| Consumer Cyclical | 38 | TRENT.NS, RELAXO.NS, RBLX |
| Industrials | 37 | GNRC, MTZ, EME, FIX |
| Technology | 35 | ZS, OKTA, CRWD, FTNT, PERSISTENT.NS |
| Financial Services | 28 | EWBC, MUTHOOTFIN.NS, CHOLAFIN.NS |
| Basic Materials | 22 | Various |
| Consumer Defensive | 21 | Various |
| Real Estate | 16 | Various |
| Energy | 13 | Various |
| Utilities | 11 | Various |
| Communication Services | — | Various |

### Top-Ranked Companies (Balanced Profile)
1. MUTHOOTFIN.NS (82.4), 2. EWBC (76.9), 3. INCY (73.4), ...

## ESG Data Sourcing

### 6-Tier ESG Data Pipeline
| Tier | Source | Type | Coverage |
|------|--------|------|----------|
| 1 | Yahoo Finance governance risk scores | **REAL** | auditRisk, boardRisk, compensationRisk, shareholderRightsRisk, overallRisk |
| 2 | SEC EDGAR XBRL filings | **REAL** | R&D expenditure, revenue, employee counts (US only, 73-77 companies) |
| 3 | Financial proxy derivation | **SYNTHETIC** (calibrated) | 12 proxies with documented economic rationale (e.g., R&D intensity -> environmental innovation) |
| 4 | Sector-median imputation | **IMPUTED** | Fill remaining gaps using sector-level medians |
| 5 | Cross-sector imputation | **IMPUTED** | Global median fallback |
| 6 | Missing (NaN) | **ABSENT** | Remaining unfilled indicators |

### Data Provenance Tracking
The pipeline tracks per-company, per-indicator provenance (`real_yahoo`, `financial_proxy`, `missing`) in `reports/tables/esg_data_provenance.csv`.

### Transparency Notes
- ESG pillar scores (E, S, G) from Yahoo Finance are **mostly empty** — only governance risk scores are reliably populated
- The `synthetic_esg.csv` contains algorithmically generated ESG indicators calibrated to realistic distributions
- The `hybrid_esg.csv` merges real Yahoo/SEC data with financial-proxy-derived indicators
- All ESG data limitations are explicitly documented in the thesis

## Key Results

### Portfolio Performance (12-month horizon)
| Strategy | Return | Excess vs Universe |
|----------|--------|-------------------|
| Multi-factor Top-20 (Balanced) | +23.39% | +16.07 pp |
| Full Universe | +7.32% | — |
| Bear-market excess | — | +10.04 pp |

### Key Statistical Findings
- **ESG-Financial correlation:** R² = 0.1058 (p < 0.0001) — modest positive relationship
- **S_score dominance:** S_score → financial_score R² = 0.4097 (strongest single pillar)
- **No multicollinearity:** Max VIF = 1.70 (financial_score)
- **Factor importance:** financial_score R² = 0.608 (dominant), ESG_composite R² = 0.413
- **Weight robustness:** All profiles Spearman ρ > 0.99 at ±20% weight perturbation
- **PCA:** 3 Kaiser components explain majority of variance; silhouette = 0.277
- **External validation:** Sustainalytics ρ = +0.780 (p = 0.001), MSCI ρ = +0.563 (p = 0.036)
- **Bootstrap stability:** Top-20 stability = 97.7% across 500 iterations

### Generated Outputs
- **160+ CSV tables** covering descriptive stats, correlations, regressions, ANOVA, VIF, weight sensitivity, benchmarks, PCA, clustering, bootstrap CIs, factor ablation, financial validation, proxy calibration, geographic robustness, rankings
- **52+ PNG figures** including score distributions, radar charts, correlation heatmaps, scatter plots, sector boxplots, PCA biplots, dendrograms, CDFs, efficient frontiers, dashboards
- **7 markdown analysis reports** covering benchmarks, tradeoffs, profiles, robustness, sectors

## Configuration System

### `config/index_config.yaml` (659 lines — master config)
- **Universe:** 45 large-cap benchmark tickers, INR/USD rate = 83.0
- **Normalization:** robust z-score (MAD-based), winsorize at 1%/99%
- **Missing data:** 60% minimum coverage, group_median imputation by sector
- **ESG pillars:** E (0.30-0.40), S (0.30-0.40), G (0.25-0.35) with 12 ESG category weights
- **Financial scoring:** 5 categories (profitability 0.30, scale 0.20, efficiency 0.20, stability 0.15, valuation 0.15)
- **Market factors:** 3 categories (liquidity 0.40, volatility 0.30, momentum 0.30)
- **Scoring formula:** `50 + z * 20`, clipped to [0, 100]
- **Preference aggregation:** `rank` mode (percentile-rank transform)
- **3 investor profiles** with full 10-factor weight specifications

### `config/sasb_materiality.yaml` (66 lines)
- SASB-derived sector-specific E/S/G materiality weights for 11 sectors
- Example: Energy = E:0.55/S:0.20/G:0.25; Technology = E:0.20/S:0.45/G:0.35

### `config/data_sources.yaml` (80 lines)
- Yahoo Finance (enabled), SEC EDGAR (enabled)
- Commercial providers (Refinitiv, MSCI, Bloomberg, S&P Global) — all disabled

## Entry Points
- **Main pipeline:** `python scripts/run_all.py` (or `--skip-download`)
- **Individual scripts:** `python scripts/01_download_data.py` through `python scripts/24_geographic_robustness.py`
- **Web app:** `python app.py` → `http://localhost:7860`
- **Tests:** `pytest` (from project root)

## Conventions
- **Script naming:** Numbered `01_` through `24_` for pipeline ordering
- **Score scaling:** All scores normalized to 0-100 via `50 + z * 20` formula
- **Variable types:** 7 types (binary, ordinal, bounded_pct, ratio, count, rate, continuous) with type-aware processing
- **Lower-is-better flipping:** ESG_LOWER_IS_BETTER set in constants.py — emissions, injury rates, risk ratings are automatically inverted
- **Import pattern:** Scripts use `PROJECT_ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(PROJECT_ROOT))`
- **Reproducibility:** `RANDOM_SEED = 42` used throughout
- **Config-driven:** All weights and thresholds from YAML configs, not hardcoded
- **Circularity prevention:** market_score excluded from default weights (DEFAULT_WEIGHTS_EX_MARKET) to prevent forward-looking bias; similarity computed on ESG vectors only

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (downloads data + all analysis)
python scripts/run_all.py

# Run full pipeline (skip data download if already done)
python scripts/run_all.py --skip-download

# Run individual pipeline steps
python scripts/01_download_data.py
python scripts/02_clean_data.py
python scripts/03_build_index.py
# ... through 24

# Run tests (252 tests)
pytest
pytest -v                    # verbose
pytest -k "not slow"         # skip slow tests
pytest -m integration        # integration tests only
pytest tests/test_build_index.py  # specific module

# Launch web interface
python app.py
# Open http://localhost:7860

# Build thesis (LaTeX)
cd Thesis_report && make

# Build paper (LaTeX)
cd Paper && pdflatex Thesis.tex && bibtex Thesis && pdflatex Thesis.tex && pdflatex Thesis.tex
```

## Environment Variables
| Variable | Purpose | Required |
|----------|---------|----------|
| `SEC_EDGAR_USER_AGENT` | User-agent string for SEC EDGAR API requests | Yes (for script 01 only) |

No `.env` file is committed (listed in `.gitignore`). No other environment variables are required.

## Testing

### Test Suite Overview
- **Framework:** pytest 8.3.2 with conftest.py fixtures
- **Total tests:** ~223-252 tests across 13 test modules
- **Test patterns:** Unit tests for each `src/` module + integration tests for full pipeline
- **Markers:** `slow` (long-running), `integration` (end-to-end)
- **Config:** `pytest.ini` — testpaths=tests, addopts=-v --tb=short -q

### Test Coverage by Module
| Test File | Tests | Module Tested |
|-----------|-------|---------------|
| `test_build_index.py` | 38 | `scripts/03_build_index.py` — score construction, z-scoring, variable types |
| `test_clean_data.py` | 42 | `scripts/02_clean_data.py` — type classification, winsorization, outlier detection, imputation |
| `test_composite_index.py` | 22 | `src/index_construction/composite_index.py` — normalization, pillar scores, direction flipping |
| `test_constants.py` | 19 | `src/constants.py` — column lists, weight sums, config alignment |
| `test_cosine_similarity.py` | 23 | `src/similarity/cosine_similarity.py` — 3 metrics, matrix properties, ranking |
| `test_data_pipeline.py` | 10 | `src/data_collection/data_pipeline.py` — YAML loading, config discovery, standardization |
| `test_data_quality.py` | 17 | `src/data_collection/data_quality.py` — missingness, coverage filtering, imputation |
| `test_financial_scorer.py` | 10 | `src/financial_scoring/financial_scorer.py` — financial + market scoring, inversion |
| `test_integration.py` | 7 | Full 6-step pipeline (quality → clean → ESG → financial → similarity → preference) |
| `test_preference_scoring.py` | 21 | `src/similarity/preference_scoring.py` — 3 profiles, 3 aggregation modes, ranking |
| `test_time_splits.py` | 6 | `scripts/07b_cross_sectional_validation.py` — IC, quintile returns, bootstrap |
| `test_utils.py` | 8 | `src/utils.py` — project root, directory creation, data loading |

### Fixtures (from conftest.py)
- `sample_esg_df` — 8 companies, 4 sectors, 16 ESG indicators
- `sample_financial_df` — 8 companies, 11 financial metrics
- `sample_market_df` — 8 companies, 6 market factors
- `sample_scores_df` — 8 companies, all 10 preference scoring components
- `index_config` — Minimal config matching index_config.yaml structure
- `df_with_nans` — 6 rows with intentional NaN values for imputation testing

## Literature Review
Located at `docs/LITERATURE_REVIEW.txt` (406 lines), covering:
1. ESG rating divergence (Berg, Kolbel & Rigobon 2022)
2. Materiality research (Khan, Serafeim & Yoon 2016)
3. Meta-analysis: ESG-financial link (Friede et al. 2015 — 2,200+ studies, r = 0.13-0.18)
4. Mid-cap ESG gap: ESG-financial correlation stronger for mid-caps (0.18-0.22 vs 0.10-0.15 for large-cap)
5. Multi-tier ESG data pipeline design rationale
6. Weight optimization methodology (grid search + cross-validation)
7. SASB sector materiality framework
8. Comparison with existing indices (S&P MidCap 400 ESG, NIFTY Midcap 150 ESG, MSCI USA Mid Cap ESG Leaders)
9. Target venues: PRI Academic Network, GRASFI, EFMA, FMA, JSFI, JPM, Review of Finance

## External Dependencies / Integrations
- **Yahoo Finance API** (`yfinance`) — financial data, market prices, basic ESG risk scores
- **SEC EDGAR XBRL API** — R&D expenditure, governance disclosures (US companies only)
- **Benchmark indices** — S&P 500, S&P MidCap 400, Russell 2000, NIFTY 50 (via Yahoo Finance)
- **ESG rating providers** (validation only) — MSCI, Sustainalytics, S&P Global scores for 27 benchmark companies

## Known Gotchas
1. **ESG data is predominantly synthetic/hybrid** — only Yahoo governance risk scores and SEC EDGAR data are real; all E/S environmental and social indicators are derived from financial proxies or sector-median imputation
2. **Yahoo Finance ESG scores are mostly empty** — `esgScore`, `environmentScore`, `socialScore`, `governanceScore` columns are unpopulated for most companies; only governance risk audit scores are reliable
3. **Script numbering gaps** — there is no `12_` script; scripts 21-24 exist but are not all in `run_all.py`
4. **Market score circularity** — `market_score` is excluded from default weights (`DEFAULT_WEIGHTS_EX_MARKET`) to prevent forward-looking bias; the `DEFAULT_WEIGHTS_WITH_MARKET` is kept for reference only
5. **Indian tickers need `.NS` suffix** — all NSE-listed tickers use Yahoo Finance `.NS` suffix
6. **Currency conversion** — Indian financial data converted at fixed INR/USD = 83.0 (March 2024 RBI rate), not dynamic
7. **SEC EDGAR requires user-agent** — set `SEC_EDGAR_USER_AGENT` environment variable before running `01_download_data.py`
8. **Large-cap filtering** — `load_indexed_data()` filters out 45 large-cap benchmarks by default; pass `include_benchmarks=True` to include them
9. **LaTeX build** — `Thesis_report/Missing_Packages/` contains fallback .sty files for environments without full TeX Live
10. **276 vs 321 companies** — raw universe is 321 tickers, but only 276 pass the 60% minimum indicator coverage filter

## Thesis Documents
- **Full thesis:** `Thesis_report/` — BITS Pilani format, 5 chapters + 3 appendices
- **Conference paper:** `Paper/` — IEEE format, 10 sections
- **Chapters:** Introduction, Literature Review, Data & Methods, Results, Future Work
- **Appendices:** A (supplementary tables), B (supplementary figures), C (Profile Justification)

## Last Updated
2026-04-05
