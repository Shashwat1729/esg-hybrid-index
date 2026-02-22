# Multi-Factor ESG-Integrated Investment Index for Mid-Cap Companies

This repository contains the complete codebase for constructing and evaluating a multi-factor investment index that integrates ESG (Environmental, Social, Governance) metrics with traditional financial, market, and operational factors. The research focuses on mid-capitalization companies across U.S. and Indian markets.

## Research Overview

- **Objective**: Construct a transparent, replicable multi-factor index for mid-cap companies that combines ESG with financial, market, and operational quality factors
- **Sample**: 60 companies (40 U.S. mid-caps from S&P MidCap 400 ESG, 20 Indian mid-caps from NIFTY Midcap 150 ESG)
- **Benchmarks**: S&P MidCap 400 ESG Index, NIFTY Midcap 150 ESG Index, MSCI USA Mid Cap ESG Leaders Index
- **Key Innovation**: Dynamic weight optimization across ten factor categories with empirical validation through 75+ statistical tests, PCA, clustering, bootstrap analysis, cross-validation, market regime analysis, and type-aware data handling (binary, ordinal, continuous)

## Repository Structure

```
Economic Thesis/
├── config/
│   ├── index_config.yaml          # Index weights, normalization, scoring config
│   └── data_sources.yaml          # Data source configuration
├── data/
│   ├── raw/                       # Downloaded raw data files
│   └── processed/                 # Cleaned and indexed datasets
├── docs/
│   ├── LITERATURE_REVIEW.txt      # Comprehensive literature review
│   └── data_sources.xlsx          # Detailed data source documentation
├── scripts/
│   ├── 01_download_data.py        # Download financial, market, ESG data
│   ├── 02_clean_data.py           # Clean, standardize, impute missing data
│   ├── 03_build_index.py          # Construct multi-factor index scores
│   ├── 04_statistical_tests.py    # Comprehensive statistical analysis
│   ├── 05_weight_sensitivity.py   # Weight optimization and sensitivity
│   ├── 06_benchmark_comparison.py # Compare with established indices
│   ├── 07_visualizations.py       # Generate all 30 research figures
│   ├── 08_advanced_analysis.py    # PCA, clustering, bootstrap, ablation
│   ├── 09_generate_report.py      # Compile summary report and key findings
│   └── run_all.py                 # Master pipeline (runs all 9 scripts)
├── src/
│   ├── data_collection/           # Data ingestion and pipeline utilities
│   ├── financial_scoring/         # Financial and market factor scoring
│   ├── index_construction/        # ESG index normalization and aggregation
│   └── similarity/                # Cosine similarity and preference scoring
├── reports/
│   ├── tables/                    # 75+ CSV outputs for research paper tables
│   ├── figures/                   # 30 PNG figures for research paper
│   ├── research_summary.txt       # Compiled research summary
│   └── key_findings.csv           # Key findings table
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline (includes data download)
python scripts/run_all.py

# Or skip data download if raw data already exists:
python scripts/run_all.py --skip-download

# Or run individual steps:
python scripts/01_download_data.py        # Download from Yahoo Finance, SEC EDGAR, NSE
python scripts/02_clean_data.py           # Clean and standardize
python scripts/03_build_index.py          # Build multi-factor index
python scripts/04_statistical_tests.py    # Run all statistical tests
python scripts/05_weight_sensitivity.py   # Weight sensitivity analysis
python scripts/06_benchmark_comparison.py # Compare with benchmark indices
python scripts/07_visualizations.py       # Generate all 30 figures
python scripts/08_advanced_analysis.py    # PCA, clustering, bootstrap, ablation
python scripts/09_generate_report.py      # Compile summary report
```

## Index Methodology

The composite index integrates six factor categories with dynamic weight optimization:

| Factor | Default Weight | Exploration Range | Sub-Factors |
|--------|---------------|-------------------|-------------|
| ESG Score | 15% | 10-30% | Environmental, Social, Governance pillars |
| Financial Score | 25% | 15-40% | Profitability, growth, efficiency, stability, valuation |
| Market Score | 10% | 5-20% | Liquidity, volatility, momentum |
| Operational Score | 10% | 5-15% | Efficiency, innovation, market position, management |
| Risk-Adjusted Score | 10% | 5-15% | Sharpe ratio, Sortino ratio, max drawdown |
| Growth Score | 12% | 5-20% | Revenue growth, earnings growth, margin expansion |
| Value Score | 7% | 3-15% | PE ratio, PB ratio, dividend yield |
| Stability Score | 5% | 2-10% | Earnings consistency, dividend consistency |
| Similarity Rank | 3% | 1-8% | Cosine similarity across factor profiles |
| Sector Position | 3% | 1-8% | Within-sector relative ranking |

Weights are determined through PCA-based factor contribution analysis, empirical grid search optimization, sensitivity analysis, and 5-fold cross-validation rather than fixed a priori.

## Analysis Pipeline

| Step | Script | Outputs |
|------|--------|---------|
| 1. Data Download | `01_download_data.py` | Raw financials, market data, ESG estimates |
| 2. Data Cleaning | `02_clean_data.py` | Variable type classification, adaptive outlier detection, type-aware imputation, winsorized & standardized dataset |
| 3. Index Construction | `03_build_index.py` | Factor scores, composite rankings |
| 4. Statistical Tests | `04_statistical_tests.py` | Normality, correlation, regression, ANOVA, VIF, quintile, decile, heteroscedasticity, binary/ordinal analysis, non-parametric robustness |
| 5. Weight Sensitivity | `05_weight_sensitivity.py` | Grid search, rank stability, profile comparison |
| 6. Benchmark Comparison | `06_benchmark_comparison.py` | Sector composition, portfolio performance, US vs India |
| 7. Visualizations | `07_visualizations.py` | 30 research-quality figures |
| 8. Advanced Analysis | `08_advanced_analysis.py` | PCA, hierarchical & K-means clustering, bootstrap CI, factor ablation, efficient frontier, Gini inequality |
| 9. Report Generation | `09_generate_report.py` | Research summary, key findings |

## Data Sources

- **Financial**: Yahoo Finance (fundamentals, ratios), SEC EDGAR (10-K filings, R&D)
- **Market**: Yahoo Finance (prices, volume, returns, beta)
- **ESG**: Synthetic scores calibrated to Refinitiv/MSCI distributions; real ESG requires subscription
- **Operational**: SEC EDGAR (R&D intensity), derived metrics (asset turnover, SGA efficiency)
- **Benchmarks**: NSE India (NIFTY indices), S&P Global (MidCap 400 ESG)

See `docs/data_sources.xlsx` for full documentation of each source.

## Key Outputs

After running the pipeline, results are saved in `reports/`:

- **75+ tables** in `reports/tables/` covering descriptive statistics, normality tests, correlations, regressions, ANOVA, quintile/decile analysis, VIF, weight sensitivity, benchmark comparisons, alpha/beta decomposition, multi-horizon returns, PCA loadings, cluster profiles, bootstrap CIs, factor ablation, efficient frontier, market regime analysis, factor monotonicity, company rankings, binary/ordinal variable analysis, non-parametric tests, outlier reports, and variable type classification
- **30 figures** in `reports/figures/` including score distributions, correlation heatmaps, sector boxplots, radar charts, scatter matrices, PCA biplots, dendrograms, CDF comparisons, bootstrap confidence intervals, efficient frontier, Gini inequality, weight sensitivity heatmaps, and company profile dashboards
- **Research summary** in `reports/research_summary.txt`
- **Key findings** in `reports/key_findings.csv`
- **Complete indexed dataset** in `data/processed/indexed_data.csv`

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies
- Internet connection for data download (Step 1 only)

## Citation

If using this codebase for research, please cite
