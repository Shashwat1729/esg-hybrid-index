# ESG Hybrid Index

A multi-factor framework integrating Environmental, Social, and Governance (ESG) metrics with traditional financial factors for portfolio construction and analysis.

## Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
- [Pipeline](#pipeline)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Overview

This research constructs a hybrid ESG index that combines sustainability scores with financial performance metrics to identify companies that excel on both dimensions. Unlike traditional approaches that treat ESG as a separate screen, our framework integrates ESG data directly into a multi-factor model alongside value, growth, quality, size, and momentum factors.

### Motivation

Current ESG investing faces three challenges:
1. **Proprietary data fragmentation**: ESG scores from different providers disagree (correlation as low as 0.38)
2. **Trade-off naivety**: Simple exclusion-based approaches ignore companies that score well on both ESG and financial metrics
3. **Geographic inconsistency**: ESG data quality varies significantly between developed and emerging markets

### Key Contributions

- A transparent, reproducible ESG scoring methodology using public data
- Multi-factor index construction that balances ESG and financial objectives
- Comprehensive validation against market benchmarks (S&P 500, Nifty 50, Russell 2000)
- Geographic robustness analysis (US vs. India markets)
- Proxy calibration for missing data scenarios

## Methodology

### ESG Scoring Framework

```
+------------------------------------------------------------+
|                      ESG Scoring                            |
+------------------------------------------------------------+
|                                                             |
|  Raw Data Sources                                           |
|  +----------+ +----------+ +----------+ +----------+      |
|  | Yahoo    | | SEC     | | Public   | | SASB     |      |
|  | Finance  | | Filings | | Reports  | | Material-|      |
|  |          | |         | |          | | ity Map  |      |
|  +----------+ +----------+ +----------+ +----------+      |
|       |            |           |            |               |
|       v            v           v            v               |
|  +--------------------------------------------------+      |
|  |          Data Cleaning & Normalization            |      |
|  |  - Missing value imputation                      |      |
|  |  - Outlier treatment (IQR + winsorization)        |      |
|  |  - Cross-source reconciliation                   |      |
|  +--------------------------------------------------+      |
|       |                                                    |
|       v                                                    |
|  +--------------------------------------------------+      |
|  |          Pillar Scoring                           |      |
|  |  +--------+  +--------+  +--------+              |      |
|  |  |   E    |  |   S    |  |   G    |              |      |
|  |  |Emissions| |Labor   | |Board   |              |      |
|  |  |Resource | |Rights  | |Structure|              |      |
|  |  |Mgmt     | |Safety  | |Pay     |              |      |
|  |  +--------+  +--------+  +--------+              |      |
|  +--------------------------------------------------+      |
|       |                                                    |
|       v                                                    |
|  +--------------------------------------------------+      |
|  |       Composite ESG Score (PCA + Equal Weight)    |      |
|  +--------------------------------------------------+      |
+------------------------------------------------------------+
```

### Multi-Factor Index Construction

The final index score for company i is:

```
Score_i = w_ESG * Z_ESG_i + w_F * Z_F_i + w_G * Z_G_i + w_M * Z_M_i
```

where:
- Z_ESG: ESG composite score (standardized)
- Z_F: Financial factor score (ROE, earnings yield, etc.)
- Z_G: Growth factor score (revenue growth, earnings growth)
- Z_M: Momentum factor score (6-month return, volatility-adjusted)

Weights w are optimized via grid search to maximize information coefficient (IC) across validation periods.

### Geographic Robustness

The framework is tested on two distinct markets:
- **US**: S&P 500 constituents (2015-2024)
- **India**: Nifty 50 constituents (2015-2024)

Cross-market analysis reveals systematic differences in ESG data availability and factor efficacy, leading to market-specific calibration adjustments.

## Pipeline

```
scripts/
+-- 01_download_data.py          # Fetch raw ESG + financial data
+-- 02_clean_data.py              # Clean, impute, normalize
+-- 03_build_index.py             # Construct composite index
+-- 04_statistical_tests.py       # Statistical validation
+-- 05_weight_sensitivity.py      # Weight optimization grid search
+-- 06_benchmark_comparison.py    # Compare vs market benchmarks
+-- 07_visualizations.py          # Generate figures
+-- 08_advanced_analysis.py       # PCA, clustering, factor analysis
+-- 09_generate_report.py         # Produce summary report
+-- 10_esg_benchmarking.py        # Compare with ESG providers
+-- 11_profile_justification.py   # Investor profile construction
+-- 15_robustness_highcap.py      # Large-cap robustness checks
+-- 16_financial_validation.py    # Financial ratio validation
+-- 17_proxy_validation.py        # Missing data proxy calibration
+-- 18_sector_cv.py               # Sector cross-validation
+-- 19_synthetic_sensitivity.py   # Synthetic data robustness
+-- 20_subsampling_stability.py   # Bootstrap stability analysis
+-- 21_temporal_stability.py      # Time-split validation
+-- 22_esg_incremental_value.py   # Value-add analysis
+-- 23_pca_weight_validation.py   # PCA weight validation
+-- 24_geographic_robustness.py   # US vs India comparison
+-- run_all.py                    # Execute full pipeline
```

## Key Components

### Core Library (`src/`)

| Module | Description |
|--------|-------------|
| `data_pipeline.py` | Multi-source data ingestion (Yahoo Finance, SEC, public reports) |
| `data_quality.py` | Data quality assessment and missing value treatment |
| `financial_scorer.py` | Financial factor computation (ROE, earnings yield, momentum) |
| `composite_index.py` | Multi-factor index construction with configurable weights |
| `cosine_similarity.py` | Company similarity analysis for peer comparison |
| `preference_scoring.py` | Investor preference profile construction |

### Configuration (`config/`)

| File | Description |
|------|-------------|
| `data_sources.yaml` | API endpoints, ticker lists, date ranges |
| `index_config.yaml` | Factor weights, scoring parameters, rebalancing schedule |
| `sasb_materiality.yaml` | SASB materiality map for industry-specific ESG factors |

## Installation

### Prerequisites

- Python 3.10+
- Virtual environment recommended

### Setup

```bash
git clone https://github.com/Shashwat1729/esg-hybrid-index.git
cd esg-hybrid-index
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Full Pipeline

```bash
python scripts/run_all.py
```

This executes all stages sequentially: data download, cleaning, index construction, statistical tests, benchmarking, and report generation.

### Individual Stages

```bash
# Data pipeline
python scripts/01_download_data.py
python scripts/02_clean_data.py

# Index construction
python scripts/03_build_index.py

# Analysis
python scripts/04_statistical_tests.py
python scripts/05_weight_sensitivity.py
python scripts/06_benchmark_comparison.py

# Reports
python scripts/07_visualizations.py
python scripts/09_generate_report.py
```

## Results

### Benchmark Comparison

| Metric | ESG-Only | Financial-Only | Hybrid Index |
|--------|----------|----------------|--------------|
| Annualized Return | 9.2% | 11.8% | 13.4% |
| Sharpe Ratio | 0.61 | 0.78 | 0.92 |
| Max Drawdown | -22% | -18% | -16% |
| ESG Score (avg) | 72.4 | 51.2 | 68.7 |

### Factor Contribution

```
Value  +######################## 32%
Quality+##################     28%
Momentum+###########           18%
ESG    +##########             16%
Growth +####                    6%
```

### Geographic Analysis

The hybrid index performs consistently across US and Indian markets, with slightly higher ESG scores achieved in the US (due to better data coverage) and stronger financial factor contributions in India.

## Citation

```bibtex
@misc{esg2025,
  author = {Shashwat Bajpai},
  title = {ESG Hybrid Index: Multi-Factor ESG Integration for Portfolio Construction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Shashwat1729/esg-hybrid-index}
}
```
