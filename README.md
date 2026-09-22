# Multi-Factor ESG-Integrated Investment Index — Mid-Cap (US + India)

> **Transparent 9-factor ESG-integrated composite for 276 mid-cap equities (186 US + 90 India).**  
> Built from public data (Yahoo Finance, SEC EDGAR) with hybrid ESG proxies; 45 large-cap benchmarks for robustness.  
> All "return" figures are **cross-sectional trailing momentum proxies**, not realized forward P&L — see caveat below.

[![Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/Shashwat1729/esg-hybrid-index)
[![Paper](https://img.shields.io/badge/paper-PDF-red)](Paper/Thesis.tex)
[![Tests](https://img.shields.io/badge/tests-308%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)

## Table of Contents

- [Overview](#overview)
- [Methodology (9-Factor)](#methodology)
- [Pipeline](#pipeline)
- [Public Demo](#public-demo)
- [Installation](#installation)
- [Reproducing the Research](#reproducing-the-research)
- [Documentation](#documentation)
- [Results (Honest Summary)](#results)
- [Key Caveat — Momentum Proxies](#key-caveat--momentum-proxies)
- [Citation](#citation)

## Overview

Current mid-cap ESG investing is bottlenecked by sparse, inconsistent non-financial disclosure. This project constructs a **reproducible, auditable 9-factor composite** that integrates **ESG quality** with **financial/market/operational** signals for mid-cap equity evaluation — tested on S&P MidCap 400 / Russell Midcap and NIFTY Midcap 150 universes.

- **Hybrid ESG pipeline (6 tiers per `company × indicator` cell):** real Yahoo governance risk → real SEC EDGAR governance → financial-proxy indicators → sector-median → global-median → NaN, with full provenance in `reports/tables/esg_data_provenance.csv`.
- **Cross-sectional ranking, not a backtest:** single snapshot as of March 2025; no delisted-firm panel (survivorship bias disclosed).
- **Circularity-corrected:** `market_score` (contains momentum) excluded from deployed preference weights; validation uses `market_score_ex_momentum` and a forward quality proxy.

Full methodology: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) · Audit: [`docs/RESEARCH_AUDIT.md`](docs/RESEARCH_AUDIT.md) · Paper↔code map: [`docs/PAPER_CODE_RECONCILIATION.md`](docs/PAPER_CODE_RECONCILIATION.md).

## Methodology

### 9 Deployed Factors

| # | Factor | Construction |
|---|--------|-------------|
| 1 | **ESG Composite** | 12 ESG categories → 3 pillars (E/S/G) → SASB sector-materiality weighted composite |
| 2 | **Financial Score** | profitability / scale / efficiency / stability / valuation (5 categories) |
| 3 | **Market Score** | liquidity / volatility / momentum — *excluded from deployed weights (circularity guard)* |
| 4 | **Operational Score** | productivity, innovation, profitability |
| 5 | **Risk-Adjusted Score** | Sharpe/Sortino/drawdown + liquidity/tail-risk |
| 6 | **Value Score** | inverted multiples (P/E, P/B, P/S, EV/EBITDA) |
| 7 | **Growth Score** | revenue/earnings growth, sustainable growth (no momentum) |
| 8 | **Stability Score** | liquidity / leverage / volatility |
| 9 | **Sector Position** | within-sector percentile |

`similarity_rank` (10th legacy factor) excluded — collinear with ESG_composite (r=0.70). Indicator overlap after dedup: **0%**. All scores → `50 + 10·z`, clipped [0,100] (single re-standardization).

### Investor Profiles

| Factor | ESG-First | **Balanced** | Financial-First |
|--------|-----------|-------------|-----------------|
| ESG | 0.55 | **0.22** | 0.05 |
| Financial | 0.08 | **0.18** | 0.25 |
| Market | 0.02 | **0.05** | 0.10 |
| Risk-Adjusted | 0.08 | **0.15** | 0.18 |
| Growth | 0.06 | **0.14** | 0.15 |
| Value | 0.03 | **0.04** | 0.07 |
| Stability | 0.06 | **0.08** | 0.07 |
| Sector Pos. | 0.07 | **0.08** | 0.05 |

Weights from `config/index_config.yaml`; deployed `pref_*` are ex-market (clean).

## Pipeline

```
scripts/
├── 01_download_data.py          # Yahoo Finance + SEC EDGAR + benchmarks + ESG hybrid
├── 02_clean_data.py              # Type-aware cleaning, winsorization, INR→USD (83.0)
├── 03_build_index.py             # 9-factor scoring, sector blend, isotonic monotonicity, PCA rationale
├── 04_statistical_tests.py       # Normality, correlations, VIF, ANOVA, quintiles
├── 05_weight_sensitivity.py      # 99-combo grid search, perturbation stability
├── 06_benchmark_comparison.py    # Multi-horizon + alpha/beta + regime + clean IC
├── 07_visualizations.py          # 30+ publication figures
├── 07b_cross_sectional_validation.py  # IC, quintile spreads, bootstrap (B=500), CV
├── 08_advanced_analysis.py       # PCA, clustering, ablation, frontier
├── 09_generate_report.py         # 22-section research summary
├── 10_esg_benchmarking.py        # External provider correlations (27 firms)
├── 11_profile_justification.py   # Profile differentiation & overlap
├── 15_robustness_highcap.py      # 45 large-cap transfer (Kendall W=0.933)
├── 16_financial_validation.py    # Turnover, capacity ($25M–$500M pass)
├── 17_proxy_validation.py        # Provenance audit + held-out proxy validation
├── 18_sector_cv.py               # Leave-one-sector-out CV
├── 19_synthetic_sensitivity.py   # Noise injection + factor dropout
├── 20_subsampling_stability.py   # 500-iteration subsampling stability
├── 21_temporal_stability.py      # Temporal OOS (quasi-temporal splits)
├── 22_esg_incremental_value.py   # Incremental ESG R² beyond sector dummies
├── 23_pca_weight_validation.py   # PCA vs configured weight audit
├── 24_geographic_robustness.py   # US vs India normalization bias
└── run_all.py                    # Canonical orchestrator (use --skip-download for exact replication)
```

`src/` library: `data_pipeline`, `data_quality`, `financial_scorer`, `composite_index`, `cosine_similarity`, `preference_scoring` — see [`Project.md`](Project.md).

## Public Demo

**Interactive verification demo** — uses the actual research outputs, not a fake model:

- **Company Explorer** — 10-factor radar + ESG rating (AAA–CCC) + rank
- **Company Comparison** — side-by-side radar + profile-weighted recommendation
- **Investment Screener** — threshold filter (ESG/financial/volatility/sector) + screening
- **Portfolio Builder** — equal-weighted aggregate vs universe + sector diversification
- **Index Methodology** — live weight sliders (auto-normalized) → re-ranking
- **About & Methodology** — pipeline, provenance, caveats, links, dataset version

| Access | Command / URL |
|--------|---------------|
| **Local** | `python app.py` → `http://localhost:7860` |
| **Hosted (Hugging Face Spaces)** | `https://huggingface.co/spaces/Shashwat1729/esg-hybrid-index` *(deploy via `docs/DEMO.md` — bundle ready in `demo/`)* |

Every chart/table reads `data/processed/indexed_data.csv` / `reports/tables/*.csv` — no hardcoded demo numbers. Deployment steps with Space card front-matter: [`docs/DEMO.md`](docs/DEMO.md). Demo footer shows dataset version/date and a **Not financial advice** disclaimer; all benchmark tables carry the header `# Cross-sectional momentum proxy, not time-series returns`.

## Installation

- **Python 3.10+** (developed on 3.11/3.12)
- No paid APIs; no secrets except optional `SEC_EDGAR_USER_AGENT` for fresh downloads.

```bash
git clone https://github.com/Shashwat1729/esg-hybrid-index.git
cd esg-hybrid-index
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Research

### Exact replication (recommended — uses committed raw data)

```bash
pip install -r requirements.txt
python scripts/run_all.py --skip-download   # regenerates data/processed, reports/tables, reports/figures
pytest -q                                   # 308 tests
python app.py                               # demo on :7860
```

### With fresh downloads (prices will drift; use for live update)

```bash
# PowerShell: $env:SEC_EDGAR_USER_AGENT = "YourName your@email.edu"
# Bash:      export SEC_EDGAR_USER_AGENT="YourName your@email.edu"
python scripts/run_all.py
```

Determinism: `RANDOM_SEED=42`, deterministic winsorization/imputation, `--skip-download` → bit-identical outputs. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for full instructions, env vars, and what cannot be reproduced exactly (survivorship-free panel, live prices).

Individual stages: `python scripts/02_clean_data.py`, `python scripts/03_build_index.py`, … `python scripts/24_geographic_robustness.py`. Each is idempotent.

## Documentation

| Doc | Purpose |
|-----|---------|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Full pipeline, variable types, scoring equations, SASB weights, profile weights |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | One-command reproduction, determinism, configs, data requirements |
| [`docs/RESEARCH_AUDIT.md`](docs/RESEARCH_AUDIT.md) | Independent audit: leakage, bias, formula checks, verified key results |
| [`docs/PAPER_CODE_RECONCILIATION.md`](docs/PAPER_CODE_RECONCILIATION.md) | Paper ↔ code ↔ output mapping for every key table/figure |
| [`docs/DEMO.md`](docs/DEMO.md) | HF Spaces deployment (exact steps, Space card, troubleshooting) |
| [`Project.md`](Project.md) | Architecture, directory map, entry points, conventions |

## Results

**Honest headline:** the framework is a **defensible cross-sectional quality ranking with ESG as a risk filter**, not evidence of realized alpha.

| Finding | Value | Source |
|---------|-------|--------|
| Universe | 276 mid-cap (186 US + 90 India) + 45 benchmarks | `indexed_data.csv` |
| Variables | 202 | `indexed_data.csv` |
| Max VIF | 2.80 (LOW) | `factor_vif.csv` |
| ESG–financial R² | 0.544 — **proxy-construction artifact**, not independent discovery | `correlation_pearson.csv` |
| High-cap transfer | Kendall W=0.933, ρ=0.9993, 9/9 factors pass z-test | `15_robustness_highcap.py` |
| Bootstrap | Kendall τ=1.000 but rank CIs 247–255 wide (portfolio-level only) | `07b_cross_sectional_validation.py` |
| Monotonicity | 7/8 vs forward quality proxy; 2/9 vs JT return proxy | `predictive_validation_summary.csv` |
| Capacity | Pass $25M–$500M; fail at $1B+ | `16_financial_validation.py` |
| ESG risk filter | Volatility ρ=−0.36 (p<0.001), beta ρ=−0.15 (p=0.015) | `04_statistical_tests.py` |

Benchmark "excess momentum" (e.g. +10.65 pp, bear-market +7.27 pp) = **cross-sectional momentum-proxy differential**, not P&L. Table headers carry this disclaimer; see paper footnotes + §VI Limitations.

## Key Caveat — Momentum Proxies

> **All performance metrics labeled "momentum proxy" are trailing `price_momentum_1m/3m/6m` at a single point in time.** They measure relative stock-selection quality across N companies, not time-series portfolio returns. `benchmark_*` CSVs are prefixed `# Cross-sectional momentum proxy, not time-series returns`.

## Citation

```bibtex
@misc{esg2025,
  author = {Shashwat Bajpai},
  title  = {Multi-Factor ESG-Integrated Investment Index for Mid-Cap Equities (US + India)},
  year   = {2025},
  publisher = {GitHub},
  url    = {https://github.com/Shashwat1729/esg-hybrid-index}
}
```

---

*Academic research demonstration — not financial advice. Past cross-sectional patterns do not predict future performance.*
