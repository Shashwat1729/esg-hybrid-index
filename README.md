# ESG Hybrid Index — Multi-Factor ESG-Integrated Mid-Cap Index

<p align="center">
  <strong>Transparent · Reproducible · Auditable</strong><br/>
  9-factor ESG-integrated composite for <strong>276</strong> mid-cap equities (186 US + 90 India) + 45 S&P 500 benchmarks
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Shashwat1729/esg-hybrid-index"><img src="https://img.shields.io/badge/🤗_Demo-Hugging_Face_Spaces-FFD21E?style=for-the-badge" alt="HF Spaces"/></a>&nbsp;
  <a href="Paper/Thesis.pdf"><img src="https://img.shields.io/badge/Paper-PDF-C41E3A?style=for-the-badge&logo=adobeacrobatreader" alt="Paper"/></a>&nbsp;
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-1_command-00C853?style=for-the-badge" alt="Quick start"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/tests-308_passed-brightgreen?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/pipeline-19%2F19-success?style=flat-square" alt="Pipeline"/>
  <img src="https://img.shields.io/badge/coverage-276_mid--cap-blue?style=flat-square" alt="Coverage"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/data-Yahoo_%2B_SEC_EDGAR-orange?style=flat-square" alt="Data"/>
</p>

<p align="center">
  <em>All “return” figures are <strong>cross-sectional trailing momentum proxies</strong> (1m/3m/6m at a point in time), not realized forward P&L — see caveat ↓</em>
</p>

---

## ✨ Why this project

Mid-cap ESG investing is bottlenecked by **sparse disclosure, provider disagreement (r≈0.38), and single-factor tilts**. This repo builds a **fully open, config-driven** 9-factor index you can run, audit, fork, and deploy — from raw Yahoo/SEC data to portfolio, validation, paper, and live demo in **one command**.

| What you get | How |
|---|---|
| **Live demo** you can verify, not a mock | Gradio app reads `data/processed/indexed_data.csv` + `reports/tables/*.csv` — zero hardcoded numbers |
| **Provenance per cell** | 6-tier ESG pipeline (`real_yahoo` → `real_sec` → `financial_proxy` → `sector_median` → `global_median` → `NaN`) tracked in `esg_data_provenance.csv` |
| **Leakage-guarded** | `market_score` (contains momentum) excluded from deployed weights; validation uses `market_score_ex_momentum`, `VIF max 3.26 LOW`, `0%` factor overlap |
| **Honest results** | Cross-sectional ranking with ESG as **risk filter** (vol ρ=−0.36, beta ρ=−0.15), not alpha — disclosed with Bonnferroni/BH, bootstrap CIs 247–255 wide |
| **19/19 reproducible** | `run_all.py --skip-download` → bit-identical outputs (seed 42), 308 tests, LaTeX paper + thesis compiled from same outputs |

---

## 🚀 Live Demo

|  |  |
|---|---|
| **Local** | `python app.py` → `http://localhost:7860` |
| **Hosted** | **https://huggingface.co/spaces/Shashwat1729/esg-hybrid-index** · bundle ready in `demo/` — see `docs/DEMO.md` for one-push deploy |

**5 tabs** (all from real outputs):

- **Company Explorer** — 10-factor radar + AAA–CCC rating + balanced rank
- **Company Comparison** — side-by-side radar + profile-weighted winner & differentiator
- **Investment Screener** — filter by ESG/financial/volatility/sector, ranked by `pref_*`
- **Portfolio Builder** — equal-weighted aggregate vs universe radar + sector pie
- **Index Methodology** — live weight sliders (auto-renormalized) → re-ranking
- **About & Methodology** — pipeline, provenance, caveats, dataset stamp `N=321 (276+45) · INR/USD=83.0`, disclaimer

> **Not financial advice.** Past cross-sectional patterns ≠ future returns. See `docs/RESEARCH_AUDIT.md` §5 + Paper §VI.

<p align="center">
  <img src="reports/figures/fig20_summary_dashboard.png" alt="Summary dashboard" width="85%"/>
  <br/><em>Fig. 20 — Index summary dashboard (auto-generated, 200 DPI)</em>
</p>

---

## ⚡ Quick Start

### Option A — Exact replication (recommended, no API keys)

```bash
git clone https://github.com/Shashwat1729/esg-hybrid-index.git
cd esg-hybrid-index
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# set UTF-8 for Windows consoles (fixes ⚠/→ in logs)
# PowerShell: $env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
# Bash:      export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

python scripts/run_all.py --skip-download   # 19/19 · ~7.5 min · uses committed data/raw
pytest -q                                   # 308 passed
python app.py                               # → http://localhost:7860
```

### Option B — Fresh market data (prices will drift)

```bash
# PowerShell
$env:SEC_EDGAR_USER_AGENT="YourName your@email.edu"
$env:PYTHONUTF8="1"; python scripts/run_all.py

# Bash
export SEC_EDGAR_USER_AGENT="YourName your@email.edu"
export PYTHONUTF8=1
python scripts/run_all.py
```

### Option C — Docker

```bash
docker build -t esg-hybrid-index .
docker run -p 7860:7860 esg-hybrid-index   # demo
# or pipeline
docker run --rm -v $(pwd)/reports:/app/reports esg-hybrid-index python scripts/run_all.py --skip-download
```

<details><summary><strong>Make shortcuts</strong></summary>

```bash
make setup      # venv + pip install
make pipeline   # run_all --skip-download
make test       # pytest
make demo       # app.py
make paper      # compile Paper + Thesis_report
```
</details>

---

## 🧩 How it works

### The 9 deployed factors

| # | Factor | What it captures | Config key |
|---|---|---|---|
| 1 | **ESG Composite** | 12 categories → 3 pillars E/S/G → SASB sector-materiality weighted (Energy E 0.55 … Tech S 0.45) | `esg_index` |
| 2 | **Financial** | profitability / scale / efficiency / stability / valuation | `financial_scoring` |
| 3 | **Market** | liquidity / volatility / momentum — *excluded from deployed weights (C1 guard)* | `market_factors` |
| 4 | **Operational** | productivity / innovation / market position | `operational_quality` |
| 5 | **Risk-Adjusted** | Sharpe/Sortino/drawdown + tail risk | `risk_adjusted_scoring` |
| 6 | **Value** | inverted multiples (PE/PB/EV…) | `value_scoring` |
| 7 | **Growth** | revenue/earnings/sustainable growth | `growth_scoring` |
| 8 | **Stability** | leverage / liquidity / vol | `stability_scoring` |
| 9 | **Sector Position** | within-sector percentile | — |

`similarity_rank` (10th, r=0.70 with ESG) excluded. Overlap after dedup **0%**, `VIF max 3.26` LOW, `50 + 10·z` → `[0,100]` single re-standardization, `70/30` sector blend + isotonic monotonicity.

### Investor profiles (sum to 1.0)

| Factor | **ESG-First** | **Balanced** | **Financial-First** |
|---|---:|---:|---:|
| ESG | **0.55** | **0.22** | 0.05 |
| Financial | 0.08 | **0.18** | 0.25 |
| Market | 0.02 | 0.05 | 0.10 |
| Operational | 0.05 | 0.06 | 0.08 |
| Risk-Adjusted | 0.08 | **0.15** | 0.18 |
| Growth | 0.06 | 0.14 | 0.15 |
| Value | 0.03 | 0.04 | 0.07 |
| Stability | 0.06 | 0.08 | 0.07 |
| Sector Pos. | 0.07 | 0.08 | 0.05 |

From `config/index_config.yaml` (`DEFAULT_WEIGHTS` = ex-market clean; `_with_market` kept for audit). Aggregation: percentile-rank → weighted sum.

### Architecture

```mermaid
flowchart LR
  A[Yahoo Finance\n321 tickers\n+ SEC EDGAR\n+ NSE benchmarks] --> B[02_clean_data\n7 types · IQR/MAD/Z\nWinsor 1/99 · INR→USD 83]
  B --> C[03_build_index\nrobust MAD-z · SASB\n9 factors · sector blend]
  C --> D[04–24 Validation\nIC · quintiles · bootstrap B=500\nCV · walk-forward 165 · high-cap 45]
  D --> E[07_visualizations\n57 figs 200dpi]
  E --> F[09_report\nresearch_summary.txt]
  F --> G[app.py / demo\nGradio 5 tabs]
```

---

## 📦 Repository structure

```
.
├── app.py                 # Gradio (Company Explorer / Comparison / Screener / Portfolio / Methodology / About)
├── demo/                  # Hugging Face Space bundle (app.py + data/processed + config + README)
├── config/
│   ├── index_config.yaml       # universe, normalization, 9-factor weights, 3 profiles
│   ├── sasb_materiality.yaml   # sector E/S/G (11 sectors)
│   └── data_sources.yaml       # Yahoo/SEC on, commercial off
├── src/
│   ├── constants.py            # 9-factor weights ex-market, type metadata, LOWER_IS_BETTER
│   ├── utils.py                # robust_zscore, load_indexed_data, paths
│   ├── data_collection/        # data_pipeline, data_quality
│   ├── financial_scoring/      # financial_scorer, market_scorer
│   ├── index_construction/     # composite_index (normalize→aggregate→rescale)
│   └── similarity/             # cosine_similarity, preference_scoring
├── scripts/
│   ├── run_all.py              # canonical 19-step orchestrator (UTF-8 enforced)
│   ├── 01_download_data.py     # 6-tier hybrid ESG (real_yahoo/real_sec/financial_proxy/…)
│   ├── 02_clean_data.py … 24_geographic_robustness.py
│   └── 07_visualizations.py    # 57 figs 200dpi
├── tests/                 # 308 tests (13 modules + test_research_correctness 36 guards)
├── data/
│   ├── raw/               # committed 321-ticker snapshot (regenerated via 01)
│   └── processed/         # indexed_data.csv 321×202 + cleaning_metadata.json (tracked)
├── reports/
│   ├── tables/            # 170+ CSVs (VIF, IC, quintiles, bootstrap, CV, high-cap…)
│   ├── figures/           # 57 PNGs 200dpi (tracked for paper)
│   ├── research_summary.txt + key_findings.csv
│   └── analysis/          # 40+ deeper dives
├── docs/
│   ├── METHODOLOGY.md          # full pipeline & equations
│   ├── REPRODUCIBILITY.md      # one-command repro, seeds, env, what can't be reproduced
│   ├── RESEARCH_AUDIT.md       # leakage/bias/formula audit + verified numbers
│   ├── PAPER_CODE_RECONCILIATION.md # Paper ↔ code ↔ output line-by-line
│   └── DEMO.md                 # HF Spaces exact steps, Space card, troubleshooting
├── Paper/                 # IEEE conference paper (LaTeX, Figures/ 28 PNGs)
├── Thesis_report/         # BITS Pilani thesis (LaTeX, Figures/ 24 PNGs, Missing_Packages/)
├── requirements.txt       # pinned (pandas 2.2.2, numpy 1.26.4, scipy, sklearn, statsmodels…)
├── pytest.ini             # 308 tests, markers slow/integration
└── LICENSE                # MIT 2026 Shashwat Bajpai
```

---

## 🔧 Configuration

All weights, thresholds, and methodology live in YAML — no code edits needed:

| File | What to tweak |
|---|---|
| `config/index_config.yaml` | universe (45 benchmarks, `large_cap_benchmarks`), `normalization: robust_zscore`, pillar/category weights, `preference_scoring.investor_profiles`, `RANDOM_SEED 42` |
| `config/sasb_materiality.yaml` | per-sector E/S/G (e.g. Energy 0.55/0.20/0.25) |
| `config/data_sources.yaml` | enable/disable Yahoo/SEC/commercial |

Change → `python scripts/03_build_index.py` → outputs, figures, tables, paper numbers update.

---

## 📊 Results — honest headline

> **Defensible cross-sectional quality ranking with ESG as a risk filter**, not realized alpha.

| Finding | Value | Source |
|---|---:|---|
| Universe | **276** mid-cap (186 US + 90 India) + 45 benchmarks = 321, **202** vars, **11** sectors | `indexed_data.csv` |
| Coverage | **6-tier** provenance: `real_yahoo 213` / `real_sec 102` / `financial_proxy 6` per firm | `esg_data_provenance.csv` |
| Max VIF | **3.26 LOW** (financial) | `factor_vif.csv` |
| 0% overlap | deduped 0% shared indicators | `indicator_factor_mapping.csv` |
| ESG↔financial | **R² 0.544** — *proxy-construction artifact*, not discovery | `correlation_pearson.csv` (r 0.738) |
| High-cap transfer | **Kendall W 0.933, ρ 0.9993, 9/9 pass |z|<2.5, p 0.50, 0.2% degr.** | `15_robustness_highcap.py` |
| Bootstrap | **τ 1.000** but CI **247–255 wide** (portfolio-level only) | `07b` B=500 |
| Monotonicity | **7/8 vs forward quality**; **2/9 vs JT return proxy** | `predictive_validation_summary.csv` |
| Capacity | **PASS $25M–$500M**, fail $1B+ | `16_financial_validation.py` |
| ESG risk filter | **vol ρ −0.36 p<0.001, beta ρ −0.15 p=0.015**, drawdown **−27.8% vs −38.3% 10.5pp** but worse Sharpe | `04` |
| PCA | **3 comps 67.4% (31.8/20.9/14.6)**, silhouette 0.261 (3 clusters) | `advanced_pca_variance.csv` |
| Weight stability | **≥0.99 Spearman at ±20%** | `05` |

Benchmark “excess momentum’’ e.g. **+10.65 pp (β 0.82, bear IR +0.585), bear +7.27 pp, balanced −4.01% CS-IR −0.710** = *cross-sectional momentum-proxy differential*, not P&L. Every `benchmark_*` CSV prefixed `# Cross-sectional momentum proxy, not time-series returns` + Paper §IV footnote + §VI Limitations.

---

## ⚠️ Key Caveat — Momentum Proxies (M4)

> **Trailing `price_momentum_1m/3m/6m` at a point in time.** Selected mean − universe mean; `mean/std` = cross-sectional IR. No forward returns, no timestamped rebalancing, no P&L.

---

## 🧪 Development

```bash
# tests
pytest -q                      # 308 passed
pytest tests/test_research_correctness.py -v   # leakage/overlap/direction guards (36)
pytest -m "not slow"           # skip 500-iter bootstrap
pytest -m integration          # full pipeline integration

# lint / type (optional)
ruff check src/ scripts/
mypy src/

# single stage
python scripts/03_build_index.py   # idempotent, bit-identical on same clean_data.csv
```

CI: `CI=true pytest -q` (no prompts), `python scripts/14_run_checks.py` health checks.

---

## ☁️ Deployment

### Hugging Face Spaces (CPU basic, no GPU)

`demo/` **is** the Space. Push whole repo or just `demo/`:

```bash
# 1. create Space at hf.co/new-space → SDK Gradio, public, hardware CPU basic
# 2. push
git clone https://huggingface.co/spaces/<user>/esg-hybrid-index hf-space
cp -r demo/* hf-space/   # or cp app.py + data/processed/indexed_data.csv + config/
# ensure hf-space/README.md has Space card (title, sdk: gradio 4.44.0, app_file: app.py)
cd hf-space && git add . && git commit -m "deploy" && git push   # auto-builds ~1–2 min
```

`demo/README.md` Space card, `demo/requirements.txt` minimal (`pandas,numpy,gradio,plotly,pyyaml`). No secrets.

### Docker / Local

```bash
docker build -t esg-hybrid-index .
docker run -p 7860:7860 esg-hybrid-index
# or
python app.py --server-name 0.0.0.0 --server-port 7860
```

---

## 📚 Documentation

| Doc | Purpose |
|---|---|
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Full pipeline, 7 variable types, 3-stage normalization, SASB, profile weights |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | One-command repro, seeds, configs, `--skip-download` bit-identical, env `SEC_EDGAR_USER_AGENT`, `PYTHONUTF8`, what can't be reproduced (survivorship, live prices) |
| [`docs/RESEARCH_AUDIT.md`](docs/RESEARCH_AUDIT.md) | Leakage/bias/formula audit, verified numbers, remaining limitations |
| [`docs/PAPER_CODE_RECONCILIATION.md`](docs/PAPER_CODE_RECONCILIATION.md) | Every table/figure ↔ CSV ↔ script ↔ LaTeX |
| [`docs/DEMO.md`](docs/DEMO.md) | HF Spaces exact steps, Space card, troubleshooting |
| [`Project.md`](Project.md) | Architecture, directory map, entry points |
| `reports/research_summary.txt` | Auto-generated 22-section report (743 lines) |

Paper: `Paper/Thesis.tex` (IEEE), Thesis: `Thesis_report/main.tex` (BITS) — both compile from same `reports/`.

---

## 🙏 Acknowledgments

Yahoo Finance (`yfinance`), SEC EDGAR XBRL, SASB Materiality Map, Fama-French / Jegadeesh-Titman / Amihud / Gompers et al. / Khan et al. literature, Gradio, Hugging Face Spaces.

---

## 📄 Citation & License

```bibtex
@misc{bajpai2026esg,
  author = {Shashwat Bajpai},
  title  = {Multi-Factor ESG-Integrated Investment Index for Mid-Cap Equities (US + India)},
  year   = {2026},
  publisher = {GitHub},
  url    = {https://github.com/Shashwat1729/esg-hybrid-index}
}
```

**MIT** © 2026 Shashwat Bajpai — see [`LICENSE`](LICENSE). Data via Yahoo/SEC fair-access; Indian monetary fields converted at **INR/USD 83.0** (Mar 2024 RBI) stored in `data/processed/cleaning_metadata.json`.

---

<p align="center">
  <em>Academic research demonstration — not financial advice. Past cross-sectional patterns do not predict future performance.</em><br/>
  <sub>Built with ❤️ for reproducible finance research · PRs welcome</sub>
</p>
