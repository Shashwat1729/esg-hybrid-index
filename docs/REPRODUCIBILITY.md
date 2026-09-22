# Reproducibility

> One obvious command to reproduce the analysis; no hardcoded secrets; deterministic seeds.

## 1. Quick Start (≤ 5 minutes, no downloads)

Uses **committed raw data** already in `data/raw/` and regenerates all processed outputs, tables, figures, and demo.

```bash
# 1 — create environment
python -m venv .venv
# Windows: .venv\Scripts\activate   |  macOS/Linux: source .venv/bin/activate

# 2 — install pinned dependencies
pip install -r requirements.txt

# 3 — rebuild processed data + index + all analysis (skip network downloads)
python scripts/run_all.py --skip-download

# 4 — verify
pytest -q
python app.py          # → http://localhost:7860
```

Expected: `308 passed, 50 warnings` (tests), `data/processed/indexed_data.csv` (321×202), `reports/tables/` ~174 CSVs, `reports/figures/` ~60 PNGs.

## 2. Full Pipeline (with fresh downloads)

Requires network + SEC EDGAR user-agent. Re-downloads Yahoo Finance + SEC + EPA + World Bank.

```bash
# Set SEC EDGAR user-agent (required by SEC fair-access policy)
# PowerShell:
$env:SEC_EDGAR_USER_AGENT = "YourName your@email.edu"
# Bash:
export SEC_EDGAR_USER_AGENT="YourName your@email.edu"

python scripts/run_all.py
```

Fresh downloads will **overwrite** `data/raw/*.csv`; cleaned/processed outputs regenerate deterministically. Yahoo Finance data is point-in-time (prices move); expect small numeric drift if re-downloaded months later. For exact replication of the paper, use the committed `data/raw/` and `--skip-download`.

## 3. Individual Stages

```bash
python scripts/02_clean_data.py            # → data/processed/clean_data.csv
python scripts/03_build_index.py            # → data/processed/indexed_data.csv
python scripts/04_statistical_tests.py      # → 25+ tables
python scripts/05_weight_sensitivity.py
python scripts/06_benchmark_comparison.py
python scripts/07b_cross_sectional_validation.py
python scripts/08_advanced_analysis.py
python scripts/07_visualizations.py         # → 30+ figures
python scripts/09_generate_report.py        # → reports/research_summary.txt
# … plus 10_ through 24_ (see scripts/run_all.py)
python scripts/14_run_checks.py             # pipeline health checks
```

Each script is idempotent — re-running overwrites its outputs.

## 4. Determinism

| Concern | Handling |
|---------|----------|
| **Random seeds** | `src/constants.py: RANDOM_SEED = 42`; used in `np.random`, `sklearn`, bootstrap, subsampling |
| **Bootstrap / CV splits** | Fixed `random_state=42`; stratified by sector where applicable |
| **Winsorization / imputation** | Deterministic quantiles and medians; no random fill |
| **Parallelism** | No nondeterministic ordering; DataFrame sorts are stable |
| **Floating point** | `pandas`/`numpy` deterministic at fixed versions (see `requirements.txt`) |

Re-running `03_build_index.py` on identical `clean_data.csv` yields **bit-identical** `indexed_data.csv` (verified via `pytest` regression tests).

## 5. Dependencies

Pinned in `requirements.txt` (Python 3.10+):

```
pandas>=2.2.2, numpy>=1.26.4, scipy>=1.13.1, scikit-learn>=1.5.1
pyyaml>=6.0.1, yfinance>=0.2.40, requests>=2.31.0
gradio>=4.44.0, matplotlib>=3.9.2, plotly>=5.24.0, seaborn>=0.13.2
statsmodels>=0.14.2, openpyxl>=3.1.5, pytest>=8.3.2
```

Optional (download only): `yfinance`, `requests` for live market/SEC data.

No private data, no paid APIs. Commercial ESG providers (MSCI, Sustainalytics, etc.) are **disabled** in `config/data_sources.yaml` and used only for external benchmarking of 27 companies (`data/benchmarking/`).

## 6. Configuration

All weights, thresholds, and methodology choices live in YAML, not hardcoded:

| File | Controls |
|------|----------|
| `config/index_config.yaml` | Universe, normalization method, pillar/category weights, investor profiles, validation params |
| `config/sasb_materiality.yaml` | Sector-specific E/S/G materiality weights |
| `config/data_sources.yaml` | API endpoints, ticker coverage toggles |

Changing a weight → re-run `03_build_index.py` → outputs update. No code edits needed.

## 7. Data Requirements

| Tier | Availability | Reproducibility |
|------|--------------|-----------------|
| Yahoo Finance financials + market history | Public API; committed in `data/raw/` | Regenerable via `01_download_data.py` |
| SEC EDGAR XBRL | Public API (fair-access) | Regenerable with `SEC_EDGAR_USER_AGENT` |
| Hybrid ESG (synthetic + proxies) | Generated locally; provenance tracked | Regenerated deterministically |
| Commercial ESG benchmarks (27 firms) | Partial, in `data/benchmarking/` | Reference only; not required to reproduce main results |
| Outputs (`data/processed/`, `reports/`) | Generated; committed for verification | Fully regenerable |

**No fabricated data.** Synthetic ESG indicators are flagged in provenance and absent from `reports/tables/esg_data_provenance.csv` as `financial_proxy`; never presented as observed ESG.

## 8. Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `SEC_EDGAR_USER_AGENT` | Only for `01_download_data.py` | SEC EDGAR fair-access header |

No `.env` file is committed (gitignored). No other secrets.

## 9. Platform Notes

- Developed on **Python 3.11/3.12** (Windows); tested on 3.10+.  
- `yahoo_financials` download may be rate-limited — script retries 3× with backoff.  
- `app.py` binds `0.0.0.0:7860`; no external credentials.

## 10. Verifying Outputs Match the Paper

```bash
# After run_all.py --skip-download, diff generated tables vs paper claims:
python tests/run_time_split_tests.py   # cross-sectional validation smoke test
# Spot-check key numbers:
# - reports/tables/factor_vif.csv          → max VIF ≈ 2.80
# - reports/tables/benchmark_summary.csv    → check header "# Cross-sectional momentum proxy"
# - reports/tables/predictive_validation_ic.csv → IC values; confirm market_score clean vs raw
# - data/processed/cleaning_metadata.json   → INR/USD = 83.0
```

See `docs/PAPER_CODE_RECONCILIATION.md` for line-by-line paper ↔ code ↔ output mapping.

## 11. What Cannot Be Reproduced Exactly

- **Historical survivorship-free universe:** current constituent lists; no delisted-firm recovery without survivorship-free database (CRSP/Compustat). Disclosed as limitation.  
- **Live Yahoo prices:** will drift if re-downloaded later; use committed `data/raw/` for exact replication.  
- **FX path:** fixed rate 83.0 vs realized ±5–8% annual variation; sensitivity noted in `24_geographic_robustness.py`.

## 12. CI / Headless

```bash
CI=true pytest -q                      # no interactive prompts
python scripts/03_build_index.py --help  # (scripts accept no flags; idempotent)
```

## 13. Demo Reproducibility

`app.py` (local) and `demo/app.py` (HF Spaces) both read **the same** `data/processed/indexed_data.csv` generated by `03_build_index.py`. No hardcoded demo numbers — every chart/table traces to `reports/tables/*.csv` or the indexed dataset. Version/date shown in demo footer from `data/processed/cleaning_metadata.json`.

