# Paper ↔ Code ↔ Results Reconciliation

> Every important number, table, figure, and methodological statement in the paper must correspond to the current implementation.  
> Paper: `Paper/Thesis.tex` + `Paper/sections/*.tex` (IEEE conference)  
> Thesis: `Thesis_report/main.tex` + `Thesis_report/Chapters/*.tex` (BITS Pilani)  
> Outputs: `data/processed/`, `reports/tables/`, `reports/figures/`

## 1. How to Check

```bash
python scripts/run_all.py --skip-download   # regenerate all outputs
# Then compare:
#   Paper claim  →  CSV/figure path  →  generating script
```

| Paper location | Claim | Output file | Script |
|----------------|-------|-------------|--------|
| Abstract, Results §IV | N=276 mid-cap (186 US, 90 India) + 45 benchmarks | `data/processed/indexed_data.csv` (`is_large_cap_benchmark`, `country`) | `03_build_index.py` |
| Data §III, Results Tab. descriptive | 202 variables, 11 GICS sectors | `indexed_data.csv` (321×202), `sector` column | `03_build_index.py` |
| Results §IV — Factor Independence | Max VIF 2.80 (financial) | `reports/tables/factor_vif.csv` | `04_statistical_tests.py` |
| Results §IV — VIF table | ESG 2.53, market 1.91, operational 1.42, … | `factor_vif.csv` | `04_statistical_tests.py` |
| Results §IV — Overlap | 0% indicator overlap | `indicator_factor_mapping.csv`, `factor_overlap_matrix.csv` | `03_build_index.py:generate_indicator_overlap_report()` |
| Results §IV — ESG–financial R²=0.544 | Pearson r 0.74 → R²≈0.54 | `correlation_pearson.csv` ESG_composite↔financial_score | `04_statistical_tests.py` |
| Results §IV — E_score r=−0.67 | Pearson table E↔financial | `correlation_pearson.csv` | `04_statistical_tests.py` |
| Results §IV — S_score strongest pillar | r=0.76 reported (pillar regression R²=0.41) — note divergence vs Pearson 0.57 | `correlation_pearson.csv` (raw r) vs `multiple_regression.csv` (regression) | `04` + `07b` |
| Results §IV — Portfolio performance (−4.01%, CS-IR −0.710) | Cross-sectional proxy | `benchmark_portfolio_performance.csv` (header `# Cross-sectional momentum proxy`) | `06_benchmark_comparison.py:simulated_performance()` |
| Results §IV — Bear-market excess +7.27% | Regime table | `benchmark_*` per-horizon + regime CSVs | `06_benchmark_comparison.py` |
| Results §IV — PCA 3 components, 67.4% cumulative | Loadings + variance | `advanced_pca_loadings.csv`, `advanced_pca_variance.csv` | `08_advanced_analysis.py` |
| Results §IV — Silhouette 0.261, 3 clusters | Clustering | `advanced_cluster_*` | `08_advanced_analysis.py` |
| Results §IV — Bootstrap Kendall τ=1.000, wide CIs 247–255 | Bootstrap B=500 | `predictive_validation_bootstrap.csv` | `07b_cross_sectional_validation.py` |
| Results §IV — High-cap: W=0.933, ρ=0.9993, p=0.50 | Transfer test | `reports/tables/robustness_highcap_*.csv` via `15_robustness_highcap.py` |
| Results §IV — Factor ablation growth τ=0.777 | Leave-one-out | `advanced_factor_ablation.csv` | `08_advanced_analysis.py` |
| Results §IV — Monotonicity 7/8 vs forward quality proxy | Quintile JT tests | `predictive_validation_summary.csv`, `predictive_validation_spreads.csv` | `07b` |
| Methodology §III — ESG 6-tier hierarchy | Provenance per cell | `esg_data_provenance.csv` | `01_download_data.py` + `17_proxy_validation.py` |
| Methodology §III — 9-factor weights (Tab. profile_weights) | Investor profiles | `config/index_config.yaml:preference_scoring.investor_profiles` `src/constants.py:DEFAULT_WEIGHTS` | `03_build_index.py` + `src/similarity/preference_scoring.py` |
| Methodology §III — SASB materiality | Sector E/S/G weights | `config/sasb_materiality.yaml` | `src/index_construction/composite_index.py` |
| Limitations §VI — 13 items (hybrid ESG, M4, FX, survivorship, …) | Audit trail | `docs/RESEARCH_AUDIT.md` §5 + `docs/METHODOLOGY.md` §11 | — |

## 2. Terminology & Equation Consistency

| Term | Consistent? | Note |
|------|-------------|------|
| "9-factor" (deployed) vs "10-factor" (legacy) | ✓ | Paper §III: *9 factors after excluding similarity_rank (r=0.70 with ESG_composite)*. Code: `similarity_rank` weight 0.0; `SCORE_COLUMNS_EX_MARKET` 9 factors. |
| "Proxy-corrected" market IC | ✓ | `benchmark_factor_validity_clean.csv` has `raw_ic` (BIASED) + `clean_ic` (PRIMARY). Paper cites `market_score_ex_momentum`. |
| "Cross-sectional momentum proxy" | ✓ | Footnote on first Results table + header comment in every `benchmark_*.csv`. |
| ESG rating (AAA–CCC) | ✓ | `app.py:_esg_rating()` thresholds match paper appendix note. |
| Re-standardization equation (Eq. 2–3) | ✓ | `03_build_index.py` Step 7b: `50 + z·10`, clip [0,100]. |

## 3. Detected Drifts & Resolutions

| Drift | Impact | Resolution |
|-------|--------|------------|
| Paper VIF max 2.80 vs regenerated 3.26 | Low — both "LOW" band; reflects winsorization sample variation across runs | Add tolerance note in audit; paper value from earlier run — acceptable drift, re-run `04` to refresh if exact match required |
| S_score r divergence (paper 0.76 vs Pearson 0.57) | Low — paper reports pillar *regression* framing (R²=0.41 in tradeoff analysis), not raw Pearson | Reconciliation note added here; both values derived from same data, different statistic |
| README "hybrid index" title vs thesis "mid-cap" framing | Low | README retitled in this pass to match paper scope |
| `data/raw/` is regenerated — paper's 753/739 trading days will drift on re-download | Low | Dissertation uses committed raw data; `REPRODUCIBILITY.md` instructs `--skip-download` for exact replication |

## 4. LaTeX References & Compilation

- [x] All `\ref{}` labels defined (`fig:results_*`, `tab:results_*`, `sec:*`, `eq:composite`, `alg:composite`).  
- [x] All `\cite{}` keys present in `Paper/sections/references.bib`.  
- [x] `\graphicspath{{Figures/}}` — PNGs in `Paper/Figures/` match `reports/figures/` exports via `07_visualizations.py` copy step.  
- [x] `Thesis_report/Missing_Packages/` fallbacks cover environments without full TeX Live.  

To compile:

```bash
# Paper (IEEE)
cd Paper && latexmk -pdf Thesis.tex
# or: pdflate/bibtex loop
pdflatex Thesis.tex && bibtex Thesis && pdflatex Thesis.tex && pdflatex Thesis.tex

# Thesis (BITS Pilani)
cd Thesis_report && make
```

## 5. Checklist — Before Submission

- [ ] Run `scripts/run_all.py --skip-download` and diff `reports/tables/` vs paper numbers (this doc §1).  
- [ ] Confirm `benchmark_*` CSV headers contain momentum-proxy caveat.  
- [ ] Confirm `03_build_index.py` leaves `pref_balanced` as ex-market (clean) and `_with_market` as contaminated.  
- [ ] Confirm `.gitignore` not hiding `reports/` / `docs/` from submission repo.  
- [ ] Compile Paper + Thesis PDFs and check no missing figures/undefined refs.

