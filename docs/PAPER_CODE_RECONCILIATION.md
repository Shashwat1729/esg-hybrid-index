# Paper ↔ Code Reconciliation

Since the 2026-09 audit, the manuscript contains **no hand-typed empirical
numbers**. `scripts/26_paper_artifacts.py` (the last step of
`scripts/run_all.py`) writes:

| Output | Content | Source |
|---|---|---|
| `Paper/generated/numbers.tex` | ~210 `\newcommand` macros used in the text (sample sizes, provenance shares, R², VIF, OOS estimates, CIs, p-values) | `data/processed/*`, `reports/tables/*`, `reports/pre_audit/*`, `config/*.yaml` |
| `tab_provenance.tex` (Table I) | ESG cell sources by pillar | `reports/tables/esg_data_provenance.csv` |
| `tab_weights.tex` (Table II) | Effective profile weights (market removed) | `config/index_config.yaml` |
| `tab_insample_vs_oos.tex` (Table III) | In-sample "ICs" vs OOS IC | `circularity_comparison.csv`, `oos_information_coefficients.csv` |
| `tab_pre_post.tex` (Table IV) | Pre- vs post-audit diagnostics | `reports/pre_audit/*` vs current tables |
| `tab_oos_primary.tex` (Table V) | H1–H3 | `oos_primary_hypotheses.csv` |
| `tab_oos_ic.tex` (Table VI) | OOS ICs by factor and horizon | `oos_information_coefficients.csv` |
| `tab_oos_portfolio.tex` (Table VII) | Top-N excess returns | `oos_portfolio_excess.csv` |
| `tab_oos_risk.tex` (Table VIII) | ESG and realised risk | `oos_esg_risk_filter.csv`, `oos_esg_risk_regression.csv` |
| `tab_ext_controls.tex` (Table IX) | ESG vs realised risk with trailing / full controls | `ext_risk_controls.csv` (step 27) |
| `tab_ext_robust.tex` (Table X) | H2/H4 on subsamples; recent-volatility control | `ext_subsample_robustness.csv`, `ext_recent_vol_control.csv` |
| `tab_top_ranks.tex` (Table XI) | Top-N rank intervals | `advanced_bootstrap_ci.csv`, `oos_firm_outcomes.csv` |
| `tab_sasb.tex` (Table XIII) | SASB pillar weights | `config/sasb_materiality.yaml` |
| `Figures/fig_overview.pdf` | Figure 1 (study design) | sample macros + `preregistration/window2/manifest.json` |
| `Figures/fig_oos_ic.pdf` | Figure 2 | `oos_information_coefficients.csv` |
| `Figures/fig_ext_mechanism.pdf` | Figure 3 (where the risk signal lives) | `ext_risk_controls.csv`, `ext_recent_vol_control.csv`, `dry_run_results.csv` |
| `Figures/fig_ext_weights.pdf` | Figure 4 | `ext_weight_uncertainty.csv`, `ext_esg_weight_sweep.csv` |
| `Figures/fig_rank_ci.pdf` | Figure 5 | `advanced_bootstrap_ci.csv` |
| Counts in text (`\NControls`, `\CostBpsUS`, `\CostBpsIndia`) | Sections IV–V | `27_paper1_extensions.py` constants, `ext_costs.csv` |
| Pre-registration macros (`\PreregAnchor`, `\DryHTwo`, …) | Section IV-E | `preregistration/window2/manifest.json`, `dry_run_results.csv` |

Qualitative statements that depend on the data (e.g. the sector composition of
the ESG top-19 in §V-C) were checked by hand against
`oos_firm_outcomes.csv` on 2026-09-24 and must be re-checked if the data are
refreshed.

## Pre-submission check (2026-10-15)

Every remaining hand-written statement in the manuscript was checked against
the code and data. Corrections:

| Paper statement (before) | Code / data | Fix |
|---|---|---|
| All indicators winsorised at 1/99 and robust-z scored | Cleaning winsorises by variable type (2.5/97.5 for valuation ratios); ESG, financial and market indicators use 1/99 plus robust z; operational, risk-adjusted, value, growth and stability indicators use within-sector percentiles with a Shapiro-Wilk switch to normal scores, and five factors are blended 70:30 with their sector percentiles (`03_build_index.py`) | §III-C rewritten |
| Source priority lists SEC above ISS | ISS QualityScores overwrite SEC items (`01_download_data.py`, tier 3) | Order swapped in §III-B |
| Top-N = ceil(7% × N) with N = 314 scored firms | 19 = ceil(0.07 × 269 mid-caps) | Formula uses N_mid |
| "All CIs from 5,000 resamples" | 5,000 for H1–H3; 1,000–2,000 for secondary analyses | Stated separately |
| ESG top-19 "overweights utilities and energy, underweights technology" | Largest overweight is real estate; no technology and no healthcare holdings, the two best sectors | Corrected in §V-C and §VI-C |
| "Roughly half of the shortfall is a sector bet" | −19.9 pp to −8.4 pp after neutralisation (58%) | "More than half" |
| Risk signal "comes from financial-proxy cells" | Proxy-cell sub-score partial ρ −0.086, CI includes 0; S pillar significant | Attributed to the proxy-built S pillar |
| Registration "before the window opens" | Package pushed to the public repo on 2026-09-25; window opened 2026-09-28 | Past tense; the push is named as the timestamp |
| "adding 45 firms" (literal) | Large-cap count macro | Uses the `NLarge` macro |

Reference check (same day, against publisher records):

| Entry | Was | Corrected to |
|---|---|---|
| `chatterji_2016` | *Management Science* | *Strategic Management Journal* 37(8):1597–1614, doi 10.1002/smj.2407 |
| `gibson_2021` | *Review of Finance*, 4 authors | *Financial Analysts Journal* 77(4):104–127, Gibson Brandon, Krueger and Schmidt |
| `damato_2022` | *Annals of Operations Research* | *Computational Management Science* 19(2):347–373, doi 10.1007/s10287-021-00419-3 |
| `alighanbari_2022` | MSCI report, not verifiable | Citation removed from §II |
| `yfinance`, `yahoo_finance`, `sec_edgar` | Undated access notes | Version / access period stated |

Also: double negative in the double-sort sentence (§V-D) reworded; Appendix B
given body text.
