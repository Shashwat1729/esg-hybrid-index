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
| `tab_top_ranks.tex` (Table IX) | Top-N rank intervals | `advanced_bootstrap_ci.csv`, `oos_firm_outcomes.csv` |
| `tab_sasb.tex` (Table XI) | SASB pillar weights | `config/sasb_materiality.yaml` |
| `tab_ext_controls.tex` | ESG vs realised risk with trailing / full controls | `ext_risk_controls.csv` (step 27) |
| `tab_ext_robust.tex` | H2/H4 on subsamples; recent-volatility control | `ext_subsample_robustness.csv`, `ext_recent_vol_control.csv` |
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
