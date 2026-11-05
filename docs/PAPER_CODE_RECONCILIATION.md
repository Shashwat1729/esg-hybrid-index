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

## Final proofread (2026-11-04)

| Location | Was | Now |
|---|---|---|
| §V-A Redundancy | "The pillars are nearly unrelated" (E–S ρ = 0.42) | E and S moderately correlated; both nearly unrelated to G |
| §VI-C | "beyond trailing volatility and 8 other characteristics" | 8 characteristics *including* trailing volatility (`CONTROLS` in step 27 has 8 entries) |
| §VI-A | S pillar carries the "whole" forward risk signal | "the" forward risk signal (matches "mostly" in §V-C) |
| §V-C, G pillar bullet | "none with volatility" while partial ρ = −0.125 | Refers explicitly to the regression coefficient |
| Tables III, VI | OOS IC interval differs from Table V without explanation | Captions state 1,000 vs 5,000 resamples (`25_out_of_sample_evaluation.py`) |
| §V-E | "−7.0 pp below the universe" (double negative) | "relative to the universe is already −7.0 pp" |
| Style | Sentences opening with numerals; "high-release"; "only positive result"; "Dr." in IEEE author block | Reworded |

## Second final proofread (2026-11-04)

| Location | Was | Now |
|---|---|---|
| §V-C, survivorship bounds | "Both estimates are robust to the 9 M&A exits", although H3 = +1.9 pp while both bounds are negative | Says the *conclusions* are unchanged and explains why the excess turns negative: `portfolio_excess` drops exits and promotes the 20th firm, while the bounds keep CTRA (`25_out_of_sample_evaluation.py`) |
| Abstract | partial ρ = −0.163 "even with 8 risk controls" (−0.163 is the trailing-volatility-only H2 estimate) | "and the effect survives 8 risk controls and sector effects" |
| §III-A, ref. [39] | NIFTY Midcap 150 cited to the *NIFTY ESG* index methodology | Cited to the NIFTY Midcap 150 index page (URL checked 2026-11-04) |
| §V-A | "share 14 and 14 names, respectively" | "with the balanced list" |
| Fig. 4(a) | "deployed" legend overlapped the dashed line | Legend moved above the axes (`27_paper1_extensions.py`; no data changed) |

## Iterative proofread to a clean pass (2026-11-04)

Full rereads were repeated until one pass needed no edits. Only wording changed; no numbers, code or data changed.

| Pass | Location | Change |
|---|---|---|
| 1–2 | §III-A | Universe sentence now reads as 321 candidates: mid-cap constituents of the three indices, plus the 45 S&P 500 reference caps |
| 1–2 | §IV-D | H1–H3/H4 registration sentence split; the anchor-date controls of H2 and H4 are now stated explicitly |
| 1–2 | App. C, item 8 | Portfolio-size rule written as ⌈0.07·N_mid⌉ bounded to [10, 50] |
| 1–2 | §I | Spelling: "artifacts" → "artefacts" (British spelling, used throughout) |
| 1–2 | Abstract, §I, §II, §III, §VII | Acronyms defined at first use: ESG, IC, SASB, ISS, CSR, TRI, MAD, FCF, VIF, PCA, HC3, BRSR |
| 3 | App. B, Table XIII | Config path moved out of the small-caps caption and into the body text |
| 3 | §VI-B | "four artefacts" → "four validation pitfalls" (Section V-B also covers the direction errors, so "four artefacts" miscounted) |
| 4 | — | No changes |

## Code identifiers removed from the manuscript (2026-11-05)

The paper now refers to the code only through the repository link (withheld in
the anonymised build). File names, column names, configuration keys and
command-line flags were replaced by plain-language descriptions:

| Location | Before | After |
|---|---|---|
| Appendix A, Table XII | `scope1_emissions`, `board_size`, ... (21 column names) | Scope 1 emissions, Board size, ... |
| Appendix B | "weights are read from `config/sasb_materiality.yaml`" | removed |
| Appendix C, items 1, 3-5, 7 | column names, `monotonicity_correction: false`, `market_share`, "download script" | indicator names, "disabled in all reported results", "within-sector revenue share", "data-collection stage" |
| Sec. III-F, item 1 | `scope1_emissions`, `esg_risk_rating` | Scope 1 emissions and ESG risk-rating indicators |
| Sec. IV-A | `forward_quality_proxy` | "forward quality proxy" |
| Sec. IV-D | `(preregistration/window2)`, "evaluation script" | removed, "evaluation code" |
| Conclusion, Reproducibility | `run_all.py --skip-download`, Step 25 `--refresh`, Step 28 `evaluate` | one prose sentence; the repository documents each step |

No numbers changed. Both builds: 0 errors/overfull/undefined/bib warnings, 14 pages; anonymity check 0; 325 tests pass; registration hashes match.

## Named submission build (2026-11-05)

The venue is not double-blind. The anonymous option was removed from `Paper/Thesis.tex`; `Paper/Submission.tex` now builds the same named paper (three authors, affiliations, repository link https://github.com/Shashwat1729/esg-hybrid-index). The two PDFs have identical text; both build with 0 errors/overfull/undefined/bib warnings at 14 pages.
