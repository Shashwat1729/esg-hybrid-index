# Robustness & Validation Summary

## Multi-Factor ESG Index Methodology Assessment

**Analyst**: ResultsAnalyst-5 (Robustness & Validation)
**Date**: 2026-03-25
**Universe**: 60 companies (40 US, 20 India) across 10 sectors
**Factors**: 8 scoring dimensions (ESG_composite, financial_score, market_score, operational_score, risk_adjusted_score, value_score, growth_score, stability_score)

---

> **CRITICAL DATA LIMITATION — SYNTHETIC ESG SCORES**
>
> All ESG data in this study is **synthetically generated** (see `scripts/01_download_data.py`,
> `generate_synthetic_esg()`). The synthetic ESG scores incorporate a **built-in correlation
> weight (`corr_weight=0.35`)** that blends real financial quality signals (ROA, profit margins,
> market cap) into the ESG base score. This means:
>
> 1. **Any ESG-financial regression recovers the construction parameter (~r=0.35)**, not a
>    genuine empirical relationship. Such results constitute **calibration validation**, not
>    empirical evidence of an ESG-performance link.
> 2. The correlation magnitude (~0.35) was chosen to match the meta-analysis of Friede, Busch
>    & Bassen (2015), but the **causal direction is reversed**: financials are used to GENERATE
>    ESG, rather than ESG PREDICTING financials.
> 3. **Only analyses using real ESG provider data** (Refinitiv, MSCI, S&P Global) can make
>    genuine empirical claims about ESG-financial relationships.
>
> Sections affected by this limitation are marked with **[SYNTHETIC CAVEAT]**.

---

## 1. Factor Validity (Information Coefficient Analysis) **[SYNTHETIC CAVEAT]**

> *Note: ESG_composite IC values reflect synthetic data calibration (corr_weight=0.35).
> The near-zero IC (0.02) for ESG_composite may partly reflect that the built-in correlation
> targets financial quality, not forward returns. With real ESG data, these results could differ.*

### 1.1 Full-Sample IC Results (benchmark_factor_validity.csv)

| Factor | IC (Spearman) | p-value | Significant? | Monotonic? | Q5-Q1 Spread |
|--------|--------------|---------|-------------|-----------|---------------|
| market_score | **0.4688** | **0.00016** | **Yes** | **Yes (rho=1.0)** | **+22.99** |
| sector_position | 0.2380 | 0.0671 | Yes* | No | +13.70 |
| growth_score | 0.1768 | 0.1766 | No | Partial | +3.45 |
| stability_score | 0.0780 | 0.5534 | No | Partial | +9.93 |
| ESG_composite | 0.0197 | 0.8815 | No | No | -3.10 |
| financial_score | -0.0178 | 0.8929 | No | No | +1.46 |
| operational_score | -0.0099 | 0.9402 | No | No | +3.76 |
| risk_adjusted_score | -0.0272 | 0.8368 | No | No | +3.55 |
| value_score | -0.0408 | 0.7572 | No | No | -6.46 |
| similarity_rank | -0.1097 | 0.4041 | No | Yes (rho=-0.9) | -7.86 |

**Key Finding**: Only **market_score** demonstrates statistically significant predictive power (IC = 0.469, p < 0.001) with perfect monotonicity across quintiles. This is the single factor with robust return-predictive validity. All other individual factors, including ESG_composite and financial_score, fail to achieve statistical significance at the 5% level.

*Note: sector_position is marginal at p = 0.067 (10% level).

### 1.2 Multi-Horizon Predictive IC (predictive_validation_ic.csv)

Out of 33 factor-horizon tests (11 factors x 3 horizons):
- **1 test significant at p < 0.05**: stability_score at 6m horizon (IC = -0.343, p = 0.015) -- but in the *negative* direction
- **0 tests significant after adjusting for multiple comparisons**
- The strongest near-significant result: market_score at 3m (IC = 0.271, p = 0.057)

**Critical Assessment**: The multi-horizon out-of-sample IC analysis is devastating for individual factor predictive claims. No factor reliably predicts forward returns across horizons.

### 1.3 Multiple Testing Correction

No `multiple_testing_correction.csv` file exists. Manual assessment:

- Total tests conducted across factor validity analyses: ~33 factor-horizon combinations
- At alpha = 0.05 with 33 tests, Bonferroni threshold = 0.05/33 = 0.0015
- The single significant result (stability_score 6m, p = 0.015) does **not** survive Bonferroni correction
- Under Benjamini-Hochberg (BH) at FDR = 0.05: ordering the 33 p-values, the smallest (p = 0.015) would need p < 0.05 * (1/33) = 0.0015 to survive -- it does not
- **Zero tests survive any multiple testing correction**

For the full-sample analysis, market_score (p = 0.00016) with 10 tests would survive both Bonferroni (threshold = 0.005) and BH correction.

**Summary**: 1 of 10 factors (market_score) survives multiple testing correction in the full-sample analysis. Zero of 33 multi-horizon tests survive.

---

## 2. Bootstrap Stability Analysis

### 2.1 Rank Bootstrap (benchmark_bootstrap_results.csv)

| Metric | Value |
|--------|-------|
| Mean bootstrap std of ranks | 1.13 positions |
| Mean 95% CI width | 4.1 positions |
| Top 20 stability (probability of remaining in top 20) | **98.4% +/- 2.6%** |
| Companies with 100% top-20 probability | 14 of 20 (ranks 1-14) |
| Lowest top-20 probability in top 20 | MSFT: 74.7% (rank 17) |

**Rank stability detail (advanced_bootstrap_ci.csv)**:
- Top 5 companies (GOOGL, RELIANCE.NS, TCS.NS, BHARTIARTL.NS, META): CI width of 1-2 positions -- extremely stable
- Mid-range companies (ranks 30-50): CI widths of 7-14 positions -- substantial uncertainty
- Only 1 company flagged as rank-unstable: HDFCBANK.NS (std = 3.79, range = 19)

### 2.2 Perturbation-Based Rank Stability (rank_stability.csv)

- 59 of 60 companies classified as "rank_stable" (98.3%)
- Only outlier: HDFCBANK.NS (rank 37, std = 3.79, range = 19 positions)
- Top 5 companies show zero rank variation under perturbation

**Assessment**: Rankings are highly stable under bootstrap resampling. The top and bottom of the distribution are locked in; the middle is more fluid, which is expected and methodologically acceptable.

---

## 3. Weight Sensitivity Analysis

### 3.1 Profile Weight Sensitivity (profile_weight_sensitivity.csv)

| Profile | Perturbation | Mean Spearman rho | Mean Top-20 Overlap | Min Top-20 |
|---------|-------------|-------------------|---------------------|-----------|
| ESG-First | +/- 5% | 0.9991 | 20.0/20 | 20 |
| ESG-First | +/- 10% | 0.9980 | 19.8/20 | 19 |
| ESG-First | +/- 20% | 0.9949 | 19.3/20 | 17 |
| Balanced | +/- 5% | 0.9994 | 19.7/20 | 19 |
| Balanced | +/- 10% | 0.9988 | 19.6/20 | 19 |
| Balanced | +/- 20% | 0.9962 | 19.4/20 | 18 |
| Financial-First | +/- 5% | 0.9995 | 19.9/20 | 19 |
| Financial-First | +/- 10% | 0.9988 | 19.7/20 | 19 |
| Financial-First | +/- 20% | 0.9969 | 19.5/20 | 19 |

**Assessment**: Rankings are remarkably insensitive to weight perturbations. Even 20% random perturbation to all weights simultaneously maintains Spearman rho > 0.99 and top-20 overlap of 17-19 out of 20. This is a strong robustness result.

### 3.2 Single-Factor Weight Sensitivity (weight_sensitivity_single.csv)

All single-factor weight shifts of +/-10% produce:
- Spearman rho > 0.97 in all cases
- Top-20 overlap of 18-20 in all cases
- All p-values < 1e-24 (rank correlations are always highly significant)

Most sensitive factor: **similarity_rank** (rho drops to 0.944 at +10% shift, overlap = 18)
Least sensitive factor: **operational_score** (rho = 0.986 at +10%, overlap = 10/10)

### 3.3 Grid Search (weight_grid_search.csv)

Across 100 weight configurations:
- Kendall tau vs. base ranges from 0.17 to 0.89
- Top-10 overlap: 8-10 out of 10 consistently
- Cross-sectional IR: 0.31-0.55 (all configurations produce positive selection quality ratios)
- The methodology is robust: even substantially different weight vectors identify similar top companies

---

## 4. Cross-Provider Validation **[SYNTHETIC CAVEAT]**

> *Note: Our ESG scores are synthetic, so low cross-provider correlations are expected.
> These correlations measure agreement between our synthetic methodology and real provider
> scores, NOT inter-provider disagreement. Low correlations here are a limitation of the
> synthetic approach, not evidence of provider disagreement per se.*

### 4.1 Provider Correlations (benchmark_provider_correlations.csv)

| Provider | N Overlap | Spearman rho | p-value | Pearson r | p-value |
|----------|-----------|-------------|---------|-----------|---------|
| MSCI (numeric) | 14 | 0.084 | 0.776 | 0.180 | 0.537 |
| S&P Global | 14 | 0.402 | 0.154 | 0.442 | 0.114 |
| Sustainalytics (inverted) | 14 | 0.354 | 0.215 | 0.539 | **0.047** |

**Critical Assessment**: This is a significant weakness.
- None of the rank correlations (Spearman) are statistically significant at p < 0.05
- Only Sustainalytics shows a marginally significant Pearson correlation (r = 0.539, p = 0.047)
- The MSCI correlation is effectively zero (rho = 0.08)
- S&P Global shows moderate but non-significant correlation (rho = 0.40)

**Context**: The small overlap sample (N=14) severely limits statistical power. However, the low MSCI correlation is concerning and will face reviewer scrutiny. The ESG research literature widely documents provider disagreement (Berg et al., 2022, "Aggregate Confusion"), which provides partial justification.

---

## 5. PCA Analysis

### 5.1 Variance Explained (advanced_pca_variance.csv)

| Component | Eigenvalue | Variance Explained | Cumulative |
|-----------|-----------|-------------------|-----------|
| PC1 | 1.962 | **24.1%** | 24.1% |
| PC2 | 1.735 | 21.3% | 45.4% |
| PC3 | 1.418 | 17.4% | 62.9% |
| PC4 | 1.095 | 13.5% | 76.3% |
| PC5 | 0.742 | 9.1% | 85.5% |
| PC6 | 0.660 | 8.1% | 93.6% |
| PC7 | 0.367 | 4.5% | 98.1% |
| PC8 | 0.156 | 1.9% | 100% |

**Key Finding**: The first 4 components (eigenvalue > 1.0, Kaiser criterion) explain **76.3%** of variance. No single component dominates (PC1 = 24.1%), indicating the 8 factors capture genuinely multi-dimensional information. This is a strong validation of the multi-factor approach -- a unidimensional ESG score would show PC1 >> 50%.

### 5.2 Factor Loadings (advanced_pca_loadings.csv)

| Factor | PC1 | PC2 | PC3 | PC4 |
|--------|------|------|------|------|
| financial_score | **0.612** | 0.273 | 0.005 | -0.178 |
| growth_score | **0.583** | 0.222 | 0.222 | 0.105 |
| ESG_composite | 0.374 | -0.115 | -0.010 | **0.575** |
| market_score | -0.170 | **0.651** | 0.166 | 0.155 |
| risk_adjusted_score | -0.280 | **0.542** | 0.305 | 0.050 |
| value_score | 0.164 | -0.157 | **0.421** | **-0.617** |
| operational_score | 0.082 | 0.173 | **-0.642** | 0.041 |
| stability_score | 0.064 | 0.301 | **-0.491** | -0.466 |

**Interpretation**:
- **PC1** (24.1%): "Financial-Growth Quality" -- driven by financial_score and growth_score
- **PC2** (21.3%): "Market-Risk Performance" -- driven by market_score and risk_adjusted_score
- **PC3** (17.4%): "Operational vs. Value/Stability Trade-off" -- contrasts operational & stability against value
- **PC4** (13.5%): "ESG vs. Value Orientation" -- contrasts ESG_composite against value_score

The PCA confirms that financial and ESG dimensions load on different components, validating the multi-factor integration rationale.

---

## 6. Multicollinearity Assessment

### 6.1 VIF Analysis (vif_multicollinearity.csv)

| Factor | VIF | Assessment |
|--------|-----|-----------|
| ESG_composite | 1.16 | Low |
| financial_score | 3.03 | Low |
| market_score | 2.08 | Low |
| operational_score | 1.26 | Low |
| risk_adjusted_score | 1.94 | Low |
| value_score | 1.20 | Low |
| growth_score | 2.93 | Low |
| stability_score | 1.34 | Low |

**All VIF values < 5** (conservative threshold). Maximum VIF = 3.03 (financial_score), well below concern levels. The factors contribute distinct information to the composite index.

---

## 7. Factor Ablation Analysis (advanced_factor_ablation.csv)

| Removed Factor | Spearman rho | Top-10 Overlap | Mean Rank Shift | Max Rank Shift |
|---------------|-------------|---------------|----------------|----------------|
| financial_score | 0.916 | 7/10 | 5.57 | 22 |
| market_score | 0.940 | 8/10 | 4.33 | 21 |
| ESG_composite | 0.964 | 8/10 | 3.37 | 13 |
| growth_score | 0.967 | 9/10 | 3.17 | 15 |
| value_score | 0.972 | 9/10 | 2.60 | 19 |
| stability_score | 0.981 | 9/10 | 2.40 | 11 |
| risk_adjusted_score | 0.982 | 9/10 | 2.30 | 10 |
| operational_score | 0.986 | 8/10 | 2.13 | 8 |

**Assessment**: Removing financial_score has the largest impact (rho = 0.916, mean shift = 5.57), followed by market_score (rho = 0.940). ESG_composite removal has moderate impact (rho = 0.964). Operational_score has the least impact -- its removal barely changes rankings. This suggests the index is most structurally dependent on financial and market factors.

---

## 8. Normality & Distributional Properties

### 8.1 Normality Tests (normality_tests.csv)

| Variable | Shapiro p | Normal (Shapiro)? | Normal (JB)? |
|----------|----------|-------------------|--------------|
| ESG_composite | 0.187 | Yes | Yes |
| E_score | 0.469 | Yes | Yes |
| S_score | 0.274 | Yes | Yes |
| G_score | 0.890 | Yes | Yes |
| financial_score | **0.0002** | **No** | **No** |
| market_score | **0.006** | **No** | **No** |
| operational_score | 0.055 | Yes (marginal) | Yes |
| risk_adjusted_score | 0.669 | Yes | Yes |
| value_score | **2.4e-06** | **No** | **No** |
| growth_score | 0.018 | **No** | Yes |
| stability_score | **2.5e-07** | **No** | **No** |

**Assessment**: Several factors (financial_score, market_score, value_score, stability_score) violate normality, justifying the use of non-parametric methods (Spearman correlations, bootstrap, Kruskal-Wallis) throughout the analysis. The use of rank-based normalization in index construction mitigates these concerns.

---

## 9. Profile Comparisons & Rank Reversals

### 9.1 Profile Statistical Tests (profile_statistical_tests.csv)

| Comparison | Cohen's d | Interpretation | Max Rank Diff |
|-----------|----------|---------------|---------------|
| ESG-First vs. Balanced | 0.278 | Small | 18 |
| ESG-First vs. Financial-First | 0.486 | Small | 20 |
| Balanced vs. Financial-First | 0.923 | **Large** | 8 |

The Balanced and Financial-First profiles are most similar (Spearman rho = 0.988), while ESG-First diverges more substantially. The maximum rank reversal between ESG-First and Financial-First is 20 positions (TSLA), demonstrating that weighting philosophy meaningfully impacts stock selection.

### 9.2 Friedman Test

Friedman statistic = 2.04, p = 0.564. **Not significant** -- the four factor scores do not produce statistically different median ranks, suggesting the composite aggregation smooths out individual factor differences.

---

## 10. Portfolio Performance Validation **[SYNTHETIC CAVEAT for ESG-Only strategy]**

> *Note: The ESG-Only portfolio strategy uses synthetic ESG scores with a built-in
> financial correlation. Its performance partly reflects this construction, not a pure
> ESG signal. The multi-factor balanced and financial-only strategies are less affected.*

### 10.1 Benchmark Comparisons (benchmark_weighting_methods.csv)

| Strategy | N | Return (6m) | Std | CS-IR† |
|----------|---|------------|-----|--------|
| Preference-Weight Top 20 (Ours) | 20 | **10.49%** | 16.57 | **0.633** |
| Score-Weight Top 20 (Ours) | 20 | 10.26% | 16.76 | 0.612 |
| Value-Weight Full Universe | 60 | 7.93% | 12.59 | 0.630 |
| Equal-Weight Full Universe | 60 | 7.34% | 14.79 | 0.496 |
| Random Top 20 (50 draws avg) | 20 | 7.41% | 15.14 | 0.523 |

†CS-IR = cross-sectional information ratio (mean momentum / std momentum across N stocks). This is NOT a time-series Sharpe ratio.

Our Top-20 portfolios outperform random selection by ~3% return and ~0.11 CS-IR points. The outperformance vs. equal-weight universe is ~2.9% with similar improvement.

### 10.2 Excess Momentum / Beta Analysis (benchmark_alpha_beta.csv)

| Strategy | Excess Avg Momentum† | Beta | CS-IR | Info Ratio |
|----------|-------|------|--------|-----------|
| Our Multi-Factor Top 20 | **2.89** | 1.004 | 0.612 | 0.174 |
| ESG-Only Top 20 | 1.95 | 1.051 | 0.660 | 0.158 |
| Financial-Only Top 20 | 0.70 | 1.054 | 0.481 | 0.063 |
| Growth-Only Top 20 | -0.15 | 1.117 | 0.527 | 0.046 |

†"Excess avg momentum" = portfolio mean momentum - beta * universe mean momentum. This is a single-point cross-sectional estimate, NOT a CAPM regression intercept (Jensen's alpha).

Our multi-factor index generates the **highest excess average momentum** (2.89) with a **beta near 1.0** (market-neutral exposure). However, ESG-Only achieves a higher CS-IR (0.66 vs. 0.61) in this sample period.

### 10.3 Regime Analysis (advanced_regime_analysis.csv)

| Regime | Our Balanced | ESG-Only | Financial-Only | Growth-Only |
|--------|-------------|----------|---------------|-------------|
| Bull (N=37) | 15.91% | 15.67% | 14.82% | 13.94% |
| Bear (N=23) | -4.62% | -5.22% | -5.41% | -4.42% |
| Bull excess | +0.80 | +0.56 | -0.29 | -1.16 |
| Bear excess | +0.55 | -0.05 | -0.24 | +0.75 |

Our balanced strategy shows **positive excess returns in both bull and bear regimes**, with bear downside capture of 0.94 (loses less than the market in downturns). This is the strongest regime-consistency result.

---

## 11. Cross-Validation (advanced_cv_weights.csv)

| Fold | Train CS-IR | Test CS-IR | Overfit Ratio | Market_score Weight |
|------|------------|------------|--------------|-------------------|
| 1 | 1.41 | 0.66 | 2.14 | 0.793 |
| 2 | 1.25 | 1.41 | 0.89 | 0.802 |
| 3 | 1.74 | 1.14 | 1.53 | 0.550 |
| 4 | 1.82 | 0.70 | 2.59 | 0.509 |
| 5 | 1.39 | 1.27 | 1.09 | 0.688 |

**Key Findings**:
- Mean overfit ratio = 1.65 (moderate overfitting, but test CS-IRs remain positive)
- Market_score consistently receives the **highest weight** across all folds (0.51-0.80), confirming it as the dominant return-predictive factor
- All test folds produce positive CS-IR values (0.66-1.41), indicating out-of-sample validity
- ESG_composite weight varies widely (0.03-0.31), suggesting it is not reliably return-predictive

---

## 12. Additional Robustness Checks

### 12.1 Sector Effects (sector_anova.csv)

Only 2 of 14 variables show significant sector differences:
- risk_adjusted_score (F = 2.53, p = 0.021)
- stability_score (F = 11.47, p < 0.001, eta-squared = 0.647)

Most scoring factors do NOT show significant sector bias, suggesting the index provides fair cross-sector comparisons for most dimensions.

### 12.2 US vs. India Comparison (benchmark_us_vs_india.csv)

| Metric | US (N=40) | India (N=20) |
|--------|----------|-------------|
| Mean ESG_composite | 49.62 | 50.77 |
| Mean financial_score | 48.61 | 52.87 |
| Mean market_score | 49.94 | 50.11 |
| Mean growth_score | 47.06 | 55.55 |

Indian companies score higher on financial and growth metrics; ESG scores are comparable. The balanced profile mean is similar (51.13 vs. 53.07), suggesting no extreme country bias.

### 12.3 Gini Inequality (advanced_gini_inequality.csv)

All score distributions show "Moderate" inequality (Gini = 0.15-0.37). No score is excessively concentrated (which would indicate poor discrimination) or too dispersed (which would indicate noise).

### 12.4 Time-Split Holdout (timesplit_evaluation_synthetic_summary.csv)

- Training: 2014-2021, Holdout: 2022-2023
- Holdout average return: 0.12% (near zero)
- Holdout CS-IR: 0.028 (essentially zero)
- Percent positive: 50%

**This is a weak result**: the index shows no predictive power in the out-of-sample holdout period. However, N=40 observations and a single 2-year holdout window make this test low-powered.

---

## 13. Overall Methodology Assessment

### 13.1 Strengths

1. **Rank Stability**: Bootstrap and perturbation analyses demonstrate exceptional rank stability (top-20 probability = 98.4%, weight sensitivity Spearman > 0.99 under 20% perturbation). This is the strongest robustness result.

2. **Multi-Dimensionality Validated**: PCA shows 4 components needed to explain 76.3% of variance with no single dominant factor. VIF values all < 3.1. The multi-factor approach captures genuinely distinct dimensions.

3. **Low Multicollinearity**: All VIF < 3.03, confirming factors contribute independent information.

4. **Consistent Portfolio Outperformance**: The Top-20 portfolios outperform equal-weight universe and random selection on both return (by ~3%) and cross-sectional IR basis (CS-IR improvement of ~0.11). Excess avg momentum is positive (2.89) with near-unit beta.

5. **Regime Robustness**: Positive excess returns in both bull and bear markets, with good downside protection (bear excess = +0.55%).

6. **Profile Robustness**: All three weighting profiles produce highly correlated rankings (rho > 0.89), and weight perturbation sensitivity is minimal.

7. **Appropriate Use of Non-Parametric Methods**: Given normality violations in key factors, the reliance on Spearman correlations, bootstrap methods, and Kruskal-Wallis tests is methodologically sound.

### 13.2 Weaknesses & Limitations

1. **Synthetic ESG Data with Built-In Financial Correlation**: ESG scores are synthetically generated with `corr_weight=0.35` (see `01_download_data.py`). Any ESG-financial regression recovers this construction parameter, not a genuine empirical relationship. All ESG-financial claims in this study are calibration validation, not empirical evidence. This is the most fundamental limitation of the study.

2. **Individual Factor Predictive Validity is Weak**: Only market_score shows significant IC (0.469, p < 0.001). The composite ESG_composite has IC = 0.02 (essentially zero). This undermines claims that the ESG signal itself drives portfolio selection quality. [Note: the near-zero ESG IC may partly reflect that synthetic ESG targets financial quality, not forward returns.]

3. **Multi-Horizon IC is Uniformly Non-Significant**: Zero out of 33 factor-horizon tests survive multiple testing correction. This is highly damaging for predictive validity claims.

4. **Cross-Provider Correlations are Low**: MSCI rho = 0.08 (effectively uncorrelated), S&P rho = 0.40 (non-significant). Only Sustainalytics shows marginal significance. While the small N=14 overlap explains some of this, a reviewer will challenge construct validity. [Note: low correlations are partly expected since our scores are synthetic, not from a real provider.]

5. **Time-Split Holdout Shows No Predictive Power**: Holdout CS-IR = 0.028, essentially random. Although the test is underpowered, this is unfavorable.

6. **Small Sample Size**: N=60 is small for robust statistical inference. Many tests are underpowered, confidence intervals are wide, and results may not generalize.

7. **Cross-Validation Shows Overfitting**: Mean overfit ratio = 1.65 indicates the in-sample model overstates out-of-sample performance by ~65%.

8. **Market_score Dominance**: CV consistently assigns 50-80% weight to market_score, suggesting the return-predictive component is driven by price momentum, not ESG or fundamental quality. This raises questions about whether the "multi-factor ESG" framing is accurate.

9. **Single Time Period**: All analysis uses one cross-sectional snapshot. No evidence of temporal stability or persistence of rankings over time.

### 13.3 Anticipated Reviewer Challenges

| Challenge | Evidence to Counter | Residual Risk |
|-----------|-------------------|---------------|
| "ESG data is synthetic, not real" | Calibrated to Friede et al. (2015) meta-analysis; sector profiles from MSCI/S&P CSA; methodology valid even if ESG-financial claims require real data | **High** -- fundamentally limits ESG-financial claims |
| "ESG factors don't predict returns" | Composite integration adds value; ESG is for risk management not alpha; ESG-financial results are calibration validation | **High** -- IC data and synthetic nature both support this critique |
| "Why not just use market_score?" | Multi-dimensionality validated by PCA; ablation shows financial_score also matters | **Medium** -- ablation analysis helps |
| "Provider correlations too low" | N=14 is underpowered; our scores are synthetic (low correlation expected); Berg et al. (2022) documents provider disagreement | **Medium** -- synthetic data explains much of this |
| "Overfitting concerns" | CV shows positive test CS-IRs; weight sensitivity shows stability | **Medium** -- overfit ratio of 1.65 is moderate |
| "N=60 too small" | Effect sizes are large where significant; bootstrap provides robust inference | **High** -- fundamentally limits generalizability |
| "No out-of-sample temporal validation" | Time-split attempted; portfolio outperforms random | **High** -- holdout CS-IR = 0.028 |
| "Rankings are just financial quality" | PCA shows 4 distinct dimensions; ESG loads independently on PC4 | **Low** -- PCA evidence is convincing |

### 13.4 Summary Statistics for Paper

| Robustness Test | Result | Verdict |
|----------------|--------|---------|
| **Synthetic ESG data** | **corr_weight=0.35 built-in** | **Critical limitation** |
| Factor IC significance | 1/10 factors significant | Mixed |
| Multi-horizon IC | 0/33 survive correction | Weak |
| Bootstrap rank stability | 98.4% top-20 stability | **Strong** |
| Weight sensitivity (20% perturb) | Spearman > 0.99 | **Strong** |
| Cross-provider correlation | rho = 0.08-0.40 (non-sig) | Weak |
| VIF multicollinearity | All < 3.03 | **Strong** |
| PCA dimensionality | 4 PCs explain 76.3% | **Strong** |
| Factor ablation | financial & market most impactful | Informative |
| Portfolio excess avg momentum | 2.89 (positive, near-unit beta) | **Strong** |
| Regime robustness | Positive excess in bull & bear | **Strong** |
| Time-split holdout | CS-IR = 0.028 | Weak |
| Cross-validation | Test CS-IRs positive, overfit 1.65x | Moderate |
| Normality | 4/8 factors non-normal | Handled (non-parametric) |
| Sector bias | 2/14 variables show sector effects | Acceptable |

### 13.5 Recommended Narrative for Paper

The multi-factor ESG index demonstrates **structural robustness** (rank stability, weight insensitivity, low multicollinearity, validated multi-dimensionality) but **limited individual factor predictive validity** (only market_score is significant) and relies on **synthetic ESG data with a built-in financial correlation**. The value proposition should be framed as:

1. **A stock selection framework** rather than a return prediction model
2. **An integrative methodology** that balances ESG and financial quality dimensions
3. **A stable, reproducible ranking system** that is insensitive to reasonable weight perturbation
4. The ESG dimension provides **risk characterization and values alignment** rather than direct alpha generation
5. **ESG-financial regression results validate the synthetic data calibration** (consistent with Friede et al., 2015 meta-analysis range of r=0.15-0.35), but do NOT constitute empirical evidence of an ESG-performance link

The paper should **explicitly acknowledge** that:
- ESG data is synthetically generated with `corr_weight=0.35`
- ESG-financial association results are calibration validation, not empirical discovery
- Replication with real ESG provider data is needed for empirical ESG-financial claims

The PCA evidence (4 genuinely distinct dimensions), rank stability analysis, and the regime analysis (consistent outperformance) are the strongest empirical supports that do NOT depend on the synthetic ESG limitation.

---

## Appendix: Key Numbers for Direct Citation

- **Market_score IC**: 0.469 (p = 0.00016), only significant factor
- **Bootstrap top-20 stability**: 98.4% +/- 2.6%
- **Mean bootstrap rank std**: 1.13 positions
- **Weight sensitivity at 20% perturbation**: Spearman rho = 0.995 (average across profiles)
- **PCA**: 4 components explain 76.3% of variance; no single PC > 24.1%
- **VIF range**: 1.16-3.03 (all low)
- **Portfolio excess avg momentum**: 2.89 (6-month price momentum as return proxy)
- **Portfolio CS-IR**: 0.633 (preference-weighted top 20)
- **Bear market excess return**: +0.55%
- **Cross-validation overfit ratio**: 1.65 (mean)
- **Provider correlations**: Spearman rho = 0.08 (MSCI), 0.40 (S&P), 0.35 (Sustainalytics)
- **Factor ablation**: removing financial_score shifts mean rank by 5.57; removing ESG_composite shifts by 3.37
- **N surviving multiple testing correction (full sample)**: 1 of 10 (market_score)
- **N surviving multiple testing correction (multi-horizon)**: 0 of 33
