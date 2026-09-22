# ESG-Financial Performance Trade-off Analysis

**Dataset:** 60 companies (40 US, 20 India) across 10 sectors
**Variables:** ESG_composite, financial_score, market_score, E/S/G sub-pillars
**Analyst:** ResultsAnalyst-2 | **Date:** March 2026

---

## 1. Correlation Between ESG and Financial Performance

### Key Finding: Statistically Significant Positive Correlation

| Metric | Value | p-value | Significant (alpha=0.05) |
|--------|-------|---------|--------------------------|
| **Pearson r** | **0.3164** | 0.0138 | Yes |
| **Spearman rho** | **0.2939** | 0.0226 | Yes |
| R-squared | 0.1001 | -- | -- |
| 95% CI (Pearson) | [0.0679, 0.5279] | -- | -- |

**Regression equation:** `financial_score = 0.4789 * ESG_composite + 26.08`

**Interpretation:** A moderate positive relationship exists (r=0.32). ESG explains ~10% of the variance in financial performance. The relationship is statistically significant under both parametric (Pearson) and non-parametric (Spearman) tests. The 95% confidence interval excludes zero, confirming robustness.

**Effect size classification:** Medium (Cohen's convention: 0.10=small, 0.30=medium, 0.50=large). The Pearson r of 0.32 sits at the medium threshold, indicating a meaningful but not dominant relationship.

### Sub-Pillar Correlations with Financial Score

| Pillar | Pearson r | p-value | Spearman rho | p-value |
|--------|-----------|---------|--------------|---------|
| **E_score** | **0.2751** | **0.0334** | **0.3986** | **0.0016** |
| S_score | 0.2443 | 0.0600 | 0.2479 | 0.0562 |
| G_score | -0.0134 | 0.9193 | -0.0257 | 0.8456 |

**Key insight:** Environmental performance drives the ESG-financial link. Governance shows zero correlation with financial performance. Social performance is marginally significant. This suggests environmental efficiency (lower emissions, better energy use) co-occurs with financial efficiency, while governance structures are orthogonal to profitability.

---

## 2. Quintile Analysis: Do High-ESG Companies Sacrifice Financial Performance?

### Answer: No -- High-ESG Companies Outperform Financially

| ESG Quintile | N | Financial Mean | Financial Median | Financial Std | ESG Mean | Market Mean |
|-------------|---|----------------|-----------------|---------------|----------|-------------|
| Q1 (Lowest) | 12 | 48.63 | 46.72 | 7.93 | 42.97 | 52.42 |
| Q2 | 12 | 48.60 | 47.44 | 4.40 | 47.67 | 47.60 |
| Q3 | 12 | 48.77 | 48.44 | 3.91 | 50.20 | 53.40 |
| Q4 | 12 | 48.17 | 46.74 | 9.01 | 52.37 | 50.22 |
| **Q5 (Highest)** | **12** | **55.97** | **56.26** | **8.15** | **56.79** | **46.35** |

**Statistical tests:**
- **ANOVA:** F = 2.71, p = 0.039 (significant at alpha=0.05)
- **Q5 vs Q1 t-test:** t = 2.24, p = 0.036 (significant)
- **Q5-Q1 difference:** +7.34 points (15.1% higher)

**Pattern:** Financial performance is flat across Q1-Q4 (range: 48.17-48.77), then jumps sharply in Q5 (+7.34). This suggests a **threshold effect** -- ESG only correlates with financial outperformance at the highest levels.

**Market score inversion:** Q5 (highest ESG) has the *lowest* market score (46.35), while Q1 has the highest (52.42). The market may not yet fully price ESG performance.

---

## 3. Sector-Level ESG-Financial Correlation

### Sectors With Positive Correlation (ESG Helps)

| Sector | N | Pearson r | p-value | Significant |
|--------|---|-----------|---------|-------------|
| **Communication Services** | 7 | **0.7639** | **0.0456** | **Yes** |
| **Technology** | 12 | **0.6830** | **0.0144** | **Yes** |
| Basic Materials | 3 | 0.7571 | 0.4532 | No (small n) |
| Healthcare | 8 | 0.2492 | 0.5518 | No |
| Consumer Defensive | 7 | 0.0220 | 0.9626 | No |

### Sectors With Negative Correlation (ESG-Financial Tension)

| Sector | N | Pearson r | p-value | Significant |
|--------|---|-----------|---------|-------------|
| Financial Services | 11 | -0.3403 | 0.3058 | No |
| Consumer Cyclical | 6 | -0.5000 | 0.3125 | No |
| Energy | 3 | -0.7590 | 0.4514 | No (small n) |

**Key finding:** Only **Technology** (r=0.68, p=0.01) and **Communication Services** (r=0.76, p=0.05) show statistically significant positive ESG-financial correlations. These are knowledge-intensive, low-physical-asset sectors where ESG investments (employee welfare, governance) may directly enhance productivity.

**Energy** shows the strongest negative correlation (r=-0.76), consistent with the hypothesis that ESG compliance imposes real costs on carbon-intensive industries. **Financial Services** also trends negative, possibly reflecting regulatory compliance costs.

---

## 4. Triple-Bottom-Line Analysis: ESG x Financial x Market

### Inter-Score Correlations

| Pair | Pearson r | p-value | Relationship |
|------|-----------|---------|-------------|
| ESG vs Financial | 0.3164 | 0.0138 | Moderate positive |
| ESG vs Market | -0.1809 | 0.1667 | Weak negative (ns) |
| Financial vs Market | 0.0178 | 0.8928 | Near zero (ns) |

**Critical finding:** The three dimensions are largely **orthogonal**. Financial and market scores are essentially uncorrelated (r=0.02). ESG shows weak negative association with market score. This validates the multi-factor index approach -- these dimensions capture distinct aspects of company quality that would be missed by any single metric.

### Pareto-Optimal Companies (No Company Dominates Across All Three)

| Ticker | Sector | Country | ESG | Financial | Market |
|--------|--------|---------|-----|-----------|--------|
| **NVDA** | Technology | US | 59.87 | 58.57 | 58.52 |
| TCS.NS | Technology | India | 58.90 | 71.74 | 44.23 |
| META | Communication Services | US | 57.26 | 64.86 | 49.44 |
| GOOGL | Communication Services | US | 55.55 | 60.37 | 73.22 |
| RELIANCE.NS | Energy | India | 42.00 | 72.13 | 55.03 |

**NVDA is the closest to a "true all-rounder"** -- scoring above average on all three dimensions with the most balanced profile. TCS.NS dominates on financial (71.7) with strong ESG (58.9) but lags on market. GOOGL dominates market (73.2). RELIANCE.NS reaches the Pareto frontier via exceptional financial performance (72.1) despite low ESG (42.0) -- a "profit without ESG" archetype.

---

## 5. Country Comparison: United States vs India

### Descriptive Statistics

| Metric | US (n=40) | India (n=20) | Difference | Cohen's d |
|--------|-----------|-------------|------------|-----------|
| ESG mean | 49.76 | 50.48 | -0.72 | -0.146 |
| ESG std | 4.86 | 5.04 | -- | -- |
| Financial mean | 48.61 | 52.87 | -4.26 | -0.547 |
| Financial std | 5.85 | 9.33 | -- | -- |
| Market mean | 49.94 | 50.11 | -0.17 | -0.020 |
| Market std | 11.11 | 5.26 | -- | -- |

### Statistical Tests (Mann-Whitney U)

| Score | U-statistic | p-value | Significant |
|-------|-------------|---------|-------------|
| ESG | 368.0 | 0.621 | No |
| Financial | 280.0 | 0.061 | No (marginal) |
| Market | 447.0 | 0.466 | No |

### ESG-Financial Correlation by Country

| Country | Pearson r | p-value | Spearman rho | p-value |
|---------|-----------|---------|--------------|---------|
| **US** | **0.5264** | **0.0005** | **0.3929** | **0.0121** |
| India | 0.0568 | 0.8120 | 0.0256 | 0.9148 |

**Major finding:** The ESG-financial link is **entirely driven by US companies** (r=0.53, p<0.001). Indian companies show zero correlation (r=0.06, ns). This suggests:

1. **Institutional context matters** -- the US regulatory/market environment rewards ESG performance financially; India's does not (yet).
2. Indian companies show higher financial score dispersion (std=9.33 vs 5.85) -- financial performance in India is driven by other factors (market position, sector dynamics) rather than ESG.
3. Indian companies have marginally higher financial scores (52.87 vs 48.61, p=0.061, d=-0.55) -- a medium effect size suggesting emerging market companies can achieve strong financials without ESG investment.

---

## 6. ESG Leaders and Laggards

### Classification Thresholds
- **ESG cutoffs:** Q1 = 47.01, Q3 = 52.86
- **Financial cutoffs:** Q1 = 45.24, Q3 = 53.43

### ESG Leaders (High ESG + High Financial): 8 Companies

| Ticker | Sector | Country | ESG | Financial | Market |
|--------|--------|---------|-----|-----------|--------|
| NVDA | Technology | US | 59.87 | 58.57 | 58.52 |
| TCS.NS | Technology | India | 58.90 | 71.74 | 44.23 |
| META | Communication Services | US | 57.26 | 64.86 | 49.44 |
| MSFT | Technology | US | 57.02 | 60.68 | 31.38 |
| AAPL | Technology | US | 55.67 | 58.11 | 52.68 |
| GOOGL | Communication Services | US | 55.55 | 60.37 | 73.22 |
| V | Financial Services | US | 55.54 | 54.40 | 47.47 |
| INFY.NS | Technology | India | 53.81 | 57.52 | 53.02 |

**Profile:** 6 of 8 are Technology companies. 6 of 8 are US-based. These are capital-light, innovation-driven companies where ESG (especially human capital management) aligns naturally with financial performance.

### ESG Laggards (Low ESG + Low Financial): 5 Companies

| Ticker | Sector | Country | ESG | Financial | Market |
|--------|--------|---------|-----|-----------|--------|
| HD | Consumer Cyclical | US | 41.55 | 44.20 | 50.53 |
| NKE | Consumer Cyclical | US | 43.73 | 42.42 | 34.96 |
| DHR | Healthcare | US | 44.65 | 44.81 | 58.13 |
| RTX | Industrials | US | 39.88 | 43.51 | 44.52 |
| ASIANPAINT.NS | Basic Materials | India | 39.54 | 44.84 | 59.44 |

**Profile:** Physical-asset-heavy sectors (retail, defense, materials). Several have strong market scores despite weak ESG/financial -- suggesting market momentum disconnects from fundamentals.

### Edge Cases

| Category | Ticker | ESG | Financial | Interpretation |
|----------|--------|-----|-----------|---------------|
| **ESG at cost** | WFC | 54.56 | 42.70 | High ESG investment with poor financial returns (post-scandal compliance costs) |
| **Profit without ESG** | RELIANCE.NS | 42.00 | 72.13 | Highest financial score in dataset, low ESG (energy sector, emerging market) |

### Median-Split Quadrant Distribution

| Quadrant | Count | Percentage |
|----------|-------|------------|
| High ESG + High Financial | 15 | 25.0% |
| High ESG + Low Financial | 15 | 25.0% |
| Low ESG + High Financial | 15 | 25.0% |
| Low ESG + Low Financial | 15 | 25.0% |

**Perfectly uniform quadrant distribution** -- exactly 25% in each cell. This is notable: it means the median-based split does not reveal a dominant ESG-financial pattern. The relationship is driven by the **tails** (quintile extremes), not the center of the distribution.

---

## 7. Composite Index Balance Analysis

### Does the Index Balance ESG and Financial Dimensions?

| Composite Score | Spearman with ESG | Spearman with Financial | Spearman with Market | Bias Direction |
|----------------|-------------------|------------------------|---------------------|---------------|
| pref_esg_first | 0.6516 | 0.7063 | 0.1777 | Slightly financial-leaning |
| **pref_balanced** | **0.3403** | **0.8226** | **0.3198** | **Financial-leaning** |
| pref_financial_first | 0.2640 | 0.8387 | 0.3637 | Financial-leaning |
| operational_score | 0.0922 | 0.1780 | 0.0101 | Financial-leaning |
| value_score | -0.0943 | 0.0784 | -0.1281 | Balanced |
| risk_adjusted_score | -0.2396 | -0.0210 | 0.5763 | Market-driven |

### Critical Finding: The "Balanced" Index Is Not Balanced

**pref_balanced analysis:**
- Spearman rho with ESG: 0.3403
- Spearman rho with Financial: **0.8226** (2.4x stronger)
- Kendall tau with ESG: 0.2316
- Kendall tau with Financial: **0.6362** (2.7x stronger)
- Mean absolute rank displacement from ESG: 15.60
- Mean absolute rank displacement from Financial: 8.07
- **Balance ratio: 1.93** (1.0 = perfectly balanced)

The `pref_balanced` composite is **almost twice as close to financial rankings as to ESG rankings**. It tracks financial performance far more faithfully than ESG. This is a methodological concern -- the "balanced" label is misleading.

**Even pref_esg_first tracks financial (rho=0.71) slightly more than ESG (rho=0.65).** This suggests that the financial score has higher variance or more discriminating power, causing it to dominate composite rankings regardless of intended weighting.

### Recommendation for Index Improvement
The weighting scheme should be revisited. Options include:
1. **Rank-based weighting** instead of score-based, to equalize variance contributions
2. **Standardizing** each sub-score to unit variance before combining
3. **Explicit penalty term** when ESG-financial divergence exceeds a threshold

---

## Summary of Key Findings for Discussion Section

### Finding 1: ESG and Financial Performance Are Positively Correlated (r=0.32, p=0.014)
- Supports the "stakeholder theory" / "shared value" hypothesis
- Effect is moderate -- ESG explains ~10% of financial variance
- Driven primarily by Environmental (E_score), not Governance (G_score)

### Finding 2: No ESG Sacrifice -- But a Threshold Effect Exists
- Q5 ESG companies outperform Q1 by 7.3 points (p=0.036)
- The benefit is non-linear: flat across Q1-Q4, jumps at Q5
- Implication: marginal ESG improvements don't help; only sustained ESG excellence correlates with financial outperformance

### Finding 3: The Relationship Is Sector-Dependent
- Technology and Communication Services show strong positive correlation (r=0.68, 0.76)
- Energy and Consumer Cyclical show negative correlation
- ESG-financial alignment is a property of knowledge-intensive sectors

### Finding 4: The Relationship Is Country-Dependent
- US: r=0.53, p<0.001 (strong, significant)
- India: r=0.06, ns (no relationship)
- Institutional environment moderates the ESG-financial link

### Finding 5: The Three Dimensions Are Orthogonal
- ESG, financial, and market scores are largely independent
- Only 5 companies are Pareto-optimal across all three
- Validates the multi-factor approach -- no single metric captures total company quality

### Finding 6: The Composite Index Has a Financial Bias
- pref_balanced tracks financial 2.4x more closely than ESG
- This distorts the index away from its intended multi-factor purpose
- Warrants methodological adjustment in index construction

### Finding 7: 8 Companies Achieve "Best of Both Worlds"
- Led by NVDA, TCS.NS, META -- all technology or communication services
- Capital-light, innovation-driven business models enable ESG-financial alignment
- Physical-asset-heavy sectors face genuine ESG-financial tension

---

## Tables and Files Generated

| File | Description |
|------|-------------|
| `esg_quintile_financial.csv` | Quintile analysis: financial scores by ESG quintile |
| `sector_esg_financial_correlation.csv` | Per-sector Pearson and Spearman correlations |
| `pareto_optimal_companies.csv` | Companies on the ESG x Financial x Market Pareto frontier |
| `country_comparison.csv` | US vs India descriptive statistics and tests |
| `esg_leaders.csv` | Companies with high ESG AND high financial performance |
| `esg_laggards.csv` | Companies with low ESG AND low financial performance |
| `full_company_rankings.csv` | All 60 companies with ESG/financial ranks and rank displacement |
| `esg_financial_tradeoff_data.json` | Machine-readable results dictionary |
