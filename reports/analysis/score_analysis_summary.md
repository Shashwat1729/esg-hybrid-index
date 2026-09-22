# Score & Rankings Analysis Summary

**Dataset:** `data/processed/indexed_data.csv`
**Companies:** 60 (40 US, 20 India)
**Factors:** 10 primary factor scores + 8 sub-factor scores
**Analysis Date:** March 25, 2026

---

## 1. Descriptive Statistics for 10 Factor Scores

| Factor | Mean | Std | Min | Q1 | Median | Q3 | Max | Range | CV(%) | Skewness | Kurtosis |
|--------|------|-----|-----|-----|--------|-----|-----|-------|-------|----------|----------|
| E_score | 50.00 | 10.42 | 22.36 | 43.56 | 51.87 | 57.16 | 68.35 | 46.00 | 20.84 | -0.70 | 0.05 |
| S_score | 50.00 | 6.68 | 33.74 | 44.49 | 50.11 | 54.80 | 62.46 | 28.72 | 13.36 | -0.11 | -0.73 |
| G_score | 50.00 | 7.01 | 36.35 | 45.80 | 49.42 | 54.54 | 66.41 | 30.06 | 14.03 | 0.19 | -0.36 |
| ESG_composite | 50.00 | 4.89 | 38.47 | 47.01 | 50.26 | 52.86 | 59.87 | 21.40 | 9.78 | -0.21 | -0.22 |
| financial_score | 50.03 | 7.40 | 35.85 | 45.24 | 47.71 | 53.43 | 72.13 | 36.28 | 14.79 | 1.19 | 1.62 |
| market_score | 50.00 | 9.51 | 22.78 | 45.12 | 50.69 | 57.11 | 73.22 | 50.44 | 19.02 | -0.73 | 0.85 |
| operational_score | 49.90 | 4.22 | 42.61 | 46.89 | 49.72 | 51.41 | 61.64 | 19.03 | 8.45 | 0.60 | 0.09 |
| risk_adjusted_score | 50.11 | 6.68 | 36.54 | 45.66 | 50.29 | 54.83 | 63.60 | 27.06 | 13.34 | -0.03 | -0.48 |
| value_score | 50.25 | 7.33 | 25.87 | 48.29 | 52.02 | 55.22 | 60.69 | 34.82 | 14.59 | -1.52 | 1.98 |
| pref_balanced | 51.60 | 3.86 | 43.85 | 49.16 | 50.78 | 54.36 | 61.41 | 17.57 | 7.49 | 0.44 | -0.22 |

### Key Observations
- All pillar scores (E, S, G) and ESG_composite are centered at mean=50 with standard deviations ranging from 4.89 (ESG_composite) to 10.42 (E_score).
- **E_score has the highest variability** (CV=20.84%) among ESG pillars, indicating environmental performance is the most differentiating ESG dimension.
- **ESG_composite is heavily compressed** (std=4.89, range=21.40) due to the averaging effect of combining three weakly correlated pillars.
- **financial_score is right-skewed** (skew=1.19, kurtosis=1.62), with a few companies scoring exceptionally high (RELIANCE.NS=72.13, TCS.NS=71.74).
- **value_score is left-skewed** (skew=-1.52, kurtosis=1.98), with several companies dragged down by extreme negative valuation z-scores.
- **pref_balanced** (the final composite) has the lowest variability (CV=7.49%), reflecting compression from multi-factor aggregation.

---

## 2. Top-10 and Bottom-10 Companies by Composite Score

### Top 10 (pref_balanced)

| Rank | Ticker | Sector | pref_balanced | ESG_composite | financial_score | market_score |
|------|--------|--------|---------------|---------------|-----------------|--------------|
| 1 | GOOGL | Communication Services | 61.41 | 55.55 | 60.37 | 73.22 |
| 2 | TCS.NS | Technology | 60.81 | 58.90 | 71.74 | 44.23 |
| 3 | META | Communication Services | 59.00 | 57.26 | 64.86 | 49.44 |
| 4 | BHARTIARTL.NS | Communication Services | 58.13 | 51.95 | 70.19 | 43.45 |
| 5 | NVDA | Technology | 57.14 | 59.87 | 58.57 | 58.52 |
| 6 | JNJ | Healthcare | 56.49 | 47.43 | 54.62 | 61.35 |
| 7 | ITC.NS | Consumer Defensive | 56.35 | 48.02 | 55.64 | 54.42 |
| 8 | RELIANCE.NS | Energy | 56.10 | 42.00 | 72.13 | 55.03 |
| 9 | PFE | Healthcare | 55.68 | 51.17 | 53.56 | 56.37 |
| 10 | PM | Consumer Defensive | 55.67 | 50.14 | 52.92 | 59.39 |

### Bottom 10 (pref_balanced)

| Rank | Ticker | Sector | pref_balanced | ESG_composite | financial_score | market_score |
|------|--------|--------|---------------|---------------|-----------------|--------------|
| 51 | JPM | Financial Services | 47.88 | 50.75 | 43.96 | 50.51 |
| 52 | CRM | Technology | 47.66 | 52.04 | 47.23 | 29.03 |
| 53 | RTX | Industrials | 47.50 | 39.88 | 43.51 | 44.52 |
| 54 | NESTLEIND.NS | Consumer Defensive | 47.48 | 47.99 | 40.46 | 53.52 |
| 55 | TSLA | Consumer Cyclical | 47.32 | 52.26 | 35.85 | 58.35 |
| 56 | TITAN.NS | Consumer Cyclical | 46.82 | 51.84 | 38.50 | 52.65 |
| 57 | ABT | Healthcare | 45.56 | 47.23 | 46.32 | 22.78 |
| 58 | HD | Consumer Cyclical | 45.39 | 41.55 | 44.20 | 50.53 |
| 59 | UNH | Healthcare | 45.27 | 48.84 | 44.73 | 28.76 |
| 60 | NKE | Consumer Cyclical | 43.85 | 43.73 | 42.42 | 34.96 |

### Ranking Characteristics
- **Top performers** are driven primarily by strong **financial_score** (TCS.NS=71.74, RELIANCE.NS=72.13, BHARTIARTL.NS=70.19) and/or strong **market_score** (GOOGL=73.22).
- **GOOGL ranks #1** with balanced strength across all three macro pillars (ESG=55.55, Financial=60.37, Market=73.22).
- **Bottom companies** suffer from weak financial performance (TSLA=35.85, TITAN.NS=38.50) and/or poor market scores (ABT=22.78, UNH=28.76, CRM=29.03).
- **Sector pattern**: Communication Services and Technology dominate the top; Consumer Cyclical dominates the bottom.
- **financial_score is the strongest driver** of the final ranking (Pearson r=0.83 with pref_balanced).

### Ranking Consistency Across Preference Weightings
The top-5 is largely stable across all three preference profiles:
- **GOOGL and TCS.NS** are consistently #1 and #2.
- **META** is consistently #3.
- Minor reordering between BHARTIARTL.NS/NVDA/RELIANCE.NS depending on ESG vs financial weighting.

---

## 3. Score Distribution Analysis (Normality Tests)

| Factor | Shapiro-Wilk W | p-value | Normal? | Anderson-Darling | AD Normal? |
|--------|---------------|---------|---------|-----------------|------------|
| E_score | 0.9574 | 0.035 | No | 0.787 | No |
| S_score | 0.9761 | 0.288 | Yes | 0.489 | Yes |
| G_score | 0.9851 | 0.676 | Yes | 0.250 | Yes |
| ESG_composite | 0.9867 | 0.758 | Yes | 0.189 | Yes |
| financial_score | 0.9052 | 0.000 | **No** | 1.781 | **No** |
| market_score | 0.9414 | 0.006 | **No** | 1.088 | **No** |
| operational_score | 0.9613 | 0.055 | Yes (marginal) | 0.734 | Yes (marginal) |
| risk_adjusted_score | 0.9850 | 0.669 | Yes | 0.160 | Yes |
| value_score | 0.8463 | 0.000 | **No** | 3.008 | **No** |
| pref_balanced | 0.9737 | 0.221 | Yes | 0.614 | Yes |

### Summary
- **6 of 10 factors pass the Shapiro-Wilk normality test** at alpha=0.05.
- **Non-normal factors**: E_score (left-skewed), financial_score (right-skewed with heavy tail), market_score (left-skewed), value_score (severely left-skewed).
- S_score, G_score, ESG_composite, risk_adjusted_score, and pref_balanced are all approximately normally distributed.
- The non-normality of financial_score (driven by a few high-performing Indian companies) and value_score (driven by extreme negative valuations) warrants use of **rank-based/non-parametric methods** in downstream analysis.

---

## 4. Inter-Factor Correlation Matrix

### Pearson Correlations

|  | E_score | S_score | G_score | ESG_comp | fin_score | mkt_score | ops_score | risk_adj | value | balanced |
|--|---------|---------|---------|----------|-----------|-----------|-----------|----------|-------|----------|
| E_score | 1.000 | 0.232 | -0.143 | 0.796 | 0.275 | -0.152 | 0.115 | -0.106 | -0.091 | 0.404 |
| S_score | 0.232 | 1.000 | -0.109 | 0.605 | 0.244 | -0.065 | 0.230 | -0.243 | -0.076 | 0.286 |
| G_score | -0.143 | -0.109 | 1.000 | 0.271 | -0.013 | -0.084 | -0.112 | -0.177 | 0.049 | -0.022 |
| ESG_composite | 0.796 | 0.605 | 0.271 | 1.000 | 0.316 | -0.181 | 0.147 | -0.271 | -0.083 | 0.429 |
| financial_score | 0.275 | 0.244 | -0.013 | 0.316 | 1.000 | 0.018 | 0.163 | -0.040 | 0.188 | **0.832** |
| market_score | -0.152 | -0.065 | -0.084 | -0.181 | 0.018 | 1.000 | 0.040 | 0.626 | -0.173 | 0.392 |
| operational_score | 0.115 | 0.230 | -0.112 | 0.147 | 0.163 | 0.040 | 1.000 | -0.078 | -0.215 | 0.289 |
| risk_adjusted | -0.106 | -0.243 | -0.177 | -0.271 | -0.040 | **0.626** | -0.078 | 1.000 | -0.015 | 0.238 |
| value_score | -0.091 | -0.076 | 0.049 | -0.083 | 0.188 | -0.173 | -0.215 | -0.015 | 1.000 | 0.129 |
| pref_balanced | 0.404 | 0.286 | -0.022 | 0.429 | **0.832** | 0.392 | 0.289 | 0.238 | 0.129 | 1.000 |

### Key Correlation Findings

**Strongest positive correlations:**
1. `financial_score` <-> `pref_balanced`: r=0.832 (financial performance dominates the ranking)
2. `E_score` <-> `ESG_composite`: r=0.796 (E pillar drives ESG composite most)
3. `market_score` <-> `risk_adjusted_score`: r=0.626 (market conditions heavily influence risk-adjusted scores)
4. `S_score` <-> `ESG_composite`: r=0.605

**Notable near-zero / weak correlations:**
- `financial_score` <-> `market_score`: r=0.018 (essentially orthogonal -- financial fundamentals and market behavior are independent dimensions)
- `ESG_composite` <-> `market_score`: r=-0.181 (weakly negative -- higher ESG companies don't systematically have better market performance)
- `G_score` <-> `financial_score`: r=-0.013 (governance and financial performance are uncorrelated)
- `value_score` is largely independent of all other factors (max |r|=0.215)

**ESG intra-pillar correlations:**
- E <-> S: r=0.232 (weak positive)
- E <-> G: r=-0.143 (weakly negative)
- S <-> G: r=-0.109 (weakly negative)
- This confirms the three ESG pillars capture **distinct, largely independent dimensions** of sustainability.

**Spearman correlations** are broadly consistent with Pearson, confirming that relationships are not driven by outliers. Key Spearman values: financial_score<->pref_balanced: rho=0.823, E_score<->ESG_composite: rho=0.779.

---

## 5. Differentiation Power Analysis

| Rank | Factor | CV(%) | Range | Std | IQR | MAD |
|------|--------|-------|-------|-----|-----|-----|
| 1 | E_score | 20.84 | 46.00 | 10.42 | 13.60 | 8.14 |
| 2 | market_score | 19.02 | 50.44 | 9.51 | 12.00 | 7.24 |
| 3 | financial_score | 14.79 | 36.28 | 7.40 | 8.19 | 5.55 |
| 4 | value_score | 14.59 | 34.82 | 7.33 | 6.92 | 5.30 |
| 5 | G_score | 14.03 | 30.06 | 7.01 | 8.74 | 5.53 |
| 6 | S_score | 13.36 | 28.72 | 6.68 | 10.31 | 5.61 |
| 7 | risk_adjusted_score | 13.34 | 27.06 | 6.68 | 9.17 | 5.36 |
| 8 | ESG_composite | 9.78 | 21.40 | 4.89 | 5.85 | 3.81 |
| 9 | operational_score | 8.45 | 19.03 | 4.22 | 4.52 | 3.18 |
| 10 | pref_balanced | 7.49 | 17.57 | 3.86 | 5.21 | 3.15 |

### Interpretation
- **E_score provides the most differentiation** (CV=20.84%), meaning environmental performance varies the most across companies. This is the strongest signal for distinguishing ESG leaders from laggards.
- **market_score is the second most differentiating** (CV=19.02%), with the largest absolute range (50.44 points). Market-based factors create meaningful separation.
- **Composite scores are the most compressed**: ESG_composite (CV=9.78%), operational_score (CV=8.45%), and pref_balanced (CV=7.49%). Multi-factor aggregation averages out extremes, reducing discriminatory power.
- The compression in pref_balanced (range=17.57, std=3.86) means the difference between rank #1 and rank #60 is only ~17.6 points on a 0-100 scale. This is a common challenge in multi-factor scoring systems.

---

## 6. Anomaly Detection

### 6.1 Scores Outside Expected 0-100 Range

Several **sub-factor scores** use z-score normalization and are therefore not bounded to 0-100:

| Sub-Factor | # Below 0 | # Above 100 | Min | Max | Issue |
|------------|-----------|-------------|-----|-----|-------|
| profitability_score | 39/60 | 0 | -0.83 | 2.03 | Z-score based; not 0-100 |
| efficiency_score | 54/60 | 0 | -0.15 | 2.75 | Z-score based; 90% at -0.15 (floor) |
| valuation_score | 18/60 | 0 | -2.64 | 1.07 | Z-score based; left-skewed |
| market_liquidity_score | 29/60 | 0 | -1.36 | 1.71 | Z-score based |
| market_volatility_score | 29/60 | 0 | -1.58 | 1.42 | Z-score based |
| market_momentum_score | 33/60 | 0 | -1.89 | 2.11 | Z-score based |

**All 10 primary factor scores (E, S, G, ESG_composite, financial_score, market_score, operational_score, risk_adjusted_score, value_score, pref_balanced) are within the 0-100 range.** The sub-factor z-scores are intermediate calculations that are rescaled before aggregation.

### 6.2 Extreme Outliers (|z| > 2.5 within each factor)

| Factor | Ticker | Score | z-score | Direction |
|--------|--------|-------|---------|-----------|
| E_score | RELIANCE.NS | 22.36 | -2.65 | Low |
| E_score | ULTRACEMCO.NS | 23.83 | -2.51 | Low |
| financial_score | RELIANCE.NS | 72.13 | +2.99 | High |
| financial_score | TCS.NS | 71.74 | +2.93 | High |
| financial_score | BHARTIARTL.NS | 70.19 | +2.73 | High |
| market_score | ABT | 22.78 | -2.86 | Low |
| operational_score | PFE | 61.64 | +2.78 | High |
| value_score | NESTLEIND.NS | 25.87 | -3.33 | Low |
| value_score | TSLA | 30.78 | -2.66 | Low |
| value_score | TITAN.NS | 30.96 | -2.63 | Low |
| pref_balanced | GOOGL | 61.41 | +2.54 | High |

**Notable anomalies:**
- **RELIANCE.NS** is an extreme case: worst E_score (22.36, z=-2.65) but best financial_score (72.13, z=+2.99). This highlights the ESG-financial tension in the energy sector.
- **Three Indian companies** (RELIANCE.NS, TCS.NS, BHARTIARTL.NS) are extreme outliers on the high end of financial_score. This may reflect the financial characteristics of large Indian conglomerates or potential data distribution differences.
- **NESTLEIND.NS** has the most extreme value_score outlier (z=-3.33), driven by extremely high price-to-book and PE ratios.
- **ABT** has an extreme low market_score (22.78, z=-2.86), driven by weak market momentum and low trading volume.
- **GOOGL** is a mild positive outlier on pref_balanced (z=+2.54), being the only company substantially separated from the pack.

### 6.3 Score Compression Analysis

| Score | Companies in 45-55 | Companies in 40-60 |
|-------|-------------------|--------------------|
| pref_balanced | 44/60 (73.3%) | 58/60 (96.7%) |
| ESG_composite | 41/60 (68.3%) | 57/60 (95.0%) |
| financial_score | 36/60 (60.0%) | 52/60 (86.7%) |
| market_score | 25/60 (41.7%) | 49/60 (81.7%) |

**73.3% of all companies have a pref_balanced score between 45 and 55**, and 96.7% fall within 40-60. This extreme compression means that small score differences drive large ranking changes -- the difference between rank #1 (61.41) and rank #30 (50.93) is only ~10.5 points.

---

## 7. Sector-Level Analysis

| Sector | N | pref_balanced (mean) | ESG_composite | financial_score | market_score |
|--------|---|---------------------|---------------|-----------------|--------------|
| Communication Services | 7 | 54.01 | 50.68 | 54.88 | 51.10 |
| Technology | 12 | 52.88 | 54.67 | 54.00 | 45.03 |
| Consumer Defensive | 7 | 52.42 | 49.38 | 47.94 | 56.43 |
| Energy | 3 | 51.94 | 43.98 | 56.09 | 58.95 |
| Financial Services | 11 | 50.82 | 52.15 | 48.30 | 48.70 |
| Basic Materials | 3 | 50.27 | 41.48 | 47.10 | 57.36 |
| Healthcare | 8 | 51.12 | 49.56 | 48.37 | 47.37 |
| Industrials | 2 | 49.91 | 43.49 | 48.45 | 45.52 |
| Consumer Cyclical | 6 | 48.57 | 47.26 | 43.78 | 51.17 |
| Utilities | 1 | 50.44 | 46.58 | 46.57 | 45.15 |

- **Communication Services** leads overall, driven by GOOGL and META.
- **Technology** has the highest ESG_composite (54.67) and strong financials.
- **Energy** has the lowest ESG_composite (43.98) but highest financial and market scores.
- **Consumer Cyclical** ranks last, with the weakest financial_score (43.78).

---

## 8. US vs India Comparison

| Factor | US (n=40) | India (n=20) | t-stat | p-value |
|--------|-----------|-------------|--------|---------|
| pref_balanced | 51.03 +/- 3.92 | 52.73 +/- 3.57 | -1.62 | 0.110 |
| ESG_composite | 49.76 +/- 4.86 | 50.48 +/- 5.04 | -0.54 | 0.594 |
| financial_score | 48.61 +/- 5.85 | **52.87 +/- 9.33** | -2.17 | **0.034** |
| market_score | 49.94 +/- 11.11 | 50.11 +/- 5.26 | -0.07 | 0.949 |
| E_score | 49.63 +/- 9.70 | 50.74 +/- 11.98 | -0.39 | 0.701 |
| S_score | 49.95 +/- 6.51 | 50.11 +/- 7.19 | -0.09 | 0.930 |
| G_score | 49.69 +/- 7.05 | 50.61 +/- 7.07 | -0.48 | 0.636 |

- **Financial_score is the only statistically significant difference** (p=0.034): Indian companies score higher on financial metrics on average. However, Indian financial_score also has much higher variance (std=9.33 vs 5.85).
- No significant differences in ESG scores, market scores, or the overall composite.
- The Mann-Whitney U test confirms financial_score as the only marginal difference (p=0.061).

---

## Summary of Key Findings for Research Paper

1. **Financial dominance in rankings**: The financial_score has a Pearson correlation of r=0.832 with the final composite (pref_balanced), making it the single strongest determinant of company rankings.

2. **ESG pillars are independent**: E, S, and G pillars show weak inter-correlations (|r| < 0.25), validating the multi-dimensional nature of ESG assessment.

3. **Environmental score differentiates most**: E_score has the highest CV (20.84%) among all factors, providing the greatest discriminatory power for distinguishing companies.

4. **Score compression is significant**: 73.3% of companies fall within a 10-point band (45-55) on the composite score, and the total range is only 17.6 points. This is inherent to multi-factor aggregation.

5. **Mixed normality**: 6/10 factors are normally distributed; financial_score, market_score, and value_score are significantly non-normal, warranting non-parametric analysis methods.

6. **ESG-financial orthogonality**: ESG_composite and financial_score have only moderate positive correlation (r=0.316, p=0.014), while ESG_composite and market_score are weakly negative (r=-0.181, p=0.167). This supports the multi-factor framework's value proposition.

7. **RELIANCE.NS is a defining anomaly**: Worst E_score (22.36) but best financial_score (72.13), exemplifying the ESG-financial tension in carbon-intensive sectors.

8. **Ranking stability**: Top-3 companies (GOOGL, TCS.NS, META) are consistent across all three preference weightings (ESG-first, balanced, financial-first), suggesting robust multi-dimensional excellence.

9. **Sector effects**: Communication Services and Technology companies cluster in the top quartile; Consumer Cyclical companies cluster in the bottom quartile.

10. **India vs US**: No significant differences in ESG or composite scores; Indian companies show marginally higher financial scores (p=0.034) driven by a few high-performers (TCS.NS, BHARTIARTL.NS, RELIANCE.NS).
