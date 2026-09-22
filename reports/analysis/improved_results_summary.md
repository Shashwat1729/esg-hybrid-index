# Improved Results Summary — Multi-Factor ESG Scoring Framework

**Generated**: 2026-03-25 | **Dataset**: 60 companies (40 US, 20 India) | **Features**: 154 columns | **Pipeline**: Full re-run with all methodology improvements

---

## 1. Score Distributions

### 1.1 Factor Sub-Scores (11 factors)

| Factor | Mean | Std | Min | Max | Median | Skew |
|--------|------|-----|-----|-----|--------|------|
| E_score | 50.000 | 13.027 | 15.444 | 72.940 | 52.340 | −0.718 |
| S_score | 50.000 | 8.352 | 29.679 | 65.580 | 50.140 | −0.111 |
| G_score | 50.000 | 8.766 | 32.934 | 70.514 | 49.275 | +0.192 |
| profitability_score | 0.000 | 0.688 | −0.832 | 2.030 | −0.245 | +1.307 |
| growth_score | 50.000 | 10.000 | 34.770 | 74.828 | 48.747 | +0.534 |
| efficiency_score | 0.007 | 0.499 | −0.150 | 2.750 | −0.150 | +3.850 |
| stability_score | 50.000 | 10.000 | 27.246 | 60.037 | 53.679 | −1.040 |
| valuation_score | 0.000 | 0.790 | −2.637 | 1.069 | 0.202 | −1.688 |
| market_liquidity_score | 0.000 | 0.586 | −1.356 | 1.706 | 0.002 | +0.360 |
| market_volatility_score | 0.000 | 0.697 | −1.584 | 1.421 | 0.052 | −0.083 |
| market_momentum_score | 0.000 | 0.866 | −1.885 | 2.105 | −0.152 | +0.097 |

### 1.2 Composite Scores (0–100 scale)

| Score | Mean | Std | Min | Max | Median | Skew |
|-------|------|-----|-----|-----|--------|------|
| ESG_composite | 50.000 | 10.000 | 24.221 | 66.455 | 50.839 | −0.450 |
| financial_score | 50.000 | 10.000 | 30.835 | 79.879 | 46.864 | +1.218 |
| market_score | 50.000 | 10.000 | 21.387 | 74.414 | 50.721 | −0.752 |
| operational_score | 50.000 | 10.000 | 32.706 | 77.850 | 49.592 | +0.618 |
| risk_adjusted_score | 50.000 | 10.000 | 29.694 | 70.188 | 50.264 | −0.026 |
| value_score | 50.000 | 10.000 | 16.734 | 64.242 | 52.409 | −1.556 |

### 1.3 Preference Scores

| Profile | Mean | Std | Min | Max | Median | Skew |
|---------|------|-----|-----|-----|--------|------|
| pref_esg_first | 50.833 | 14.952 | 19.592 | 79.608 | 52.271 | +0.123 |
| pref_balanced | 50.833 | 12.905 | 14.850 | 82.250 | 49.275 | +0.073 |
| pref_financial_first | 50.833 | 14.105 | 12.750 | 84.417 | 49.250 | −0.012 |

**Normality (Shapiro-Wilk)**: ESG_composite (W=0.974, p=0.219, normal); financial_score (W=0.905, p<0.001, non-normal); market_score (W=0.941, p=0.006, non-normal); all three pref scores are normally distributed (p>0.05).

### 1.4 Score Dispersion

| Score | Range | IQR | CV (%) |
|-------|-------|-----|--------|
| ESG_composite | 42.23 | 13.55 | 20.0 |
| financial_score | 49.04 | 11.07 | 20.0 |
| market_score | 53.03 | 12.61 | 20.0 |
| pref_esg_first | 60.02 | 21.46 | 29.4 |
| pref_balanced | 67.40 | 17.14 | 25.4 |
| pref_financial_first | 71.67 | 21.46 | 27.7 |

### 1.5 Gini Inequality (Score Concentration)

| Score | Gini | Interpretation |
|-------|------|----------------|
| ESG_composite | 0.340 | Moderate |
| financial_score | 0.272 | Moderate |
| market_score | 0.188 | Moderate |
| growth_score | 0.368 | Moderate |
| stability_score | 0.229 | Moderate |
| pref_balanced | 0.323 | Moderate |

---

## 2. Profile Differentiation

### 2.1 Rank Correlations Between Profiles

| Profile Pair | Spearman ρ | p-value | Kendall τ | p-value | Top-10 Overlap | Top-20 Overlap |
|-------------|-----------|---------|-----------|---------|----------------|----------------|
| ESG-First vs Balanced | 0.928 | 1.56e-26 | 0.771 | 3.33e-18 | 9/10 | 18/20 |
| ESG-First vs Financial-First | 0.888 | 3.05e-21 | 0.716 | 6.10e-16 | 8/10 | 18/20 |
| Balanced vs Financial-First | 0.988 | 3.67e-49 | 0.923 | 1.98e-25 | 8/10 | 19/20 |

### 2.2 Score-Level Correlations (Pearson r)

| Profile Pair | Pearson r | Spearman ρ | Kendall τ |
|-------------|-----------|-----------|-----------|
| ESG-First vs Balanced | 0.930 | 0.901 | 0.745 |
| ESG-First vs Financial-First | 0.883 | 0.854 | 0.683 |
| Balanced vs Financial-First | 0.991 | 0.986 | 0.915 |

### 2.3 Jaccard Similarity (Top-20)

| Pair | Jaccard | Overlap % | Shared Count |
|------|---------|-----------|-------------|
| ESG-First vs Balanced | 0.739 | 85.0% | 17/20 |
| ESG-First vs Financial-First | 0.667 | 80.0% | 16/20 |
| Balanced vs Financial-First | 0.905 | 95.0% | 19/20 |

### 2.4 Statistical Tests (Score Distributions)

| Pair | Paired t | p-value | Cohen's d | KS stat | KS p | Mean Rank Diff | Max Rank Diff |
|------|----------|---------|-----------|---------|------|----------------|---------------|
| ESG-First vs Balanced | −0.0 | 1.000 | ~0 (negligible) | 0.150 | 0.513 | 7.03 | 40 |
| ESG-First vs Fin-First | 0.0 | 1.000 | ~0 (negligible) | 0.133 | 0.665 | 12.30 | 55 |
| Balanced vs Fin-First | 0.0 | 1.000 | ~0 (negligible) | 0.083 | 0.987 | 6.10 | 25 |

**Key insight**: While aggregate score distributions are indistinguishable (Cohen's d ≈ 0, same mean=50.83), **rank orderings differ substantially**—max rank shift of 55 positions between ESG-First and Financial-First, confirming that the profiles reorder companies differently even though their score distributions overlap.

---

## 3. Top-10 Rankings by Profile

### 3.1 ESG-First Profile (w_ESG=0.40)

| Rank | Ticker | Sector | Country | Score | ESG | Financial |
|------|--------|--------|---------|-------|-----|-----------|
| 1 | NVDA | Technology | US | 79.61 | 66.37 | 61.54 |
| 2 | MSFT | Technology | US | 78.78 | 63.71 | 64.40 |
| 3 | GOOGL | Communication Services | US | 78.16 | 57.52 | 63.99 |
| 4 | V | Financial Services | US | 77.93 | 62.67 | 55.91 |
| 5 | TCS.NS | Technology | India | 77.18 | 66.45 | 79.35 |
| 6 | META | Communication Services | US | 75.63 | 63.13 | 70.05 |
| 7 | SUNPHARMA.NS | Healthcare | India | 72.06 | 60.76 | 50.80 |
| 8 | WIPRO.NS | Technology | India | 69.60 | 65.88 | 53.43 |
| 9 | AAPL | Technology | US | 67.13 | 61.61 | 60.93 |
| 10 | INFY.NS | Technology | India | 64.88 | 56.37 | 60.13 |

### 3.2 Balanced Profile (w_ESG=0.20, w_Fin=0.20)

| Rank | Ticker | Sector | Country | Score | ESG | Financial |
|------|--------|--------|---------|-------|-----|-----------|
| 1 | GOOGL | Communication Services | US | 82.25 | 57.52 | 63.99 |
| 2 | META | Communication Services | US | 74.40 | 63.13 | 70.05 |
| 3 | NVDA | Technology | US | 73.67 | 66.37 | 61.54 |
| 4 | MSFT | Technology | US | 71.53 | 63.71 | 64.40 |
| 5 | V | Financial Services | US | 69.33 | 62.67 | 55.91 |
| 6 | TCS.NS | Technology | India | 68.82 | 66.45 | 79.35 |
| 7 | ITC.NS | Consumer Defensive | India | 66.88 | 47.99 | 57.59 |
| 8 | JNJ | Healthcare | US | 66.69 | 45.49 | 56.21 |
| 9 | PFE | Healthcare | US | 66.60 | 52.45 | 54.78 |
| 10 | SBIN.NS | Financial Services | India | 66.58 | 52.81 | 55.19 |

### 3.3 Financial-First Profile (w_Fin=0.30)

| Rank | Ticker | Sector | Country | Score | ESG | Financial |
|------|--------|--------|---------|-------|-----|-----------|
| 1 | GOOGL | Communication Services | US | 84.42 | 57.52 | 63.99 |
| 2 | JNJ | Healthcare | US | 74.91 | 45.49 | 56.21 |
| 3 | ITC.NS | Consumer Defensive | India | 73.14 | 47.99 | 57.59 |
| 4 | META | Communication Services | US | 71.98 | 63.13 | 70.05 |
| 5 | NVDA | Technology | US | 69.92 | 66.37 | 61.54 |
| 6 | SBIN.NS | Financial Services | India | 69.80 | 52.81 | 55.19 |
| 7 | PFE | Healthcare | US | 69.35 | 52.45 | 54.78 |
| 8 | PM | Consumer Defensive | US | 69.07 | 51.68 | 53.91 |
| 9 | RELIANCE.NS | Energy | India | 68.65 | 24.22 | 79.88 |
| 10 | INFY.NS | Technology | India | 65.50 | 56.37 | 60.13 |

---

## 4. ESG–Financial Correlation

| Metric | Value | p-value | Interpretation |
|--------|-------|---------|---------------|
| Pearson r (ESG vs Financial) | **+0.164** | 0.2121 | Weak positive, not significant |
| Spearman ρ (ESG vs Financial) | **+0.226** | 0.0826 | Weak positive, marginal |
| Pearson r (ESG vs Market) | **−0.209** | 0.1099 | Weak negative, not significant |
| Pearson r (Financial vs Market) | **+0.018** | 0.8928 | Near-zero, not significant |

**Key finding**: The three composite dimensions (ESG, Financial, Market) are approximately **orthogonal** (|r| < 0.23 for all pairs), confirming they capture distinct information and justifying the multi-factor approach.

### ESG Quintile Analysis

| ESG Quintile | n | Fin Mean | Fin Median | ESG Mean | Market Mean |
|-------------|---|----------|------------|----------|-------------|
| Q1 (Lowest) | 12 | 48.63 | 46.72 | 42.97 | 52.42 |
| Q2 | 12 | 48.60 | 47.44 | 47.67 | 47.60 |
| Q3 | 12 | 48.77 | 48.44 | 50.20 | 53.40 |
| Q4 | 12 | 48.17 | 46.74 | 52.37 | 50.22 |
| Q5 (Highest) | 12 | 55.97 | 56.26 | 56.79 | 46.35 |

The top ESG quintile (Q5) has a notably higher financial score (55.97 vs ~48.5 for Q1–Q4), suggesting ESG leaders are not financially penalised.

---

## 5. Balance Ratio

### Profile Correlation with Dimensions

| Profile | r(ESG) | r(Financial) | r(Market) | Balance Ratio (ESG/Fin) |
|---------|--------|-------------|-----------|------------------------|
| pref_esg_first | **0.849** | 0.454 | 0.066 | 1.87 |
| pref_balanced | 0.510 | **0.709** | 0.332 | 0.72 |
| pref_financial_first | 0.145 | **0.756** | 0.478 | 0.19 |

**Interpretation**: The ESG-First profile correlates 1.87× more with ESG than Financial. The Balanced profile has a ratio of 0.72 (slightly financial-leaning). The Financial-First profile correlates 5.2× more with Financial than ESG.

### Effective Weight Decomposition (OLS regression on 3 dimensions, R²)

| Profile | ESG Weight | Financial Weight | Market Weight | R² |
|---------|-----------|-----------------|-------------|------|
| ESG-First | 1.268 | 0.465 | 0.355 | 0.877 |
| Balanced | 0.641 | 0.800 | 0.548 | 0.834 |
| Financial-First | 0.183 | 1.024 | 0.694 | 0.802 |

### Declared Weight Scheme (from pipeline configuration)

| Profile | ESG | Financial | Market | Operational | Risk-Adj | Growth | Value | Stability | Similarity | Sector |
|---------|-----|-----------|--------|-------------|----------|--------|-------|-----------|-----------|--------|
| ESG-First | 0.40 | 0.10 | 0.05 | 0.08 | 0.05 | 0.05 | 0.05 | 0.07 | 0.10 | 0.05 |
| Balanced | 0.20 | 0.20 | 0.10 | 0.10 | 0.08 | 0.10 | 0.08 | 0.05 | 0.05 | 0.04 |
| Financial-First | 0.05 | 0.30 | 0.15 | 0.10 | 0.10 | 0.10 | 0.10 | 0.05 | 0.03 | 0.02 |

---

## 6. Companies in 45–55 Band (pref_balanced)

| Band | Count | % of Universe |
|------|-------|-------------|
| **45–55** | **19** | **31.7%** |
| 40–60 | 36 | 60.0% |
| 35–65 | 43 | 71.7% |
| 30–70 | 53 | 88.3% |

**Companies in 45–55 band**: BAC, XOM, CVX, COST, AVGO, NFLX, ADBE, CRM, LIN, ABBV, HDFCBANK.NS, HINDUNILVR.NS, ICICIBANK.NS, KOTAKBANK.NS, LT.NS, AXISBANK.NS, MARUTI.NS, ULTRACEMCO.NS, BAJFINANCE.NS

This middle-band density (31.7% in a 10-point window) suggests moderate clustering around the median, with meaningful dispersion at the tails for portfolio differentiation.

---

## 7. Consensus vs Profile-Specific Picks

### Top-20 Consensus Analysis

| Category | Count | Companies |
|----------|-------|-----------|
| **All 3 profiles (consensus)** | **13** | AAPL, GOOGL, INFY.NS, ITC.NS, META, MSFT, NVDA, PFE, PM, SBIN.NS, SUNPHARMA.NS, TCS.NS, V |
| Only ESG-First | 5 | ADBE, AXISBANK.NS, BAJFINANCE.NS, CRM, TSLA |
| Only Balanced | 0 | (none) |
| Only Financial-First | 2 | CVX, MARUTI.NS |
| ESG-First & Balanced only | 2 | CSCO, WIPRO.NS |
| ESG-First & Fin-First only | 0 | (none) |
| Balanced & Fin-First only | 5 | AMZN, BHARTIARTL.NS, JNJ, PG, RELIANCE.NS |

**Key finding**: 13 of 20 (65%) top-20 companies are consensus picks across all profiles, demonstrating strong core agreement. The ESG-First profile has the most unique picks (5), while the Balanced profile has zero unique picks (it is a blend). Profile-specific selections like TSLA (ESG-First unique) and RELIANCE.NS (Balanced & Fin-First) illustrate meaningful differentiation at the margin.

### Biggest Rank Shifts (ESG-First → Financial-First)

| Ticker | ESG-First Rank | Fin-First Rank | Shift |
|--------|---------------|----------------|-------|
| RELIANCE.NS | 51 | 9 | +42 (gains) |
| CVX | 52 | 17 | +35 |
| ULTRACEMCO.NS | 49 | 22 | +27 |
| XOM | 48 | 23 | +25 |
| TSLA | 15 | 54 | −39 (loses) |
| CRM | 18 | 46 | −28 |
| KOTAKBANK.NS | 26 | 51 | −25 |

---

## 8. Sector Analysis

### Mean Composite by Sector for Each Profile

| Sector | N | ESG | Financial | Market | ESG-First | Balanced | Fin-First |
|--------|---|-----|-----------|--------|-----------|----------|-----------|
| Technology | 12 | **59.03** | 55.36 | 44.78 | **63.08** | **57.58** | 54.15 |
| Communication Svcs | 7 | 51.70 | 56.56 | 51.16 | 54.71 | 56.51 | **57.47** |
| Financial Services | 11 | 53.73 | 47.67 | 48.63 | 51.72 | 49.95 | 47.72 |
| Consumer Defensive | 7 | 50.15 | 47.18 | 56.76 | 53.84 | 53.70 | 54.61 |
| Healthcare | 8 | 50.41 | 47.76 | 47.24 | 51.08 | 49.74 | 49.12 |
| Consumer Cyclical | 6 | 46.61 | 41.55 | 51.23 | 41.03 | 39.43 | 38.19 |
| Energy | 3 | 31.07 | **58.19** | 59.41 | 35.56 | 50.81 | **61.76** |
| Basic Materials | 3 | 31.39 | 46.04 | 57.74 | 33.77 | 44.18 | 52.05 |
| Industrials | 2 | 38.90 | 47.87 | 45.29 | 34.01 | 39.74 | 42.40 |
| Utilities | 1 | 39.52 | 45.32 | 44.90 | 33.48 | 39.18 | 42.28 |

### Sector Rankings by Profile

| Rank | ESG-First | Balanced | Financial-First |
|------|-----------|----------|----------------|
| 1 | Technology (63.08) | Technology (57.58) | Energy (61.76) |
| 2 | Comm. Svcs (54.71) | Comm. Svcs (56.51) | Comm. Svcs (57.47) |
| 3 | Cons. Defensive (53.84) | Cons. Defensive (53.70) | Cons. Defensive (54.61) |
| 4 | Financial Svcs (51.72) | Energy (50.81) | Technology (54.15) |
| 5 | Healthcare (51.08) | Financial Svcs (49.95) | Basic Materials (52.05) |

**Key insight**: Technology dominates under ESG-First and Balanced profiles; Energy rises to #1 under Financial-First. Communication Services and Consumer Defensive are consistently top-3 across all profiles.

### Intra-Sector ESG–Financial Correlation

| Sector | n | Pearson r | p-value | Direction |
|--------|---|-----------|---------|-----------|
| Communication Services | 7 | **+0.764** | 0.046 | Positive (sig.) |
| Technology | 12 | **+0.683** | 0.014 | Positive (sig.) |
| Healthcare | 8 | +0.249 | 0.552 | Positive (n.s.) |
| Consumer Defensive | 7 | +0.022 | 0.963 | Near-zero |
| Financial Services | 11 | −0.340 | 0.306 | Negative (n.s.) |
| Consumer Cyclical | 6 | −0.500 | 0.313 | Negative (n.s.) |
| Energy | 3 | −0.759 | 0.451 | Negative (n.s.) |

---

## 9. Materiality Weights by Sector

| Sector | E Weight | S Weight | G Weight |
|--------|---------|---------|---------|
| Energy | 0.55 | 0.20 | 0.25 |
| Basic Materials | 0.50 | 0.25 | 0.25 |
| Utilities | 0.50 | 0.20 | 0.30 |
| Industrials | 0.40 | 0.30 | 0.30 |
| Consumer Defensive | 0.30 | 0.35 | 0.35 |
| Consumer Cyclical | 0.25 | 0.40 | 0.35 |
| Healthcare | 0.25 | 0.40 | 0.35 |
| Communication Services | 0.20 | 0.45 | 0.35 |
| Technology | 0.20 | 0.45 | 0.35 |
| Financial Services | 0.15 | 0.35 | 0.50 |

**8 unique materiality weight patterns** across 10 sectors. Mean: E=0.258, S=0.377, G=0.365.

---

## 10. Country Analysis (US vs India)

| Metric | US (n=40) | India (n=20) | t-statistic | p-value |
|--------|-----------|-------------|------------|---------|
| ESG_composite | 50.03 | 49.93 | +0.037 | 0.971 |
| financial_score | 48.08 | **53.84** | −2.166 | **0.034** |
| E_score | 49.54 | 50.92 | −0.386 | 0.701 |
| S_score | 49.93 | 50.14 | −0.088 | 0.930 |
| G_score | 49.62 | 50.77 | −0.477 | 0.636 |
| market_score | 49.94 | 50.12 | −0.065 | 0.949 |

**Key finding**: Indian companies score significantly higher on financial performance (53.84 vs 48.08, p=0.034). ESG scores are statistically indistinguishable between countries (p=0.971).

---

## 11. Benchmarking & External Validation

### Cross-Provider Correlation (n=14 overlap companies)

| Provider | Spearman ρ | p-value | Pearson r | p-value |
|----------|-----------|---------|-----------|---------|
| MSCI (numeric) | **+0.563** | **0.036** | +0.484 | 0.080 |
| S&P Global | +0.358 | 0.209 | +0.305 | 0.289 |
| Sustainalytics (inverted) | **+0.780** | **0.001** | **+0.761** | **0.002** |

### Factor Validity (Information Coefficient against 6-month momentum)

| Factor | IC (Spearman) | p-value | Significant? | Monotonic? |
|--------|-------------|---------|-------------|-----------|
| market_score | **+0.469** | **0.0002** | Yes | Yes |
| sector_position | +0.255 | 0.050 | Yes | Yes |
| growth_score | +0.177 | 0.177 | No | No |
| stability_score | +0.078 | 0.553 | No | No |
| ESG_composite | +0.072 | 0.584 | No | No |
| financial_score | −0.018 | 0.893 | No | No |

### Bootstrap Robustness (1,000 iterations, ±20% weight perturbation)

- **Top-20 stability**: 97.7% ± 3.0% retained
- **Average 95% CI width**: 5.1 ranks
- **Most stable**: GOOGL (rank 1, CI: 1–2), BHARTIARTL.NS (rank 4, CI: 4–4)
- **Least stable at boundary**: AMZN (rank 20, CI: 18–21, 78.0% top-20), CSCO (rank 18, CI: 17–22, 93.8% top-20)

### Portfolio Comparison: Top-20 vs Bottom-20 (by pref_balanced)

| Metric | Top-20 | Bottom-20 | Difference | p-value |
|--------|--------|-----------|-----------|---------|
| ESG_composite | 55.13 | 45.14 | **+9.99** | **0.0008** |
| financial_score | 59.87 | 42.47 | **+17.39** | **<0.0001** |
| market_score | 53.07 | 47.12 | +5.96 | 0.066 |
| E_score | 56.36 | 46.32 | +10.04 | **0.011** |
| S_score | 53.49 | 46.47 | +7.02 | **0.008** |
| G_score | 50.72 | 46.54 | +4.18 | 0.067 |

---

## 12. Weight Robustness (Sensitivity Analysis)

### Perturbation Test Results

| Profile | ±5% | ±10% | ±20% |
|---------|-----|------|------|
| ESG-First Spearman | 0.9996 | 0.9989 | 0.9970 |
| ESG-First Top-20 | 20.0/20 | 20.0/20 | 19.7/20 |
| Balanced Spearman | 0.9990 | 0.9975 | 0.9932 |
| Balanced Top-20 | 20.0/20 | 19.9/20 | 19.7/20 |
| Fin-First Spearman | 0.9994 | 0.9987 | 0.9969 |
| Fin-First Top-20 | 20.0/20 | 20.0/20 | 19.9/20 |

All profiles maintain Spearman ρ > 0.99 even at ±20% weight perturbation.

### Factor Ablation (removing one factor at a time from Balanced)

| Removed Factor | Kendall τ | Spearman ρ | Top-10 Overlap | Max Rank Shift |
|---------------|-----------|-----------|---------------|---------------|
| financial_score | 0.746 | 0.916 | 7/10 | 22 |
| market_score | 0.810 | 0.940 | 8/10 | 21 |
| ESG_composite | 0.855 | 0.964 | 8/10 | 13 |
| growth_score | 0.866 | 0.967 | 9/10 | 15 |
| value_score | 0.885 | 0.972 | 9/10 | 19 |

**Key finding**: Financial_score and market_score are the most impactful factors (largest rank disruption when removed), while operational_score has the least impact.

---

## 13. PCA & Dimensionality

| Component | Eigenvalue | Variance Explained | Cumulative |
|-----------|-----------|-------------------|-----------|
| PC1 | 1.962 | 24.1% | 24.1% |
| PC2 | 1.735 | 21.3% | 45.4% |
| PC3 | 1.418 | 17.4% | 62.9% |
| PC4 | 1.095 | 13.5% | 76.3% |
| PC5 | 0.742 | 9.1% | 85.5% |

**4 components** explain 76.3% of variance (above Kaiser criterion eigenvalue > 1). PC1 loads on financial_score (+0.46) and sector_position (+0.52); PC2 loads on ESG_composite (+0.51) and risk_adjusted_score (−0.49).

---

## 14. Portfolio Characteristics (Top-20 by Profile)

| Metric | ESG-First | Balanced | Financial-First |
|--------|-----------|----------|----------------|
| Avg ESG score | **58.34** | 55.13 | 53.85 |
| Avg Financial score | 58.04 | **59.87** | **59.87** |
| Avg Market score | 51.09 | 53.07 | 53.88 |
| Sharpe ratio (proxy) | 0.006 | 0.236 | **0.291** |
| Avg return (6m momentum) | 0.07% | 2.74% | **3.25%** |
| N sectors | 5 | 7 | 7 |
| Top sector | Tech (45%) | Tech (35%) | Tech (30%) |
| Sector HHI | 0.280 | 0.205 | **0.180** |
| US / India split | 65/35 | 60/40 | 60/40 |

---

## 15. Pareto-Optimal Companies

Companies on the ESG–Financial–Market Pareto frontier:

| Ticker | ESG | Financial | Market | Sector |
|--------|-----|-----------|--------|--------|
| NVDA | 59.87 | 58.57 | 58.52 | Technology |
| TCS.NS | 58.90 | 71.74 | 44.23 | Technology |
| META | 57.26 | 64.86 | 49.44 | Communication Services |
| GOOGL | 55.55 | 60.37 | 73.22 | Communication Services |
| RELIANCE.NS | 42.00 | 72.13 | 55.03 | Energy |

---

## Summary of Key Numbers for Paper

| Metric | Value |
|--------|-------|
| Universe size | 60 companies (40 US, 20 India), 10 sectors |
| Feature count | 154 columns, 10 factor sub-scores, 6 composites, 3 preference profiles |
| ESG–Financial correlation | r = +0.164 (p = 0.212), approximately orthogonal |
| Sustainalytics validation | ρ = +0.780 (p = 0.001) |
| MSCI validation | ρ = +0.563 (p = 0.036) |
| Bootstrap top-20 stability | 97.7% ± 3.0% |
| Weight sensitivity (±20%) | Spearman ρ > 0.993 for all profiles |
| Consensus top-20 companies | 13/20 (65%) shared across all profiles |
| ESG-First unique picks | 5 (ADBE, AXISBANK.NS, BAJFINANCE.NS, CRM, TSLA) |
| Financial-First unique picks | 2 (CVX, MARUTI.NS) |
| Max rank shift (ESG→Fin) | RELIANCE.NS (+42 ranks), TSLA (−39 ranks) |
| India financial advantage | 53.84 vs 48.08 (p = 0.034) |
| ESG country gap | None (p = 0.971) |
| Top-20 ESG premium | +9.99 points (p = 0.0008) |
| Top-20 financial premium | +17.39 points (p < 0.0001) |
| Sector materiality patterns | 8 unique weight configurations |
| PCA components (Kaiser) | 4 components, 76.3% variance |
| Balanced mid-band (45–55) | 19 companies (31.7%) |
| Financial-First Sharpe (proxy) | 0.291 vs ESG-First 0.006 |
