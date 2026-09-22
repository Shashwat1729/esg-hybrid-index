# Investor Profile & Portfolio Construction Analysis

**Generated:** 2026-03-25  
**Data Source:** `data/processed/indexed_data.csv` (60 companies, 151 features)  
**Config:** `config/index_config.yaml` (10-factor preference scoring with 3 investor profiles)

---

## 1. Preference Score Verification

All three investor profile preference scores are present and fully populated (N=60, zero nulls):

| Profile | Column | Mean | Std | Min | Max | Range | Q25 | Median | Q75 |
|---------|--------|------|-----|-----|-----|-------|-----|--------|-----|
| ESG-First | `pref_esg_first` | 51.93 | 3.56 | 44.69 | 60.82 | 16.13 | 49.75 | 51.05 | 54.77 |
| Balanced | `pref_balanced` | 51.60 | 3.86 | 43.85 | 61.41 | 17.57 | 49.16 | 50.78 | 54.36 |
| Financial-First | `pref_financial_first` | 51.19 | 3.86 | 43.46 | 60.93 | 17.47 | 48.75 | 50.45 | 53.92 |

**Key observations:**
- All scores are centered near 50 (z-score normalized components mapped to 50-scale)
- Balanced and Financial-First profiles exhibit slightly higher dispersion (std ~3.86) than ESG-First (3.56), indicating greater differentiation power when financial factors dominate
- The score range increases from ESG-First (16.13) to Balanced (17.57), suggesting financial factor weights amplify inter-company spread

### Profile Weight Configurations (from `index_config.yaml`)

| Factor | ESG-First | Balanced | Financial-First |
|--------|-----------|----------|-----------------|
| ESG Score | **0.35** | 0.15 | 0.10 |
| Financial Score | 0.15 | **0.25** | **0.30** |
| Market Score | 0.08 | 0.10 | 0.10 |
| Operational Score | 0.10 | 0.10 | 0.10 |
| Risk-Adjusted Score | 0.08 | 0.08 | **0.12** |
| Growth Score | 0.06 | **0.12** | 0.10 |
| Value Score | 0.05 | 0.08 | 0.08 |
| Stability Score | 0.05 | 0.05 | 0.05 |
| Similarity Rank | 0.05 | 0.04 | 0.03 |
| Sector Position | 0.03 | 0.03 | 0.02 |
| **Total** | **1.00** | **1.00** | **1.00** |

---

## 2. Top-20 Portfolios by Profile

### ESG-First Portfolio

| Rank | Ticker | Sector | Score | ESG | Financial | Market |
|------|--------|--------|-------|-----|-----------|--------|
| 1 | GOOGL | Communication Services | 60.82 | 55.55 | 60.37 | 73.22 |
| 2 | TCS.NS | Technology | 59.98 | 58.90 | 71.74 | 44.23 |
| 3 | META | Communication Services | 58.80 | 57.26 | 64.86 | 49.44 |
| 4 | NVDA | Technology | 58.77 | 59.87 | 58.57 | 58.52 |
| 5 | V | Financial Services | 57.13 | 55.54 | 54.40 | 47.47 |
| 6 | MSFT | Technology | 56.00 | 57.02 | 60.68 | 31.38 |
| 7 | BHARTIARTL.NS | Communication Services | 55.88 | 51.95 | 70.19 | 43.45 |
| 8 | PM | Consumer Defensive | 55.76 | 50.14 | 52.92 | 59.39 |
| 9 | INFY.NS | Technology | 55.62 | 53.81 | 57.52 | 53.02 |
| 10 | AAPL | Technology | 55.56 | 55.67 | 58.11 | 52.68 |
| 11 | JNJ | Healthcare | 55.49 | 47.43 | 54.62 | 61.35 |
| 12 | PFE | Healthcare | 55.38 | 51.16 | 53.56 | 56.37 |
| 13 | ITC.NS | Consumer Defensive | 55.30 | 48.02 | 55.64 | 54.42 |
| 14 | WIPRO.NS | Technology | 54.96 | 58.91 | 52.56 | 41.69 |
| 15 | CSCO | Technology | 54.78 | 52.58 | 46.85 | 61.00 |
| 16 | SUNPHARMA.NS | Healthcare | 54.77 | 54.60 | 50.62 | 45.78 |
| 17 | SBIN.NS | Financial Services | 54.53 | 50.21 | 53.86 | 56.70 |
| 18 | AMZN | Consumer Cyclical | 54.13 | 48.67 | 49.07 | 53.44 |
| 19 | PG | Consumer Defensive | 53.90 | 49.32 | 48.23 | 59.91 |
| 20 | ABBV | Healthcare | 53.65 | 50.63 | 47.17 | 55.37 |

### Balanced Portfolio

| Rank | Ticker | Sector | Score | ESG | Financial | Market |
|------|--------|--------|-------|-----|-----------|--------|
| 1 | GOOGL | Communication Services | 61.41 | 55.55 | 60.37 | 73.22 |
| 2 | TCS.NS | Technology | 60.81 | 58.90 | 71.74 | 44.23 |
| 3 | META | Communication Services | 59.00 | 57.26 | 64.86 | 49.44 |
| 4 | BHARTIARTL.NS | Communication Services | 58.13 | 51.95 | 70.19 | 43.45 |
| 5 | NVDA | Technology | 57.14 | 59.87 | 58.57 | 58.52 |
| 6 | JNJ | Healthcare | 56.49 | 47.43 | 54.62 | 61.35 |
| 7 | ITC.NS | Consumer Defensive | 56.35 | 48.02 | 55.64 | 54.42 |
| 8 | RELIANCE.NS | Energy | 56.10 | 42.00 | 72.13 | 55.03 |
| 9 | PFE | Healthcare | 55.68 | 51.16 | 53.56 | 56.37 |
| 10 | PM | Consumer Defensive | 55.67 | 50.14 | 52.92 | 59.39 |
| 11 | V | Financial Services | 55.58 | 55.54 | 54.40 | 47.47 |
| 12 | MSFT | Technology | 55.35 | 57.02 | 60.68 | 31.38 |
| 13 | SBIN.NS | Financial Services | 55.32 | 50.21 | 53.86 | 56.70 |
| 14 | INFY.NS | Technology | 55.18 | 53.81 | 57.52 | 53.02 |
| 15 | AAPL | Technology | 55.06 | 55.67 | 58.11 | 52.68 |
| 16 | AMZN | Consumer Cyclical | 54.13 | 48.67 | 49.07 | 53.44 |
| 17 | MARUTI.NS | Consumer Cyclical | 53.94 | 45.54 | 52.64 | 57.10 |
| 18 | SUNPHARMA.NS | Healthcare | 53.60 | 54.60 | 50.62 | 45.78 |
| 19 | PG | Consumer Defensive | 53.48 | 49.32 | 48.23 | 59.91 |
| 20 | CSCO | Technology | 53.46 | 52.58 | 46.85 | 61.00 |

### Financial-First Portfolio

| Rank | Ticker | Sector | Score | ESG | Financial | Market |
|------|--------|--------|-------|-----|-----------|--------|
| 1 | GOOGL | Communication Services | 60.93 | 55.55 | 60.37 | 73.22 |
| 2 | TCS.NS | Technology | 60.02 | 58.90 | 71.74 | 44.23 |
| 3 | META | Communication Services | 58.51 | 57.26 | 64.86 | 49.44 |
| 4 | BHARTIARTL.NS | Communication Services | 58.34 | 51.95 | 70.19 | 43.45 |
| 5 | RELIANCE.NS | Energy | 57.34 | 42.00 | 72.13 | 55.03 |
| 6 | ITC.NS | Consumer Defensive | 55.99 | 48.02 | 55.64 | 54.42 |
| 7 | JNJ | Healthcare | 55.97 | 47.43 | 54.62 | 61.35 |
| 8 | NVDA | Technology | 55.94 | 59.87 | 58.57 | 58.52 |
| 9 | INFY.NS | Technology | 55.09 | 53.81 | 57.52 | 53.02 |
| 10 | PM | Consumer Defensive | 55.06 | 50.14 | 52.92 | 59.39 |
| 11 | PFE | Healthcare | 54.95 | 51.16 | 53.56 | 56.37 |
| 12 | V | Financial Services | 54.76 | 55.54 | 54.40 | 47.47 |
| 13 | SBIN.NS | Financial Services | 54.75 | 50.21 | 53.86 | 56.70 |
| 14 | MSFT | Technology | 54.28 | 57.02 | 60.68 | 31.38 |
| 15 | AAPL | Technology | 54.25 | 55.67 | 58.11 | 52.68 |
| 16 | MARUTI.NS | Consumer Cyclical | 53.81 | 45.54 | 52.64 | 57.10 |
| 17 | PG | Consumer Defensive | 53.31 | 49.32 | 48.23 | 59.91 |
| 18 | CSCO | Technology | 53.24 | 52.58 | 46.85 | 61.00 |
| 19 | AMZN | Consumer Cyclical | 53.14 | 48.67 | 49.07 | 53.44 |
| 20 | SUNPHARMA.NS | Healthcare | 52.45 | 54.60 | 50.62 | 45.78 |

---

## 3. Portfolio Overlap Analysis

### Pairwise Overlap

| Comparison | Shared Companies | Jaccard Similarity | Overlap % |
|------------|------------------|--------------------|-----------|
| ESG-First vs Balanced | 18 / 20 | 0.8182 (81.8%) | 90.0% |
| ESG-First vs Financial-First | 18 / 20 | 0.8182 (81.8%) | 90.0% |
| **Balanced vs Financial-First** | **20 / 20** | **1.0000 (100%)** | **100.0%** |

### Three-Way Analysis

- **Consensus companies (all three profiles):** 18 / 20 (90%)
- **Total unique companies across all top-20s:** 22
- **Balanced and Financial-First are identical sets** -- they select the same 20 companies (only ordering differs)

### Key Finding

The remarkably high overlap (90-100%) demonstrates that the multi-factor index produces a **robust core portfolio** that is largely invariant to investor preference weighting. The ESG weight shift from 10% to 35% only displaces 2 companies from the top-20. This is a crucial finding: **ESG integration does not fundamentally alter the investable universe when combined with financial quality screening**.

---

## 4. Risk-Return Characteristics per Profile Portfolio

| Metric | ESG-First | Balanced | Financial-First |
|--------|-----------|----------|-----------------|
| **Scores** | | | |
| Avg Preference Score | 56.06 | 56.09 | 55.61 |
| Avg ESG Composite | **53.36** | 52.26 | 52.26 |
| Avg Financial Score | 56.08 | **57.33** | **57.33** |
| Avg Market Score | 52.94 | **53.70** | **53.70** |
| **Return Proxies** | | | |
| Avg ROA | **12.48%** | 12.27% | 12.27% |
| Avg ROE | **44.10%** | 38.65% | 38.65% |
| Avg Net Margin | 24.10% | 24.02% | 24.02% |
| Avg Operating Margin | **36.49%** | 34.48% | 34.48% |
| Avg 6M Momentum | 11.39% | **11.50%** | **11.50%** |
| Avg 3M Momentum | 2.06% | **3.25%** | **3.25%** |
| Avg 1M Momentum | 2.18% | **3.17%** | **3.17%** |
| **Risk Proxies** | | | |
| Avg Volatility | **24.12%** | 24.30% | 24.30% |
| Avg Beta | 1.0147 | **0.9985** | **0.9985** |
| Avg Debt-to-Equity | 52.48 | **50.33** | **50.33** |
| **Combined Metrics** | | | |
| Pseudo Sharpe (6M Mom/Vol) | 0.4721 | **0.4731** | **0.4731** |
| Total Market Cap | $73.84T | $90.96T | $90.96T |
| Avg Market Cap | $3,691.8B | $4,547.8B | $4,547.8B |

### Interpretation

- **ESG-First portfolio** achieves higher ROA (12.48% vs 12.27%) and ROE (44.10% vs 38.65%), driven by WIPRO.NS and ABBV replacing RELIANCE.NS and MARUTI.NS. These ESG-tilted replacements happen to have higher profitability.
- **Balanced/Financial-First portfolios** (identical composition) achieve better momentum metrics and slightly lower volatility, with marginally better Pseudo Sharpe ratios (0.4731 vs 0.4721).
- The differences are economically small (< 1 percentage point on most metrics), reinforcing that profile choice does not create dramatically different risk-return trade-offs.
- All three portfolios have near-market beta (~1.0), acceptable volatility (~24%), and strong profitability (ROA > 12%).

---

## 5. Sector Diversification per Profile Portfolio

| Sector | ESG-First | Balanced | Financial-First |
|--------|-----------|----------|-----------------|
| Technology | 7 (35.0%) | 6 (30.0%) | 6 (30.0%) |
| Healthcare | 4 (20.0%) | 3 (15.0%) | 3 (15.0%) |
| Communication Services | 3 (15.0%) | 3 (15.0%) | 3 (15.0%) |
| Consumer Defensive | 3 (15.0%) | 3 (15.0%) | 3 (15.0%) |
| Financial Services | 2 (10.0%) | 2 (10.0%) | 2 (10.0%) |
| Consumer Cyclical | 1 (5.0%) | 2 (10.0%) | 2 (10.0%) |
| Energy | 0 (0.0%) | 1 (5.0%) | 1 (5.0%) |
| **N Sectors** | **6** | **7** | **7** |
| **HHI** | **0.2200** | **0.1800** | **0.1800** |
| **Effective N Sectors** | **4.55** | **5.56** | **5.56** |

### Key Findings

- **ESG-First has higher sector concentration** (HHI = 0.22 vs 0.18) with only 6 sectors represented and Technology at 35%
- **Balanced/Financial-First add Energy (RELIANCE.NS) and a second Consumer Cyclical (MARUTI.NS)**, improving diversification
- The ESG-First portfolio's Technology overweight (35%) vs Balanced (30%) reflects higher ESG scores in the tech sector
- All portfolios share the same Communication Services and Consumer Defensive weights
- **No portfolio includes** Basic Materials, Industrials, or Utilities sectors in the top-20

---

## 6. Profile Sensitivity Analysis

### Pairwise Rank Correlation

| Comparison | Spearman rho | p-value | Mean Rank Change | Median | Max | Companies >= 10 shift |
|------------|-------------|---------|------------------|--------|-----|----------------------|
| ESG-First vs Balanced | 0.9014 | 9.59e-23 | 5.70 | 5.0 | 25 | 10 |
| ESG-First vs Financial-First | 0.8540 | 4.26e-18 | 6.80 | 5.0 | 28 | 15 |
| **Balanced vs Financial-First** | **0.9861** | **7.20e-47** | **1.97** | **2.0** | **11** | **1** |

### Most Sensitive Companies (Largest Rank Swings)

| Ticker | Sector | ESG Rank | Bal Rank | Fin Rank | Max Swing |
|--------|--------|----------|----------|----------|-----------|
| RELIANCE.NS | Energy | 33 | 8 | 5 | **28** |
| ULTRACEMCO.NS | Basic Materials | 51 | 33 | 29 | 22 |
| TSLA | Consumer Cyclical | 37 | 55 | 56 | 19 |
| CVX | Energy | 55 | 47 | 36 | 19 |
| ASIANPAINT.NS | Basic Materials | 54 | 44 | 35 | 19 |
| ADBE | Technology | 28 | 45 | 45 | 17 |
| LIN | Basic Materials | 43 | 27 | 26 | 17 |
| KOTAKBANK.NS | Financial Services | 27 | 37 | 44 | 17 |
| XOM | Energy | 46 | 31 | 31 | 15 |
| AVGO | Technology | 32 | 42 | 47 | 15 |

### Most Stable Companies (Smallest Rank Swings)

| Ticker | Sector | ESG Rank | Bal Rank | Fin Rank | Max Swing |
|--------|--------|----------|----------|----------|-----------|
| GOOGL | Communication Services | 1 | 1 | 1 | **0** |
| META | Communication Services | 3 | 3 | 3 | **0** |
| NKE | Consumer Cyclical | 60 | 60 | 60 | **0** |
| TCS.NS | Technology | 2 | 2 | 2 | **0** |
| MA | Financial Services | 44 | 43 | 43 | 1 |
| UNH | Healthcare | 58 | 59 | 59 | 1 |
| DIS | Communication Services | 42 | 41 | 41 | 1 |
| ABT | Healthcare | 57 | 57 | 58 | 1 |
| JPM | Financial Services | 52 | 51 | 50 | 2 |
| PG | Consumer Defensive | 19 | 19 | 17 | 2 |

### Sensitivity Patterns

- **RELIANCE.NS is the most profile-sensitive company** (rank swing of 28): ranked #33 by ESG-First but #5 by Financial-First. It has the highest financial score (72.13) in the universe but the lowest ESG composite among top-20 candidates (42.00).
- **Energy and Basic Materials sectors show highest sensitivity** -- their ESG scores tend to be lower, creating large profile-dependent rank shifts.
- **TSLA shows reverse sensitivity** -- ranked #37 by ESG-First but drops to #56 by Financial-First, suggesting its high ESG reputation but weak financial fundamentals.
- **4 companies are perfectly stable** (rank swing = 0): GOOGL, META, TCS.NS, NKE -- these occupy extreme positions (top or bottom) regardless of weighting.
- **Balanced and Financial-First are nearly indistinguishable** (rho = 0.986, mean shift = 1.97 ranks), confirming that the 10-percentage-point shift between these profiles has minimal practical impact.

---

## 7. Consensus Picks (In Top-20 for ALL Profiles)

**18 out of 20 companies** are consensus picks across all three investor profiles:

| Ticker | Sector | ESG Score | Financial Score | ESG Rank | Bal Rank | Fin Rank |
|--------|--------|-----------|-----------------|----------|----------|----------|
| GOOGL | Communication Services | 55.55 | 60.37 | 1 | 1 | 1 |
| TCS.NS | Technology | 58.90 | 71.74 | 2 | 2 | 2 |
| META | Communication Services | 57.26 | 64.86 | 3 | 3 | 3 |
| BHARTIARTL.NS | Communication Services | 51.95 | 70.19 | 7 | 4 | 4 |
| NVDA | Technology | 59.87 | 58.57 | 4 | 5 | 8 |
| JNJ | Healthcare | 47.43 | 54.62 | 11 | 6 | 7 |
| ITC.NS | Consumer Defensive | 48.02 | 55.64 | 13 | 7 | 6 |
| PFE | Healthcare | 51.16 | 53.56 | 12 | 9 | 11 |
| PM | Consumer Defensive | 50.14 | 52.92 | 8 | 10 | 10 |
| V | Financial Services | 55.54 | 54.40 | 5 | 11 | 12 |
| MSFT | Technology | 57.02 | 60.68 | 6 | 12 | 14 |
| SBIN.NS | Financial Services | 50.21 | 53.86 | 17 | 13 | 13 |
| INFY.NS | Technology | 53.81 | 57.52 | 9 | 14 | 9 |
| AAPL | Technology | 55.67 | 58.11 | 10 | 15 | 15 |
| AMZN | Consumer Cyclical | 48.67 | 49.07 | 18 | 16 | 19 |
| SUNPHARMA.NS | Healthcare | 54.60 | 50.62 | 16 | 18 | 20 |
| PG | Consumer Defensive | 49.32 | 48.23 | 19 | 19 | 17 |
| CSCO | Technology | 52.58 | 46.85 | 15 | 20 | 18 |

### Characteristics of Consensus Picks

- **Geographic diversity:** 11 US companies + 7 Indian companies (5 .NS tickers + 2 dual-listed)
- **Sector coverage:** 6 sectors (Technology, Communication Services, Healthcare, Consumer Defensive, Financial Services, Consumer Cyclical)
- **Top-3 are perfectly stable:** GOOGL, TCS.NS, META hold ranks 1-3 across all profiles
- **Common thread:** These companies tend to score well on BOTH ESG and financial dimensions, making them robust to weighting changes

---

## 8. Profile-Specific Picks

### ESG-First Only (2 companies)

These companies appear in the ESG-First top-20 but NOT in Balanced or Financial-First:

| Ticker | Sector | ESG Score | Financial Score | ESG Rank | Bal Rank | Fin Rank | Why Selected |
|--------|--------|-----------|-----------------|----------|----------|----------|-------------|
| **WIPRO.NS** | Technology | **58.91** (Rank #2 overall) | 52.56 (Rank #19) | 14 | 21 | 21 | Extremely high ESG score; 0.35 ESG weight compensates for middling financial score |
| **ABBV** | Healthcare | 50.63 (Rank #28) | 47.17 (Rank #37) | 20 | 24 | 24 | Above-average ESG with strong operational/risk-adjusted scores; marginal inclusion |

### Balanced/Financial-First Only (2 companies, not in ESG-First)

| Ticker | Sector | ESG Score | Financial Score | ESG Rank | Bal Rank | Fin Rank | Why Selected |
|--------|--------|-----------|-----------------|----------|----------|----------|-------------|
| **RELIANCE.NS** | Energy | 42.00 (Rank #52) | **72.13** (Rank #1 overall) | 33 | 8 | 5 | Highest financial score in universe; low ESG drags it below ESG-First threshold |
| **MARUTI.NS** | Consumer Cyclical | 45.54 (Rank #44) | 52.64 (Rank #18) | 22 | 17 | 16 | Strong market score (57.10) and growth; below-median ESG excluded from ESG-First |

### Balanced-Only or Financial-First-Only: **None**

The Balanced and Financial-First profiles select the **exact same 20 companies** (Jaccard = 1.0). The 10-percentage-point ESG weight difference between them (15% vs 10%) is insufficient to alter the top-20 membership.

---

## 9. Summary Statistics for Paper Integration

### Key Quantitative Findings

| Finding | Value |
|---------|-------|
| Universe size | 60 companies |
| Portfolio size | 20 companies per profile |
| Consensus picks (all 3 profiles) | 18/20 (90%) |
| Unique companies across all profiles | 22 |
| ESG-First unique picks | 2 (WIPRO.NS, ABBV) |
| Financial-First unique picks | 0 |
| Balanced unique picks | 0 |
| Balanced vs Financial-First Jaccard | 1.000 (identical sets) |
| ESG-First vs Balanced Jaccard | 0.818 |
| ESG-First vs Financial-First Jaccard | 0.818 |
| Spearman rho (ESG vs Financial ranks) | 0.854 (p < 1e-17) |
| Spearman rho (Balanced vs Financial ranks) | 0.986 (p < 1e-46) |
| Max rank swing (any company) | 28 (RELIANCE.NS) |
| Perfectly stable companies (0 swing) | 4 (GOOGL, META, TCS.NS, NKE) |
| ESG-First HHI | 0.220 (6 sectors, effective N=4.55) |
| Balanced/Financial-First HHI | 0.180 (7 sectors, effective N=5.56) |
| Pseudo Sharpe (ESG-First) | 0.4721 |
| Pseudo Sharpe (Balanced/Financial-First) | 0.4731 |

### Implications for the Thesis

1. **ESG integration is not disruptive:** 90% portfolio overlap across profiles demonstrates that ESG-tilted weighting produces an investable universe largely consistent with financially-driven selection. This supports the thesis that ESG factors and financial quality are not adversarial.

2. **RELIANCE.NS is the canonical trade-off case:** It is the single most profile-sensitive company (28-rank swing), representing the archetype of high financial quality + low ESG scores. Its exclusion from the ESG-First portfolio is the clearest example of ESG screening's practical impact.

3. **Balanced and Financial-First are operationally equivalent:** With Jaccard = 1.0 (identical top-20 sets) and Spearman rho = 0.986, the practical difference between these profiles exists only in rank ordering, not portfolio composition. The 15% vs 10% ESG weight is below the threshold for membership change.

4. **Sector concentration is the primary ESG cost:** The ESG-First portfolio drops from 7 to 6 sectors and increases Technology weight from 30% to 35%, reducing diversification (HHI rises from 0.18 to 0.22). The exclusion of Energy (RELIANCE.NS) and reduction in Consumer Cyclical narrows sector exposure.

5. **Risk-return profiles are remarkably similar:** All three portfolios deliver nearly identical profitability (ROA ~12.3%), momentum (~11.4%), and volatility (~24.2%). The ESG-First portfolio has marginally higher ROA and ROE (driven by WIPRO.NS and ABBV), while Balanced/Financial-First have slightly better momentum.

6. **The top-3 are universally dominant:** GOOGL, TCS.NS, and META rank #1-3 across all profiles with zero rank deviation, suggesting these companies represent the "efficient frontier" of combined ESG + financial quality.

---

## Output Files

| File | Description |
|------|-------------|
| `top20_portfolios_by_profile.csv` | Full top-20 portfolios for each profile with scores |
| `portfolio_overlap_matrix.csv` | Pairwise Jaccard similarity and overlap counts |
| `profile_risk_return_characteristics.csv` | Risk-return metrics for each profile's portfolio |
| `sector_diversification_by_profile.csv` | Sector counts and HHI per profile |
| `ranking_sensitivity_all_companies.csv` | All 60 companies ranked with cross-profile rank differences |
| `consensus_and_profile_specific_picks.csv` | Classification of each top-20 company by profile membership |
| `profile_score_correlations.csv` | Pearson, Spearman, Kendall correlations between profiles |
| `full_rankings_comparison.csv` | Complete 60-company rankings across all profiles |
