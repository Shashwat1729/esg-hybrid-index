# Sector & Geography Analysis Report

**ResultsAnalyst-3 | Multi-Factor ESG Framework**
**Dataset:** 60 companies, 10 sectors, 2 countries (US: 40, India: 20)

---

## 1. Company Distribution by Sector and Country

| Sector | India | US | Total |
|---|---|---|---|
| Technology | 3 | 9 | **12** |
| Financial Services | 6 | 5 | **11** |
| Healthcare | 1 | 7 | **8** |
| Communication Services | 1 | 6 | **7** |
| Consumer Defensive | 3 | 4 | **7** |
| Consumer Cyclical | 2 | 4 | **6** |
| Energy | 1 | 2 | **3** |
| Basic Materials | 2 | 1 | **3** |
| Industrials | 1 | 1 | **2** |
| Utilities | 0 | 1 | **1** |
| **Total** | **20** | **40** | **60** |

**Key observations:**
- Technology (12) and Financial Services (11) dominate, jointly comprising 38% of the sample.
- India's Financial Services allocation (6/20 = 30%) substantially exceeds the US (5/40 = 12.5%), reflecting the sector's weight in Indian markets.
- Healthcare is heavily US-weighted (7:1), reflecting the US pharmaceutical/biotech ecosystem.
- Utilities has only 1 company (US), limiting statistical inference for this sector.

---

## 2. Mean Factor Scores by Sector (10 Core Factors)

### 2.1 Complete Factor Score Matrix

| Sector | E_score | S_score | G_score | Profitability | Growth | Efficiency | Stability | Valuation | Market | Financial |
|---|---|---|---|---|---|---|---|---|---|---|
| Technology | **57.59** | 53.64 | 52.45 | **0.635** | 49.82 | 0.029 | 54.57 | -0.120 | 45.03 | 53.99 |
| Financial Services | 55.96 | 49.14 | 51.21 | -0.071 | 51.51 | -0.150 | 34.28 | 0.301 | 48.70 | 48.30 |
| Communication Services | 53.48 | 51.52 | 46.41 | 0.239 | 52.43 | **0.464** | 51.29 | 0.312 | 51.10 | 54.88 |
| Healthcare | 50.36 | 49.02 | 49.27 | -0.207 | 46.76 | 0.203 | 54.61 | 0.151 | 47.37 | 48.37 |
| Consumer Defensive | 49.70 | 50.01 | 48.27 | -0.109 | 49.32 | -0.150 | **57.04** | -0.298 | 56.43 | 47.94 |
| Consumer Cyclical | 46.15 | 48.58 | 47.03 | -0.388 | 48.13 | -0.150 | 47.90 | -0.787 | 51.17 | 43.78 |
| Utilities | 38.63 | **58.06** | 42.47 | -0.009 | 42.03 | -0.150 | 44.15 | 0.393 | 45.15 | 46.57 |
| Industrials | 37.24 | 40.28 | 54.53 | -0.657 | 50.61 | -0.150 | 50.67 | 0.127 | 45.52 | 48.45 |
| Basic Materials | 31.83 | 44.19 | 49.58 | -0.305 | 48.63 | -0.104 | 55.89 | -0.205 | 57.36 | 47.10 |
| Energy | 27.59 | 50.08 | **55.98** | -0.510 | **54.97** | -0.150 | **57.45** | **0.503** | **58.95** | **56.09** |

### 2.2 Sector Leaders by Dimension

**ESG Leaders:**
- **Environmental (E_score):** Technology (57.59) >> Financial Services (55.96) >> Communication Services (53.48). Energy is last (27.59) -- a 30-point gap, the largest spread of any factor.
- **Social (S_score):** Utilities (58.06) > Technology (53.64) > Communication Services (51.52). Industrials trails (40.28).
- **Governance (G_score):** Energy (55.98) > Industrials (54.53) > Technology (52.45). Utilities trails (42.47).
- **ESG Composite:** Technology (54.67) > Financial Services (52.15) > Communication Services (50.68). Basic Materials (41.48) trails.

**Financial Leaders:**
- **Profitability:** Technology (0.635) leads decisively; Industrials (-0.657) trails.
- **Growth:** Energy (54.97) > Communication Services (52.43); Utilities (42.03) trails.
- **Stability:** Energy (57.45) and Consumer Defensive (57.04) lead; Financial Services (34.28) trails significantly due to inherent leverage.
- **Valuation:** Energy (0.503) and Utilities (0.393) show best value; Consumer Cyclical (-0.787) is most overvalued.
- **Financial Score:** Energy (56.09) > Communication Services (54.88) > Technology (54.00). Consumer Cyclical (43.78) trails.
- **Market Score:** Energy (58.95) > Basic Materials (57.36) > Consumer Defensive (56.43). Technology (45.03) trails due to high volatility.

### 2.3 Key Insight: The ESG-Financial Tension

A clear **inverse pattern** emerges between ESG leadership and traditional financial/market factors:
- **Technology** leads ESG (54.67 composite) but trails on market score (45.03).
- **Energy** leads financial (56.09) and market (58.95) scores but has the worst E_score (27.59).
- This tension is central to the multi-factor model's value proposition -- it can quantify and navigate this trade-off.

---

## 3. Sector-Level Composite Score Rankings

| Rank | Sector | ESG Composite | Financial | Market | Operational | Risk-Adjusted | Value | Overall Avg |
|---|---|---|---|---|---|---|---|---|
| 1 | Communication Services | 50.68 | 54.88 | 51.10 | 51.50 | 51.89 | 53.12 | **52.19** |
| 2 | Energy | 43.98 | 56.09 | 58.95 | 48.43 | 49.40 | 55.03 | **51.98** |
| 3 | Consumer Defensive | 49.38 | 47.94 | 56.43 | 48.97 | 54.74 | 47.34 | **50.80** |
| 4 | Basic Materials | 41.48 | 47.10 | 57.36 | 49.74 | 60.48 | 47.96 | **50.69** |
| 5 | Utilities | 46.58 | 46.57 | 45.15 | 54.97 | 54.71 | 53.93 | **50.32** |
| 6 | Technology | 54.67 | 54.00 | 45.03 | 50.50 | 47.31 | 48.80 | **50.05** |
| 7 | Financial Services | 52.15 | 48.30 | 48.70 | 47.28 | 50.63 | 53.22 | **50.05** |
| 8 | Healthcare | 49.56 | 48.37 | 47.37 | 52.91 | 47.03 | 52.17 | **49.57** |
| 9 | Industrials | 43.49 | 48.45 | 45.52 | 50.89 | 46.80 | 51.27 | **47.74** |
| 10 | Consumer Cyclical | 47.26 | 43.78 | 51.17 | 48.31 | 46.89 | 43.01 | **46.74** |

**Key findings:**
- **Communication Services** ranks #1 overall -- the only sector with no score below 50 across any composite. It offers the most balanced multi-factor profile.
- **Energy** is #2 overall, driven by exceptional financial (56.09) and market (58.95) scores, despite weak ESG (43.98).
- **Consumer Cyclical** ranks last overall (46.74), with the weakest financial score (43.78) and worst value score (43.01).
- **Basic Materials** has the highest risk-adjusted score (60.48) despite mediocre ESG, suggesting risk-return efficiency.
- **Technology** ranks only #6 overall despite leading in ESG -- penalized by low market score (45.03) and moderate value (48.80).

---

## 4. Within-Sector Score Dispersion

| Sector | E_score | S_score | G_score | Profitability | Growth | Efficiency | Stability | Valuation | Market | Financial | Mean Std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Communication Services | 10.47 | 7.14 | 3.61 | 0.58 | 8.52 | 1.13 | 6.35 | 0.42 | 10.25 | 10.04 | **5.85** |
| Consumer Cyclical | 9.26 | 6.18 | 5.73 | 0.47 | 6.09 | 0.00 | 13.45 | 1.14 | 8.45 | 6.31 | **5.71** |
| Technology | 6.56 | 5.12 | 7.86 | 0.75 | 8.31 | 0.36 | 5.34 | 0.70 | 11.18 | 7.60 | **5.38** |
| Energy | 4.65 | 10.58 | 8.13 | 0.04 | 6.61 | 0.00 | 1.12 | 0.05 | 3.53 | 13.92 | **4.86** |
| Healthcare | 4.60 | 7.06 | 7.34 | 0.40 | 5.10 | 0.65 | 2.79 | 0.58 | 14.24 | 4.02 | **4.68** |
| Basic Materials | 6.94 | 12.70 | 9.68 | 0.24 | 5.57 | 0.08 | 2.67 | 0.44 | 1.97 | 2.18 | **4.25** |
| Financial Services | 4.96 | 5.37 | 7.95 | 0.89 | 5.13 | 0.00 | 6.78 | 0.73 | 5.20 | 4.57 | **4.16** |
| Industrials | 0.86 | 3.40 | 12.07 | 0.06 | 9.30 | 0.00 | 4.17 | 0.09 | 1.42 | 6.99 | **3.84** |
| Consumer Defensive | 4.20 | 4.00 | 2.72 | 0.38 | 4.81 | 0.00 | 3.17 | 1.18 | 4.25 | 5.71 | **3.04** |

*(Utilities excluded: n=1, no std computable)*

### Most Heterogeneous (Stock-Picking Matters Most)
1. **Communication Services** (mean std = 5.85) -- enormous range in E_score (10.47), market (10.25), and financial (10.04). Active management has highest potential payoff here.
2. **Consumer Cyclical** (5.71) -- stability std of 13.45 is the single largest factor dispersion in the entire dataset.
3. **Technology** (5.38) -- market score std of 11.18 highlights large differences in trading characteristics.

### Most Homogeneous (Sector Allocation Drives Returns)
1. **Consumer Defensive** (mean std = 3.04) -- companies within this sector are remarkably similar across all factors. Sector-level allocation is more important than stock selection.
2. **Industrials** (3.84) -- though n=2 limits inference.
3. **Financial Services** (4.16) -- relatively uniform despite 11 companies, suggesting strong sector-level characteristics.

---

## 5. US vs India: Statistical Comparison

### 5.1 Mean Score Differences

| Factor | US Mean | India Mean | Difference | Cohen's d | Effect Size |
|---|---|---|---|---|---|
| E_score | 49.63 | 50.74 | -1.11 | -0.10 | Negligible |
| S_score | 49.95 | 50.11 | -0.16 | -0.02 | Negligible |
| G_score | 49.69 | 50.61 | -0.92 | -0.13 | Negligible |
| ESG_composite | 49.76 | 50.48 | -0.72 | -0.15 | Negligible |
| Profitability | 0.067 | -0.134 | 0.201 | 0.30 | Small |
| **Growth** | **47.06** | **55.55** | **-8.49** | **-1.64** | **Large** |
| **Efficiency** | **0.082** | **-0.143** | **0.225** | **0.53** | **Medium** |
| Stability | 50.87 | 48.26 | 2.60 | 0.24 | Small |
| Valuation | 0.004 | -0.008 | 0.012 | 0.02 | Negligible |
| Market | 49.94 | 50.11 | -0.17 | -0.02 | Negligible |
| Financial | 48.61 | 52.87 | -4.26 | -0.55 | Medium |
| **Operational** | **51.01** | **47.66** | **3.36** | **0.89** | **Large** |
| Risk-Adjusted | 49.96 | 50.42 | -0.46 | -0.07 | Negligible |
| Value | 50.36 | 50.03 | 0.33 | 0.04 | Negligible |

### 5.2 Statistical Test Results

| Factor | t-statistic | t p-value | Sig | MW U | MW p-value | MW Sig |
|---|---|---|---|---|---|---|
| **Growth** | -6.000 | <0.0001 | *** | 101.0 | <0.0001 | *** |
| **Operational** | 3.360 | 0.0016 | ** | 581.0 | 0.0046 | ** |
| **Efficiency** | 2.367 | 0.0229 | * | 463.0 | 0.1146 | ns |
| Financial | -1.866 | 0.0731 | † | 280.0 | 0.0609 | † |
| All other factors | | >0.10 | ns | | >0.10 | ns |

**Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10**

### 5.3 Interpretation

**Three statistically significant differences emerge (p<0.05):**

1. **Growth Score: India >> US (d = -1.64, p < 0.0001)** -- The largest and most significant difference. Indian companies show dramatically higher growth scores (55.55 vs 47.06), reflecting higher revenue growth trajectories in the emerging Indian market. This is a **large** effect confirmed by both parametric and non-parametric tests.

2. **Operational Score: US > India (d = 0.89, p = 0.0016)** -- US companies demonstrate significantly higher operational scores (51.01 vs 47.66), indicating more mature operational processes, supply chain management, and ESG integration into operations. A **large** effect.

3. **Efficiency Score: US > India (d = 0.53, p = 0.023)** -- US companies show moderately better efficiency (0.082 vs -0.143). This is a **medium** effect, though the Mann-Whitney test does not confirm significance (p = 0.11), suggesting the difference may be driven by outliers.

**One marginally significant difference (p<0.10):**

4. **Financial Score: India > US (d = -0.55, p = 0.073)** -- Indian companies trend toward higher financial scores (52.87 vs 48.61), a **medium** effect driven largely by higher growth scores feeding into financial composites.

**ESG scores show no significant country-level differences** -- E, S, G, and ESG composite are all statistically equivalent between US and India in this sample. This challenges the common assumption that developed-market companies inherently have stronger ESG profiles.

---

## 6. Sector x Country Interaction Analysis

### 6.1 Are Sector Rankings Consistent Across Countries?

| Score | Spearman rho | p-value | Interpretation |
|---|---|---|---|
| **ESG Composite** | **0.850** | **0.004** | **Strong consistency** |
| **Financial Score** | **0.733** | **0.025** | **Moderate consistency** |
| Value Score | 0.183 | 0.637 | **No consistency** |

**Critical finding:** ESG sector rankings are highly consistent across US and India (rho = 0.85, p = 0.004). Technology leads ESG in both countries, and traditional sectors (Energy, Basic Materials) trail in both. Financial score rankings also show significant cross-country consistency (rho = 0.73, p = 0.025). However, **value score rankings are completely inconsistent** across geographies (rho = 0.18, p = 0.64), meaning what constitutes "value" differs fundamentally between US and Indian markets.

### 6.2 Sector-Level Country Differences

**ESG Composite (US vs India mean difference):**
| Sector | US Mean | India Mean | Difference |
|---|---|---|---|
| Industrials | 39.88 | 47.10 | **India +7.23** |
| Healthcare | 48.84 | 54.60 | **India +5.76** |
| Basic Materials | 38.47 | 42.99 | India +4.52 |
| Technology | 53.82 | 57.21 | India +3.39 |
| Consumer Cyclical | 46.55 | 48.69 | India +2.14 |
| Communication Services | 50.46 | 51.95 | India +1.49 |
| Consumer Defensive | 49.83 | 48.78 | US +1.05 |
| Energy | 44.97 | 42.00 | US +2.97 |

Indian companies tend to score higher on ESG composite in most sectors, though none of the within-sector differences are statistically significant at p<0.05 (limited by small within-cell sample sizes).

**Financial Score (US vs India):**

The most dramatic divergences:
| Sector | India Financial | US Financial | India Premium |
|---|---|---|---|
| Energy | 72.13 | 48.07 | **+24.06** |
| Communication Services | 70.19 | 52.32 | **+17.86** |
| Industrials | 53.39 | 43.51 | +9.88 |
| Technology | 60.61 | 51.79 | +8.82 |

Indian companies show substantially higher financial scores in nearly every sector, driven primarily by the growth component. This suggests a systematic growth premium for Indian equities across all sectors.

### 6.3 Implications for Portfolio Construction

The interaction analysis reveals that:
1. **Sector allocation for ESG** works similarly in both geographies -- Technology and Financial Services lead ESG in both US and India.
2. **Sector allocation for financial performance** shows larger country-dependent variation.
3. **Value identification requires geography-specific models** -- value factor rankings are uncorrelated across countries.

---

## 7. Sector-Specific Investment Opportunities

### 7.1 Dual Opportunity Companies (High ESG + High Financial + Above-Sector Value)

Only **6 companies** satisfy all three criteria simultaneously:

| Ticker | Sector | Country | ESG Composite | Financial Score | Value Score |
|---|---|---|---|---|---|
| PFE | Healthcare | US | 51.16 | 53.56 | **56.89** |
| WIPRO.NS | Technology | India | **58.91** | 52.56 | 55.62 |
| ADBE | Technology | US | 57.15 | 51.76 | 52.88 |
| INFY.NS | Technology | India | 53.81 | **57.52** | 52.66 |
| MSFT | Technology | US | 57.02 | 60.68 | 51.96 |
| TCS.NS | Technology | India | 58.90 | **71.74** | 51.49 |

**Technology dominates dual opportunities** (5 of 6), with balanced India-US representation (3 each). TCS.NS shows the highest financial score (71.74) of any dual-opportunity company, while WIPRO.NS leads on ESG (58.91).

### 7.2 ESG-Only Opportunities (14 companies)

Top ESG opportunities not in the dual list include: KOTAKBANK.NS, WFC, AXISBANK.NS, BAC, CSCO, CRM, TMO, JPM -- predominantly Financial Services and Technology.

### 7.3 Sector Opportunity Map

| Sector | n | Mean ESG | Mean Financial | Mean Value | Dual Opps | ESG Opps | Fin Opps |
|---|---|---|---|---|---|---|---|
| **Technology** | 12 | **54.67** | **54.00** | 48.80 | **5** | 7 | 6 |
| Financial Services | 11 | 52.15 | 48.30 | 53.22 | 0 | 5 | 3 |
| Healthcare | 8 | 49.56 | 48.37 | 52.17 | 1 | 2 | 2 |
| Communication Services | 7 | 50.68 | 54.88 | 53.12 | 0 | 0 | 0 |
| Consumer Defensive | 7 | 49.38 | 47.94 | 47.34 | 0 | 0 | 3 |
| Consumer Cyclical | 6 | 47.26 | 43.78 | 43.01 | 0 | 0 | 2 |
| Energy | 3 | 43.98 | 56.09 | 55.03 | 0 | 0 | 0 |
| Basic Materials | 3 | 41.48 | 47.10 | 47.96 | 0 | 0 | 1 |
| Industrials | 2 | 43.49 | 48.45 | 51.27 | 0 | 0 | 1 |
| Utilities | 1 | 46.58 | 46.57 | 53.93 | 0 | 0 | 0 |

**Technology is the clear winner** for multi-factor investing, offering the most opportunities across all lenses. Communication Services has strong averages but no individual standouts above the opportunity thresholds, suggesting the sector is uniformly good but lacks outperformers. Consumer Cyclical offers the fewest opportunities -- the weakest sector for multi-factor strategies.

### 7.4 Undervalued Sector Plays

Sectors with high financial/ESG scores but below-average value scores (potential future re-rating):
- **Communication Services:** Strong across all composites (overall #1) but mean value score (53.12) is not the highest -- individual stock selection needed.
- **Technology:** Highest ESG and strong financial, but value score (48.80) ranks only #7 -- growth premium compression may offer future entry points.

---

## 8. Preference Profile Analysis

### By Sector
| Sector | ESG-First | Balanced | Financial-First |
|---|---|---|---|
| Communication Services | 53.76 | **54.01** | **53.81** |
| Technology | **53.97** | 52.88 | 52.30 |
| Consumer Defensive | 52.95 | 52.42 | 52.02 |
| Healthcare | 51.77 | 51.12 | 50.54 |
| Energy | 49.19 | 51.94 | 52.57 |

Communication Services scores highest under both Balanced and Financial-First preferences. Technology leads under ESG-First. Energy improves notably from ESG-First (49.19) to Financial-First (52.57), a +3.38 shift -- the largest preference-dependent swing of any sector.

### By Country
| Country | ESG-First | Balanced | Financial-First |
|---|---|---|---|
| India | 52.48 | 52.73 | 52.33 |
| US | 51.65 | 51.03 | 50.62 |

Indian companies slightly outperform US companies under all three preference schemes. The gap widens from ESG-First (0.84) to Financial-First (1.71), consistent with India's growth advantage feeding into financial preferences.

---

## Summary of Key Findings for Paper Integration

### Finding 1: ESG-Financial Trade-off is Sector-Dependent
Technology leads ESG (54.67) but ranks #6 overall; Energy leads financial (56.09) but trails on ESG (43.98). The 30-point E_score gap between Technology (57.59) and Energy (27.59) is the largest factor spread in the dataset. Multi-factor frameworks are essential to navigate this trade-off.

### Finding 2: Communication Services is the Most Balanced Sector
Highest overall composite average (52.19), the only sector with no sub-50 composite score. This represents the "sweet spot" for investors seeking both ESG and financial quality.

### Finding 3: India's Growth Premium is the Dominant Country Effect
Growth score difference (India 55.55 vs US 47.06) produces a Cohen's d of -1.64 (p < 0.0001), the largest effect in the entire analysis. This drives Indian companies' higher financial scores across nearly all sectors.

### Finding 4: ESG Sector Rankings are Geography-Invariant
Cross-country Spearman rho of 0.85 (p = 0.004) for ESG composite rankings confirms that ESG quality is fundamentally sector-driven, not geography-driven. However, value identification (rho = 0.18) requires geography-specific approaches.

### Finding 5: Sector Dispersion Varies Dramatically
Consumer Defensive (std = 3.04) is 93% more homogeneous than Communication Services (std = 5.85). This implies passive/sector-allocation approaches work for defensive sectors, while active stock selection is critical in Communication Services and Technology.

### Finding 6: Technology Dominates Multi-Factor Opportunities
5 of 6 dual-opportunity companies are in Technology, with balanced US-India representation. TCS.NS achieves the highest financial score (71.74) among dual opportunities, demonstrating that emerging-market tech can simultaneously deliver ESG quality and financial performance.

### Finding 7: Financial Services Shows the Weakest Stability
Financial Services has the lowest stability score (34.28), 23 points below the leader (Energy, 57.45). This reflects inherent leverage and regulatory sensitivity, important for risk-adjusted portfolio construction.

---

*Analysis tables saved to `reports/analysis/` directory.*
*Generated by ResultsAnalyst-3 (Sector & Geography)*
