#!/usr/bin/env python3
"""
Sector & Geography Analysis for Multi-Factor ESG Framework
ResultsAnalyst-3: Comprehensive sector and geographic pattern analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
DATA_PATH = "data/processed/indexed_data.csv"
OUT_DIR = "reports/analysis"

df = pd.read_csv(DATA_PATH)
print(f"Dataset: {df.shape[0]} companies, {df.shape[1]} columns")
print(f"Sectors: {df['sector'].nunique()} | Countries: {df['country'].nunique()}")
print(f"Sectors: {sorted(df['sector'].unique())}")
print(f"Countries: {sorted(df['country'].unique())}")

# Define factor score columns
FACTOR_SCORES = [
    'E_score', 'S_score', 'G_score', 'ESG_composite',
    'financial_score', 'growth_score', 'operational_score',
    'stability_score', 'value_score',
    'market_score',
    'risk_adjusted_score'
]

# Core 10 factors for the multi-factor model
CORE_10 = [
    'E_score', 'S_score', 'G_score',
    'financial_score', 'growth_score', 'operational_score',
    'stability_score', 'value_score',
    'market_score', 'risk_adjusted_score'
]

COMPOSITE_SCORES = ['ESG_composite', 'financial_score', 'market_score',
                    'operational_score', 'risk_adjusted_score', 'value_score']

PREFERENCE_COLS = ['pref_esg_first', 'pref_balanced', 'pref_financial_first']

# ============================================================
# 1. COMPANY DISTRIBUTION: Sector × Country
# ============================================================
print("\n" + "="*80)
print("1. COMPANY DISTRIBUTION BY SECTOR AND COUNTRY")
print("="*80)

cross_tab = pd.crosstab(df['sector'], df['country'], margins=True)
print(cross_tab.to_string())
cross_tab.to_csv(f"{OUT_DIR}/sector_country_distribution.csv")

# Sector summary
sector_counts = df['sector'].value_counts().sort_values(ascending=False)
print(f"\nSector distribution:\n{sector_counts.to_string()}")

# Country summary
country_counts = df['country'].value_counts()
print(f"\nCountry distribution:\n{country_counts.to_string()}")

# ============================================================
# 2. MEAN FACTOR SCORES BY SECTOR (All 10 Core Factors)
# ============================================================
print("\n" + "="*80)
print("2. MEAN FACTOR SCORES BY SECTOR")
print("="*80)

sector_means = df.groupby('sector')[CORE_10].mean()
sector_means_rounded = sector_means.round(4)
print(sector_means_rounded.to_string())
sector_means_rounded.to_csv(f"{OUT_DIR}/sector_mean_factor_scores.csv")

# Identify which sectors lead on each factor
print("\n--- Sector Leaders by Factor ---")
leaders = {}
for col in CORE_10:
    best_sector = sector_means[col].idxmax()
    best_val = sector_means[col].max()
    worst_sector = sector_means[col].idxmin()
    worst_val = sector_means[col].min()
    leaders[col] = {'best': best_sector, 'best_val': best_val,
                    'worst': worst_sector, 'worst_val': worst_val}
    print(f"  {col:25s} BEST: {best_sector:30s} ({best_val:.4f})  WORST: {worst_sector:30s} ({worst_val:.4f})")

leaders_df = pd.DataFrame(leaders).T
leaders_df.to_csv(f"{OUT_DIR}/sector_factor_leaders.csv")

# ESG leaders
print("\n--- ESG Sub-Score Leaders ---")
for col in ['E_score', 'S_score', 'G_score', 'ESG_composite']:
    ranked = sector_means[col].sort_values(ascending=False) if col in sector_means.columns else None
    if ranked is not None:
        print(f"\n  {col} Ranking:")
        for i, (sec, val) in enumerate(ranked.items(), 1):
            print(f"    {i}. {sec:30s} {val:.4f}")

# Financial leaders
print("\n--- Financial Score Leaders ---")
if 'financial_score' in sector_means.columns:
    ranked = sector_means['financial_score'].sort_values(ascending=False)
    print(f"\n  financial_score Ranking:")
    for i, (sec, val) in enumerate(ranked.items(), 1):
        print(f"    {i}. {sec:30s} {val:.4f}")

# ============================================================
# 3. SECTOR-LEVEL COMPOSITE SCORE RANKINGS
# ============================================================
print("\n" + "="*80)
print("3. SECTOR-LEVEL COMPOSITE SCORE RANKINGS")
print("="*80)

all_composite = ['ESG_composite', 'financial_score', 'market_score',
                 'operational_score', 'risk_adjusted_score', 'value_score']
# Filter to available columns
available_composite = [c for c in all_composite if c in df.columns]

sector_composite = df.groupby('sector')[available_composite].mean().round(4)
# Add overall rank based on average of all composites
sector_composite['overall_avg'] = sector_composite.mean(axis=1)
sector_composite = sector_composite.sort_values('overall_avg', ascending=False)

print(sector_composite.to_string())
sector_composite.to_csv(f"{OUT_DIR}/sector_composite_rankings.csv")

# Rank each composite
print("\n--- Rankings per Composite ---")
for col in available_composite:
    ranked = sector_composite[col].sort_values(ascending=False)
    print(f"\n  {col}:")
    for i, (sec, val) in enumerate(ranked.items(), 1):
        print(f"    {i}. {sec:30s} {val:.4f}")

# ============================================================
# 4. WITHIN-SECTOR SCORE DISPERSION (STD)
# ============================================================
print("\n" + "="*80)
print("4. WITHIN-SECTOR SCORE DISPERSION")
print("="*80)

sector_std = df.groupby('sector')[CORE_10].std().round(4)
sector_std['mean_dispersion'] = sector_std.mean(axis=1)
sector_std = sector_std.sort_values('mean_dispersion', ascending=False)

print(sector_std.to_string())
sector_std.to_csv(f"{OUT_DIR}/sector_score_dispersion.csv")

print("\n--- Most Heterogeneous Sectors (highest dispersion) ---")
for i, (sec, val) in enumerate(sector_std['mean_dispersion'].head(5).items(), 1):
    print(f"  {i}. {sec:30s} mean_std={val:.4f}")

print("\n--- Most Homogeneous Sectors (lowest dispersion) ---")
for i, (sec, val) in enumerate(sector_std['mean_dispersion'].sort_values().head(5).items(), 1):
    print(f"  {i}. {sec:30s} mean_std={val:.4f}")

# Also compute CV (coefficient of variation) for better comparability
sector_cv = (sector_std[CORE_10].div(sector_means[CORE_10].abs().replace(0, np.nan))).round(4)
sector_cv['mean_cv'] = sector_cv.mean(axis=1)
sector_cv.to_csv(f"{OUT_DIR}/sector_coefficient_of_variation.csv")

# ============================================================
# 5. US vs INDIA: Statistical Comparison
# ============================================================
print("\n" + "="*80)
print("5. US vs INDIA: STATISTICAL COMPARISON")
print("="*80)

us_data = df[df['country'] == 'US']
india_data = df[df['country'] == 'India']

print(f"\nUS companies: {len(us_data)}")
print(f"India companies: {len(india_data)}")

# Mean comparison
country_means = df.groupby('country')[CORE_10 + available_composite + PREFERENCE_COLS].mean().round(4)
print("\n--- Country Mean Scores ---")
print(country_means.T.to_string())
country_means.T.to_csv(f"{OUT_DIR}/country_mean_scores.csv")

# Statistical tests
test_results = []
all_test_cols = CORE_10 + available_composite
if PREFERENCE_COLS[0] in df.columns:
    all_test_cols += PREFERENCE_COLS

for col in all_test_cols:
    us_vals = us_data[col].dropna()
    india_vals = india_data[col].dropna()
    
    if len(us_vals) < 3 or len(india_vals) < 3:
        continue
    
    # T-test (Welch's)
    t_stat, t_pval = stats.ttest_ind(us_vals, india_vals, equal_var=False)
    
    # Mann-Whitney U
    u_stat, mw_pval = stats.mannwhitneyu(us_vals, india_vals, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((us_vals.std()**2 + india_vals.std()**2) / 2)
    cohens_d = (us_vals.mean() - india_vals.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Significance markers
    sig_t = '***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else '†' if t_pval < 0.1 else 'ns'
    sig_mw = '***' if mw_pval < 0.001 else '**' if mw_pval < 0.01 else '*' if mw_pval < 0.05 else '†' if mw_pval < 0.1 else 'ns'
    
    test_results.append({
        'factor': col,
        'US_mean': us_vals.mean(),
        'India_mean': india_vals.mean(),
        'diff': us_vals.mean() - india_vals.mean(),
        't_stat': t_stat,
        't_pval': t_pval,
        't_sig': sig_t,
        'U_stat': u_stat,
        'MW_pval': mw_pval,
        'MW_sig': sig_mw,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small' if abs(cohens_d) >= 0.2 else 'negligible'
    })

test_df = pd.DataFrame(test_results)
test_df = test_df.round(4)
print("\n--- Statistical Tests: US vs India ---")
print(test_df.to_string(index=False))
test_df.to_csv(f"{OUT_DIR}/us_vs_india_statistical_tests.csv", index=False)

# Summary of significant differences
sig_factors = test_df[test_df['t_pval'] < 0.05]
print(f"\n--- Significant Differences (p<0.05) ---")
if len(sig_factors) > 0:
    for _, row in sig_factors.iterrows():
        direction = "US > India" if row['diff'] > 0 else "India > US"
        print(f"  {row['factor']:25s} {direction}  d={row['cohens_d']:.3f} ({row['effect_size']}), t_p={row['t_pval']:.4f}, MW_p={row['MW_pval']:.4f}")
else:
    print("  No statistically significant differences at p<0.05")

marginal_factors = test_df[(test_df['t_pval'] >= 0.05) & (test_df['t_pval'] < 0.1)]
if len(marginal_factors) > 0:
    print(f"\n--- Marginally Significant (0.05 <= p < 0.1) ---")
    for _, row in marginal_factors.iterrows():
        direction = "US > India" if row['diff'] > 0 else "India > US"
        print(f"  {row['factor']:25s} {direction}  d={row['cohens_d']:.3f} ({row['effect_size']}), t_p={row['t_pval']:.4f}")

# ============================================================
# 6. SECTOR × COUNTRY INTERACTION
# ============================================================
print("\n" + "="*80)
print("6. SECTOR × COUNTRY INTERACTION ANALYSIS")
print("="*80)

# Sector × Country means for key composites
interaction_scores = ['ESG_composite', 'financial_score', 'market_score', 'value_score']
interaction_avail = [c for c in interaction_scores if c in df.columns]

for score in interaction_avail:
    print(f"\n--- {score} by Sector × Country ---")
    pivot = df.pivot_table(values=score, index='sector', columns='country', aggfunc='mean').round(4)
    pivot['diff_US_India'] = pivot.get('US', 0) - pivot.get('India', 0)
    print(pivot.to_string())
    pivot.to_csv(f"{OUT_DIR}/interaction_{score}.csv")

# Two-way ANOVA-like analysis: test if sector effects are consistent across countries
# For sectors with companies in both countries
print("\n--- Sector × Country Interaction: Consistency Tests ---")
sectors_both = []
for sec in df['sector'].unique():
    n_us = len(df[(df['sector'] == sec) & (df['country'] == 'US')])
    n_india = len(df[(df['sector'] == sec) & (df['country'] == 'India')])
    if n_us >= 2 and n_india >= 2:
        sectors_both.append(sec)
    elif n_us >= 1 and n_india >= 1:
        sectors_both.append(sec)  # include even with 1, note it

print(f"Sectors with companies in both countries: {sectors_both}")

interaction_results = []
for score in ['ESG_composite', 'financial_score', 'value_score']:
    if score not in df.columns:
        continue
    for sec in sectors_both:
        us_vals = df[(df['sector'] == sec) & (df['country'] == 'US')][score].dropna()
        india_vals = df[(df['sector'] == sec) & (df['country'] == 'India')][score].dropna()
        if len(us_vals) >= 2 and len(india_vals) >= 2:
            t_stat, t_pval = stats.ttest_ind(us_vals, india_vals, equal_var=False)
            interaction_results.append({
                'score': score,
                'sector': sec,
                'n_US': len(us_vals),
                'n_India': len(india_vals),
                'US_mean': us_vals.mean(),
                'India_mean': india_vals.mean(),
                'diff': us_vals.mean() - india_vals.mean(),
                't_stat': t_stat,
                'p_val': t_pval,
                'sig': '***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else '†' if t_pval < 0.1 else 'ns'
            })
        else:
            interaction_results.append({
                'score': score,
                'sector': sec,
                'n_US': len(us_vals),
                'n_India': len(india_vals),
                'US_mean': us_vals.mean() if len(us_vals) > 0 else np.nan,
                'India_mean': india_vals.mean() if len(india_vals) > 0 else np.nan,
                'diff': (us_vals.mean() - india_vals.mean()) if len(us_vals) > 0 and len(india_vals) > 0 else np.nan,
                't_stat': np.nan,
                'p_val': np.nan,
                'sig': 'n/a (small n)'
            })

interaction_df = pd.DataFrame(interaction_results).round(4)
print(interaction_df.to_string(index=False))
interaction_df.to_csv(f"{OUT_DIR}/sector_country_interaction_tests.csv", index=False)

# Rank correlation: do sector rankings hold across countries?
print("\n--- Sector Rank Consistency Across Countries ---")
for score in ['ESG_composite', 'financial_score', 'value_score']:
    if score not in df.columns:
        continue
    us_sector_means = df[df['country'] == 'US'].groupby('sector')[score].mean()
    india_sector_means = df[df['country'] == 'India'].groupby('sector')[score].mean()
    common_sectors = us_sector_means.index.intersection(india_sector_means.index)
    if len(common_sectors) >= 3:
        rho, p_rho = stats.spearmanr(
            us_sector_means.loc[common_sectors],
            india_sector_means.loc[common_sectors]
        )
        print(f"  {score}: Spearman ρ = {rho:.4f}, p = {p_rho:.4f} (n={len(common_sectors)} sectors)")
    else:
        print(f"  {score}: Too few common sectors ({len(common_sectors)}) for rank correlation")

# ============================================================
# 7. SECTOR-SPECIFIC INVESTMENT OPPORTUNITIES
# ============================================================
print("\n" + "="*80)
print("7. SECTOR-SPECIFIC INVESTMENT OPPORTUNITIES")
print("="*80)

# Identify companies with high composite + undervalued sector position
# We'll use value_score and ESG_composite and financial_score

if 'value_score' in df.columns and 'ESG_composite' in df.columns:
    # Compute sector averages for value_score
    sector_value_avg = df.groupby('sector')['value_score'].mean()
    sector_esg_avg = df.groupby('sector')['ESG_composite'].mean()
    sector_financial_avg = df.groupby('sector')['financial_score'].mean()
    
    # Companies above sector average in value + above median ESG
    esg_median = df['ESG_composite'].median()
    fin_median = df['financial_score'].median()
    value_median = df['value_score'].median()
    
    # Approach 1: High composite, undervalued (high value_score = undervalued relative to fundamentals)
    df['sector_value_avg'] = df['sector'].map(sector_value_avg)
    df['sector_esg_avg'] = df['sector'].map(sector_esg_avg)
    df['sector_financial_avg'] = df['sector'].map(sector_financial_avg)
    
    # Above median ESG + above sector avg value = ESG opportunity
    df['esg_opportunity'] = (df['ESG_composite'] > esg_median) & (df['value_score'] > df['sector_value_avg'])
    # Above median financial + above sector avg value = financial opportunity
    df['fin_opportunity'] = (df['financial_score'] > fin_median) & (df['value_score'] > df['sector_value_avg'])
    # Both = dual opportunity
    df['dual_opportunity'] = df['esg_opportunity'] & df['fin_opportunity']
    
    print("\n--- ESG Opportunities (High ESG + Above-Sector Value) ---")
    esg_opps = df[df['esg_opportunity']].sort_values('ESG_composite', ascending=False)
    opp_cols = ['ticker', 'company_name', 'sector', 'country', 'ESG_composite', 'financial_score', 'value_score']
    if len(esg_opps) > 0:
        print(esg_opps[opp_cols].to_string(index=False))
    
    print("\n--- Financial Opportunities (High Financial + Above-Sector Value) ---")
    fin_opps = df[df['fin_opportunity']].sort_values('financial_score', ascending=False)
    if len(fin_opps) > 0:
        print(fin_opps[opp_cols].to_string(index=False))
    
    print("\n--- Dual Opportunities (High ESG + High Financial + Above-Sector Value) ---")
    dual_opps = df[df['dual_opportunity']].sort_values('value_score', ascending=False)
    if len(dual_opps) > 0:
        print(dual_opps[opp_cols].to_string(index=False))
    
    # Save opportunities
    esg_opps[opp_cols].to_csv(f"{OUT_DIR}/esg_opportunities.csv", index=False)
    fin_opps[opp_cols].to_csv(f"{OUT_DIR}/financial_opportunities.csv", index=False)
    dual_opps[opp_cols].to_csv(f"{OUT_DIR}/dual_opportunities.csv", index=False)

# Sector-level opportunity summary
print("\n--- Sector-Level Opportunity Map ---")
sector_opp = df.groupby('sector').agg(
    n_companies=('ticker', 'count'),
    mean_ESG=('ESG_composite', 'mean'),
    mean_Financial=('financial_score', 'mean'),
    mean_Value=('value_score', 'mean'),
    n_esg_opp=('esg_opportunity', 'sum'),
    n_fin_opp=('fin_opportunity', 'sum'),
    n_dual_opp=('dual_opportunity', 'sum')
).round(4)
sector_opp = sector_opp.sort_values('mean_Value', ascending=False)
print(sector_opp.to_string())
sector_opp.to_csv(f"{OUT_DIR}/sector_opportunity_map.csv")

# ============================================================
# ADDITIONAL: Preference score analysis by sector/country
# ============================================================
print("\n" + "="*80)
print("ADDITIONAL: PREFERENCE SCORES BY SECTOR AND COUNTRY")
print("="*80)

pref_available = [c for c in PREFERENCE_COLS if c in df.columns]
if pref_available:
    pref_by_sector = df.groupby('sector')[pref_available].mean().round(4)
    pref_by_sector = pref_by_sector.sort_values(pref_available[0], ascending=False)
    print("\n--- Preference Scores by Sector ---")
    print(pref_by_sector.to_string())
    pref_by_sector.to_csv(f"{OUT_DIR}/preference_scores_by_sector.csv")
    
    pref_by_country = df.groupby('country')[pref_available].mean().round(4)
    print("\n--- Preference Scores by Country ---")
    print(pref_by_country.to_string())

# ============================================================
# SUMMARY STATISTICS TABLE
# ============================================================
print("\n" + "="*80)
print("SUMMARY: KEY FINDINGS")
print("="*80)

# Overall dataset stats
print(f"\n1. Dataset: {len(df)} companies across {df['sector'].nunique()} sectors, {df['country'].nunique()} countries")
print(f"   US: {len(us_data)} companies | India: {len(india_data)} companies")

# Top sectors
if 'value_score' in df.columns:
    top_value_sector = sector_composite['value_score'].idxmax() if 'value_score' in sector_composite.columns else 'N/A'
    print(f"\n2. Top sector by value_score: {top_value_sector}")

top_esg_sector = sector_means['ESG_composite'].idxmax() if 'ESG_composite' in sector_means.columns else 'N/A'
top_fin_sector = sector_means['financial_score'].idxmax() if 'financial_score' in sector_means.columns else 'N/A'
print(f"   Top sector by ESG: {top_esg_sector}")
print(f"   Top sector by Financial: {top_fin_sector}")

# Significant US vs India differences
n_sig = len(test_df[test_df['t_pval'] < 0.05])
print(f"\n3. US vs India: {n_sig} factors show significant differences (p<0.05)")

# Opportunity count
if 'dual_opportunity' in df.columns:
    print(f"\n4. Investment Opportunities: {df['dual_opportunity'].sum()} dual (ESG+Financial+Value) opportunities identified")

print("\n--- ANALYSIS COMPLETE ---")
