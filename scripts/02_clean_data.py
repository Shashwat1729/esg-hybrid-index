"""
Step 02: Clean and Standardize Data
=====================================
Reads raw combined data, handles missing values, detects and treats outliers,
imputes via sector-median, applies log transforms to skewed variables,
properly encodes ordinal and binary variables, and outputs a clean dataset.

Key methodological improvements:
  - Variable type classification (continuous, binary, ordinal, bounded-pct, ratio)
  - Binary/ordinal variables get appropriate encoding (not z-scored)
  - Multi-method outlier detection (IQR, Modified Z-score / MAD, Mahalanobis)
  - Adaptive winsorization thresholds based on variable type
  - Comprehensive outlier report for transparency

Input:  data/raw/combined_raw.csv
Output: data/processed/clean_data.csv
        reports/tables/data_quality_before.csv
        reports/tables/data_quality_after.csv
        reports/tables/outlier_report.csv
        reports/tables/variable_type_classification.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
ID_COLS = ["ticker", "company_name", "currency", "sector", "industry", "country"]

ESG_ENV_COLS = [
    "scope1_emissions", "scope2_emissions", "scope3_emissions",
    "emissions_intensity", "renewable_energy_pct", "energy_efficiency",
    "water_usage_intensity", "waste_recycling_pct", "carbon_reduction_target",
    "environmental_fines",
]
ESG_SOC_COLS = [
    "employee_turnover", "gender_diversity_pct", "women_management_pct",
    "pay_gap_ratio", "injury_rate", "safety_training_hours",
    "employee_satisfaction", "community_investment_pct",
    "supply_chain_audit_pct", "human_rights_policy",
]
ESG_GOV_COLS = [
    "board_independence_pct", "board_diversity_pct", "board_size",
    "exec_comp_esg_linked", "ceo_pay_ratio", "shareholder_rights_score",
    "ethics_compliance_score", "anti_corruption_policy",
    "data_privacy_score", "tax_transparency_score",
    "esg_controversy_score", "esg_risk_rating",
]
ESG_COLS = ESG_ENV_COLS + ESG_SOC_COLS + ESG_GOV_COLS

FINANCIAL_COLS = [
    "market_cap", "total_revenue", "ebitda", "net_income", "gross_profit",
    "total_debt", "total_cash",
    "roa", "roe", "debt_to_equity", "current_ratio", "quick_ratio",
    "free_cashflow", "operating_cashflow",
    "trailing_pe", "forward_pe", "price_to_book", "price_to_sales",
    "enterprise_to_revenue", "enterprise_to_ebitda",
    "dividend_yield", "payout_ratio",
    "revenue_growth", "earnings_growth", "earnings_quarterly_growth",
    "profit_margins", "gross_margins", "operating_margins",
    "net_margin", "operating_margin", "gross_margin",
    "debt_to_ebitda", "cash_flow_to_debt", "fcf_margin",
]

MARKET_COLS = [
    "price", "avg_daily_volume", "avg_daily_volume_30d", "avg_daily_volume_90d",
    "price_volatility", "price_volatility_30d",
    "price_momentum_1m", "price_momentum_3m", "price_momentum_6m", "price_momentum_12m",
    "beta", "free_float_pct", "bid_ask_spread",
    "max_drawdown_1y", "sharpe_ratio_1y", "sortino_ratio_1y",
    "avg_daily_return", "return_skewness", "return_kurtosis",
    "amihud_illiquidity",
    "52_week_high", "52_week_low", "50d_avg", "200d_avg",
    "pct_from_52w_high",
]

OPERATIONAL_COLS = [
    "r_d_expenditure", "r_d_intensity", "employees",
    "revenue_per_employee", "market_share",
]

ALL_NUMERIC = list(set(ESG_COLS + FINANCIAL_COLS + MARKET_COLS + OPERATIONAL_COLS))

# ---------------------------------------------------------------------------
# Variable Type Classification
# ---------------------------------------------------------------------------
# Each variable is assigned a type that determines how it is treated:
#   - "binary":       0/1 indicator variables (no z-score; kept as-is or mean-coded)
#   - "ordinal":      Discrete integers with a natural order (ranked, not z-scored)
#   - "bounded_pct":  Percentages bounded in [0, 100] (min-max or logit transform)
#   - "ratio":        Financial ratios that can be extreme (PE, EV/EBITDA) — robust winsorization
#   - "count":        Count / absolute magnitude variables (log-transformed if skewed)
#   - "continuous":   Standard continuous variables (z-score normalization)
#   - "rate":         Small-range rates like ROA, ROE (already proportional)
#
# This classification follows best practices from:
#   Stevens (1946) "On the Theory of Scales of Measurement"
#   Hair et al. (2019) "Multivariate Data Analysis" — variable type determines method choice

VARIABLE_TYPES = {
    # --- Binary indicators (0/1) ---
    "carbon_reduction_target": "binary",
    "human_rights_policy": "binary",
    "anti_corruption_policy": "binary",

    # --- Ordinal / discrete integer ---
    "board_size": "ordinal",

    # --- Bounded percentages [0, 100] ---
    "renewable_energy_pct": "bounded_pct",
    "energy_efficiency": "bounded_pct",
    "waste_recycling_pct": "bounded_pct",
    "gender_diversity_pct": "bounded_pct",
    "women_management_pct": "bounded_pct",
    "board_independence_pct": "bounded_pct",
    "board_diversity_pct": "bounded_pct",
    "employee_satisfaction": "bounded_pct",
    "community_investment_pct": "bounded_pct",
    "supply_chain_audit_pct": "bounded_pct",
    "exec_comp_esg_linked": "bounded_pct",
    "free_float_pct": "bounded_pct",
    "shareholder_rights_score": "bounded_pct",
    "ethics_compliance_score": "bounded_pct",
    "data_privacy_score": "bounded_pct",
    "tax_transparency_score": "bounded_pct",
    "esg_controversy_score": "bounded_pct",
    "esg_risk_rating": "ordinal",       # Typically discrete (1-5 or similar scale)
    "profit_margins": "bounded_pct",
    "gross_margins": "bounded_pct",
    "operating_margins": "bounded_pct",

    # --- Financial ratios (can be extreme, need robust treatment) ---
    "trailing_pe": "ratio",
    "forward_pe": "ratio",
    "price_to_book": "ratio",
    "price_to_sales": "ratio",
    "enterprise_to_revenue": "ratio",
    "enterprise_to_ebitda": "ratio",
    "debt_to_equity": "ratio",
    "debt_to_ebitda": "ratio",
    "ceo_pay_ratio": "ratio",
    "pay_gap_ratio": "ratio",
    "cash_flow_to_debt": "ratio",
    "current_ratio": "ratio",
    "quick_ratio": "ratio",
    "amihud_illiquidity": "ratio",

    # --- Count / absolute magnitude (often right-skewed) ---
    "market_cap": "count",
    "total_revenue": "count",
    "ebitda": "count",
    "net_income": "count",
    "gross_profit": "count",
    "total_debt": "count",
    "total_cash": "count",
    "free_cashflow": "count",
    "operating_cashflow": "count",
    "r_d_expenditure": "count",
    "employees": "count",
    "avg_daily_volume": "count",
    "avg_daily_volume_30d": "count",
    "avg_daily_volume_90d": "count",
    "scope1_emissions": "count",
    "scope2_emissions": "count",
    "scope3_emissions": "count",
    "environmental_fines": "count",
    "52_week_high": "count",
    "52_week_low": "count",
    "50d_avg": "count",
    "200d_avg": "count",
    "price": "count",

    # --- Rates (already scaled, typically [-1, 1] or small range) ---
    "roa": "rate",
    "roe": "rate",
    "net_margin": "rate",
    "operating_margin": "rate",
    "gross_margin": "rate",
    "fcf_margin": "rate",
    "dividend_yield": "rate",
    "revenue_growth": "rate",
    "earnings_growth": "rate",
    "earnings_quarterly_growth": "rate",
    "r_d_intensity": "rate",
    "market_share": "rate",
    "emissions_intensity": "rate",
    "water_usage_intensity": "rate",

    # --- Reclassified: payout_ratio can exceed 100% or go negative ---
    "payout_ratio": "ratio",
    # --- Reclassified: revenue_per_employee is a large-magnitude absolute value ---
    "revenue_per_employee": "count",

    # --- Continuous (standard z-score normalization) ---
    "employee_turnover": "continuous",
    "injury_rate": "continuous",
    "safety_training_hours": "continuous",
    "price_volatility": "continuous",
    "price_volatility_30d": "continuous",
    "price_momentum_1m": "continuous",
    "price_momentum_3m": "continuous",
    "price_momentum_6m": "continuous",
    "price_momentum_12m": "continuous",
    "beta": "continuous",
    "bid_ask_spread": "continuous",
    "max_drawdown_1y": "continuous",
    "sharpe_ratio_1y": "continuous",
    "sortino_ratio_1y": "continuous",
    "avg_daily_return": "continuous",
    "return_skewness": "continuous",
    "return_kurtosis": "continuous",
    "pct_from_52w_high": "continuous",
}


def classify_variable(col):
    """Return the type of a variable based on the metadata dict, with auto-detection fallback."""
    if col in VARIABLE_TYPES:
        return VARIABLE_TYPES[col]
    # Auto-detect heuristics
    if col.endswith("_pct") or col.endswith("_percent"):
        return "bounded_pct"
    if col.endswith("_policy") or col.endswith("_target"):
        return "binary"
    return "continuous"


def load_raw():
    """Load raw data from any of the expected locations."""
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "combined_raw.csv",
        PROJECT_ROOT / "data" / "raw" / "combined_real_data.csv",
        PROJECT_ROOT / "data" / "processed" / "real_data_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns from {path}")
            return df
    raise FileNotFoundError(
        f"Raw data not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nRun: python scripts/01_download_data.py"
    )


def report_missing(df, label=""):
    """Report missing data statistics."""
    numeric = [c for c in ALL_NUMERIC if c in df.columns]
    total_cells = len(df) * max(len(numeric), 1)
    missing_cells = df[numeric].isna().sum().sum() if numeric else 0
    pct = missing_cells / total_cells * 100 if total_cells else 0
    print(f"  Missing data {label}: {pct:.1f}% ({missing_cells}/{total_cells})")
    return pct


def generate_quality_report(df, label=""):
    """Generate a detailed data quality report with variable type annotations."""
    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns]
    rows = []
    for col in numeric_cols:
        vals = df[col]
        vtype = classify_variable(col)
        rows.append({
            "column": col,
            "variable_type": vtype,
            "count": vals.count(),
            "missing": vals.isna().sum(),
            "missing_pct": vals.isna().mean() * 100,
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "q25": vals.quantile(0.25) if vals.count() > 0 else None,
            "median": vals.median(),
            "q75": vals.quantile(0.75) if vals.count() > 0 else None,
            "max": vals.max(),
            "skewness": vals.skew(),
            "kurtosis": vals.kurtosis(),
            "n_zeros": (vals == 0).sum(),
            "n_negative": (vals < 0).sum(),
            "n_unique": vals.nunique(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variable type classification report
# ---------------------------------------------------------------------------
def save_variable_type_report(df, tables_dir):
    """Save a report mapping each variable to its classified type."""
    rows = []
    for col in sorted([c for c in ALL_NUMERIC if c in df.columns]):
        vtype = classify_variable(col)
        vals = df[col].dropna()
        rows.append({
            "variable": col,
            "type": vtype,
            "n_unique": vals.nunique(),
            "min": vals.min() if len(vals) > 0 else None,
            "max": vals.max() if len(vals) > 0 else None,
            "treatment": {
                "binary": "Keep as 0/1; proportion-encoded for group scores",
                "ordinal": "Rank-transform then percentile-scale to [0, 100]",
                "bounded_pct": "Bounded [0,100]; min-max scale within natural bounds",
                "ratio": "Robust winsorize (2.5/97.5 MAD-based); then z-score",
                "count": "Log1p transform if skewed; then z-score",
                "rate": "Winsorize at 1/99; then z-score",
                "continuous": "Winsorize at 1/99; then z-score",
            }.get(vtype, "z-score"),
        })
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "variable_type_classification.csv", index=False, encoding="utf-8")
    print(f"  [OK] Saved variable_type_classification.csv ({len(result)} variables)")
    return result


# ---------------------------------------------------------------------------
# Outlier Detection — Multi-Method
# ---------------------------------------------------------------------------
def detect_outliers_iqr(series, k=1.5):
    """Detect outliers using Tukey's IQR rule (k=1.5 standard, k=3 extreme)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_outliers_mad(series, threshold=3.5):
    """Detect outliers using Modified Z-score (MAD-based).

    Reference: Iglewicz & Hoaglin (1993) "Volume 16: How to Detect and Handle Outliers"
    The MAD (Median Absolute Deviation) is more robust to outliers than standard deviation.
    A threshold of 3.5 is recommended by Iglewicz & Hoaglin.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad < 1e-10:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median) / mad
    return np.abs(modified_z) > threshold


def detect_outliers_zscore(series, threshold=3.0):
    """Detect outliers using standard z-score."""
    mean, std = series.mean(), series.std()
    if std < 1e-10:
        return pd.Series(False, index=series.index)
    z = np.abs((series - mean) / std)
    return z > threshold


def detect_multivariate_outliers(df, cols, threshold_pct=97.5):
    """Detect multivariate outliers using Mahalanobis distance.

    Mahalanobis distance accounts for correlations between variables,
    catching observations that are unusual in multivariate space even if
    they appear normal univariately (Rousseeuw & Van Zomeren, 1990).
    """
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import chi2

    available = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                 and classify_variable(c) not in ("binary", "ordinal")]
    sub = df[available].dropna()
    if len(sub) < len(available) + 5 or len(available) < 3:
        return pd.Series(False, index=df.index, name="mahal_outlier")

    try:
        mean = sub.mean().values
        cov = sub.cov().values
        cov_inv = np.linalg.pinv(cov)  # Use pseudo-inverse for stability

        distances = []
        for _, row in sub.iterrows():
            d = mahalanobis(row.values, mean, cov_inv)
            distances.append(d)

        dist_series = pd.Series(distances, index=sub.index, name="mahal_dist")
        cutoff = chi2.ppf(threshold_pct / 100.0, df=len(available))
        outlier_mask = pd.Series(False, index=df.index, name="mahal_outlier")
        outlier_mask.loc[dist_series.index] = dist_series > np.sqrt(cutoff)

        n_outliers = outlier_mask.sum()
        print(f"  Mahalanobis outliers ({threshold_pct}th pctile, {len(available)} vars): "
              f"{n_outliers} / {len(sub)} ({n_outliers/len(sub)*100:.1f}%)")
        return outlier_mask
    except Exception as e:
        print(f"  Mahalanobis distance failed ({e}), skipping multivariate check")
        return pd.Series(False, index=df.index, name="mahal_outlier")


def comprehensive_outlier_report(df, cols):
    """Generate a comprehensive outlier report using multiple methods.

    Methods: IQR (k=1.5), IQR (k=3), MAD (3.5 threshold), Z-score (3 sigma).
    This multi-method approach increases confidence in outlier identification,
    following recommendations from Aguinis, Gottfredson & Joo (2013).
    """
    rows = []
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue

        vtype = classify_variable(col)
        # Skip binary and ordinal variables from outlier detection
        if vtype in ("binary", "ordinal"):
            continue

        iqr_mask = detect_outliers_iqr(vals, k=1.5)
        iqr_extreme_mask = detect_outliers_iqr(vals, k=3.0)
        mad_mask = detect_outliers_mad(vals, threshold=3.5)
        z_mask = detect_outliers_zscore(vals, threshold=3.0)

        # "Consensus outliers" — flagged by at least 2 of 3 methods
        consensus = (iqr_mask.astype(int) + mad_mask.astype(int) + z_mask.astype(int)) >= 2

        rows.append({
            "variable": col,
            "variable_type": vtype,
            "n_total": len(vals),
            "n_outliers_iqr": iqr_mask.sum(),
            "n_outliers_iqr_extreme": iqr_extreme_mask.sum(),
            "n_outliers_mad": mad_mask.sum(),
            "n_outliers_zscore": z_mask.sum(),
            "n_consensus_outliers": consensus.sum(),
            "pct_outliers_iqr": iqr_mask.mean() * 100,
            "pct_consensus": consensus.mean() * 100,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adaptive Winsorization by Variable Type
# ---------------------------------------------------------------------------
def adaptive_winsorize(df, cols):
    """Apply variable-type-aware winsorization.

    - binary: no winsorization
    - ordinal: no winsorization (already discrete)
    - bounded_pct: clip to [0, 100]
    - ratio: aggressive winsorize at 2.5/97.5 percentile (ratios are often extreme)
    - count: winsorize at 1/99 then log-transform if still skewed
    - rate: winsorize at 1/99
    - continuous: winsorize at 1/99
    """
    n_clipped = 0
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vtype = classify_variable(col)

        if vtype == "binary":
            # Ensure binary values are exactly 0 or 1
            df[col] = df[col].clip(0, 1).round().astype(float)
        elif vtype == "ordinal":
            # Round to nearest integer, but don't winsorize
            df[col] = df[col].round()
        elif vtype == "bounded_pct":
            # Clip to natural bounds
            df[col] = df[col].clip(0, 100)
        elif vtype == "ratio":
            # More aggressive winsorization for extreme ratios
            lo = df[col].quantile(0.025)
            hi = df[col].quantile(0.975)
            before = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            n_clipped += before
        else:
            # Standard winsorization for continuous, count, rate
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            before = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            n_clipped += before

    print(f"  Adaptive winsorization: {n_clipped} values clipped across {len(cols)} columns")
    return df


# ---------------------------------------------------------------------------
# Ordinal Encoding
# ---------------------------------------------------------------------------
def encode_ordinal_variables(df, cols):
    """Convert ordinal variables to percentile ranks within [0, 100].

    Percentile ranking is the recommended transformation for ordinal data
    when combining with continuous variables (Conover & Iman, 1981).
    """
    encoded = []
    for col in cols:
        if col not in df.columns:
            continue
        vtype = classify_variable(col)
        if vtype == "ordinal":
            # Percentile rank: maps to [0, 1], then scale to [0, 100]
            df[f"{col}_rank"] = df[col].rank(pct=True) * 100
            encoded.append(col)
    if encoded:
        print(f"  Ordinal variables percentile-ranked: {encoded}")
    return df


# ---------------------------------------------------------------------------
# Binary Variable Handling
# ---------------------------------------------------------------------------
def validate_binary_variables(df, cols):
    """Validate and standardize binary (0/1) variables.

    Binary variables should not be z-scored; instead they are kept as 0/1
    and their group-level proportions become the meaningful statistic.
    Adds a metadata column marking them as binary for downstream use.
    """
    binary_vars = []
    for col in cols:
        if col not in df.columns:
            continue
        vtype = classify_variable(col)
        if vtype == "binary":
            # Clean: any non-zero value -> 1, missing stays NaN
            mask = df[col].notna()
            df.loc[mask, col] = (df.loc[mask, col] > 0.5).astype(float)
            binary_vars.append(col)
    if binary_vars:
        print(f"  Binary variables validated: {binary_vars}")
    return df


def impute_sector_median(df, cols):
    """Impute missing values with sector median, then global median.

    For binary variables, uses sector-level mode (most common value) instead of median.
    For ordinal variables, uses sector median rounded to nearest integer.
    """
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not df[col].isna().any():
            continue

        vtype = classify_variable(col)

        if vtype == "binary":
            # Use mode (most frequent value) for binary imputation
            if "sector" in df.columns:
                df[col] = df.groupby("sector")[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else 0)
                )
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0)
        elif vtype == "ordinal":
            # Use median rounded to nearest integer
            if "sector" in df.columns:
                df[col] = df.groupby("sector")[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].round()
        else:
            # Standard sector median imputation for continuous/rate/ratio/count/bounded_pct
            if "sector" in df.columns:
                df[col] = df.groupby("sector")[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())
    return df


def log_transform_skewed(df, cols, skew_threshold=2.0):
    """Apply log1p transform to highly skewed count-type columns."""
    transformed = []
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vtype = classify_variable(col)
        # Only log-transform count variables and continuous with extreme skew
        if vtype not in ("count", "continuous", "rate"):
            continue
        skew = df[col].skew()
        if abs(skew) > skew_threshold and (df[col].dropna() >= 0).all():
            df[f"{col}_log"] = np.log1p(df[col])
            transformed.append(col)
    if transformed:
        print(f"  Log-transformed {len(transformed)} highly-skewed columns: {transformed[:8]}...")
    return df


def remove_low_coverage_columns(df, min_pct=0.30):
    """Remove columns with less than min_pct non-missing values."""
    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns]
    to_drop = []
    for col in numeric_cols:
        coverage = df[col].notna().mean()
        if coverage < min_pct:
            to_drop.append(col)
    if to_drop:
        print(f"  Removing {len(to_drop)} low-coverage columns (<{min_pct*100:.0f}%): {to_drop[:5]}...")
        df = df.drop(columns=to_drop)
    return df


def main():
    print("=" * 70)
    print("STEP 02: CLEAN AND STANDARDIZE DATA")
    print("=" * 70)

    df = load_raw()

    # Generate pre-cleaning quality report
    quality_before = generate_quality_report(df, "before")
    report_missing(df, "(before cleaning)")

    # Drop rows with no ticker
    df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    print(f"  Unique companies: {len(df)}")

    # Ensure numeric types for all known numeric columns
    for col in ALL_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove columns with very low coverage
    df = remove_low_coverage_columns(df, min_pct=0.20)

    # Identify numeric columns present in data
    numeric_in_data = [c for c in ALL_NUMERIC if c in df.columns
                       and pd.api.types.is_numeric_dtype(df[c])]

    # --- Step A: Variable type classification ---
    print("\n  [A] Classifying variable types...")
    type_counts = {}
    for col in numeric_in_data:
        vt = classify_variable(col)
        type_counts[vt] = type_counts.get(vt, 0) + 1
    for vt, cnt in sorted(type_counts.items()):
        print(f"      {vt:15s}: {cnt} variables")

    # --- Step B: Validate binary variables ---
    print("\n  [B] Validating binary variables...")
    df = validate_binary_variables(df, numeric_in_data)

    # --- Step C: Comprehensive outlier detection (before treatment) ---
    print("\n  [C] Running multi-method outlier detection...")
    outlier_report = comprehensive_outlier_report(df, numeric_in_data)
    if len(outlier_report) > 0:
        high_outlier = outlier_report[outlier_report["pct_consensus"] > 5]
        if len(high_outlier) > 0:
            print(f"      Variables with >5% consensus outliers: "
                  f"{high_outlier['variable'].tolist()[:10]}")

    # --- Step D-1: Multivariate outlier detection (informational) ---
    print("\n  [D-1] Multivariate outlier detection (Mahalanobis)...")
    mahal_mask = detect_multivariate_outliers(df, numeric_in_data)

    # --- Step D-2: Adaptive winsorization by variable type ---
    print("\n  [D-2] Applying adaptive winsorization...")
    df = adaptive_winsorize(df, numeric_in_data)

    # --- Step E: Log transform highly skewed variables ---
    print("\n  [E] Log-transforming skewed variables...")
    skew_candidates = [c for c in numeric_in_data if classify_variable(c) in ("count",)]
    df = log_transform_skewed(df, skew_candidates)

    # --- Step F: Ordinal encoding ---
    print("\n  [F] Encoding ordinal variables...")
    df = encode_ordinal_variables(df, numeric_in_data)

    # --- Step G: Impute missing with type-aware strategy ---
    print("\n  [G] Imputing missing values (type-aware sector-based)...")
    df = impute_sector_median(df, numeric_in_data)

    report_missing(df, "(after cleaning)")

    # Standardise country column
    if "country" not in df.columns:
        df["country"] = df["ticker"].apply(lambda t: "India" if ".NS" in str(t) else "US")
    else:
        country_map = {"IN": "India", "United States": "US", "USA": "US"}
        df["country"] = df["country"].replace(country_map)
        mask = ~df["country"].isin(["US", "India"])
        df.loc[mask, "country"] = df.loc[mask, "ticker"].apply(
            lambda t: "India" if ".NS" in str(t) else "US"
        )

    # Generate post-cleaning quality report
    quality_after = generate_quality_report(df, "after")

    # Print summary
    print(f"\n  Final dataset:")
    print(f"    Companies: {len(df)}")
    print(f"    Columns: {len(df.columns)}")
    if "sector" in df.columns:
        print(f"    Sectors: {df['sector'].nunique()}")
    if "country" in df.columns:
        for c, n in df["country"].value_counts().items():
            print(f"    {c}: {n}")

    # Save
    outpath = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"\n[OK] Clean data saved to {outpath}")

    # Save quality reports and outlier report
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    quality_before.to_csv(tables_dir / "data_quality_before.csv", index=False, encoding="utf-8")
    quality_after.to_csv(tables_dir / "data_quality_after.csv", index=False, encoding="utf-8")
    if len(outlier_report) > 0:
        outlier_report.to_csv(tables_dir / "outlier_report.csv", index=False, encoding="utf-8")
        print(f"[OK] Outlier report saved ({len(outlier_report)} variables analyzed)")
    save_variable_type_report(df, tables_dir)
    print(f"[OK] Quality reports saved to {tables_dir}")
    print("[DONE] Next: python scripts/03_build_index.py")


if __name__ == "__main__":
    main()
