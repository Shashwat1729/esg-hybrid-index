"""
Step 03: Build Multi-Factor Index
===================================
Constructs composite scores for each of ten factor categories:
  1. ESG Composite (E, S, G pillars)
  2. Financial Score (profitability, growth, efficiency, stability, valuation)
  3. Market Score (liquidity, volatility, momentum)
  4. Operational Score (efficiency, innovation, market position)
  5. Risk-Adjusted Score (Sharpe, Sortino, drawdown)
  6. Value Score (P/E, P/B, EV/EBITDA)
  7. Growth Score (revenue growth, earnings growth, momentum)
  8. Stability Score (leverage, liquidity ratios)
  9. Similarity Rank (cosine similarity on ESG vectors)
  10. Sector Position (percentile rank within sector)

Then computes preference scores for three investor profiles using all 10 factors.
Finally derives data-driven weight rationale using PCA explained variance.

Input:  data/processed/clean_data.csv
Output: data/processed/indexed_data.csv
        reports/tables/company_rankings.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, spearmanr
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

import logging

logger = logging.getLogger(__name__)

from src.data_collection.data_pipeline import load_configs
from src.index_construction.composite_index import CompositeIndexBuilder, _load_materiality_map
from src.financial_scoring.financial_scorer import FinancialScorer, MarketFactorScorer
from src.similarity.cosine_similarity import compute_similarity_matrix
from src.similarity.preference_scoring import PreferenceScorer
from src.constants import (
    ESG_ENV_COLS, ESG_SOC_COLS, ESG_GOV_COLS, ESG_COLS,
    BINARY_VARS, ORDINAL_VARS,
)


RATIO_TRANSFORM_COLS = {
    "trailing_pe", "forward_pe", "price_to_book", "price_to_sales",
    "enterprise_to_ebitda", "enterprise_to_revenue",
}
GROWTH_TRANSFORM_COLS = {"revenue_growth", "earnings_growth", "earnings_quarterly_growth"}
PROFITABILITY_TRANSFORM_COLS = {"return_on_assets", "return_on_equity", "roa", "roe"}

# ============================================================================
# NORMALIZATION PIPELINE DOCUMENTATION
# ============================================================================
# The multi-factor scoring system uses a 3-stage normalization pipeline:
#
# Stage 1: INDICATOR-LEVEL NORMALIZATION
#   - Method: Robust Z-score (median/MAD) for N < 100, standard z-score
#     for N >= 100, percentile-rank for N < 20  (see _z_score_sub)
#   - For ESG indicators, normalization is handled by
#     composite_index.normalize_indicators(); "lower is better" indicators
#     (from ESG_LOWER_IS_BETTER) are flipped AFTER normalization.
#   - For financial/market indicators, "lower is better" columns are NEGATED
#     by the caller BEFORE normalize_indicators() is invoked; winsorization
#     at 1st/99th percentiles therefore clips on the inverted scale.
#   - For other factors (_z_score_sub), inversion is handled inline via
#     `if not higher_better: z = -z`, and z-scores are clipped to [-3, 3].
#   - Output: z-scored (or percentile-ranked) values centered around 0.
#
# Stage 2: FACTOR SCORE COMPUTATION
#   - ESG: Category-weighted aggregation → pillar scores (E, S, G) →
#     SASB sector-materiality-weighted composite.
#     Raw z-score composites (scale_to_score=False in pipeline).
#   - Financial: Category-weighted aggregation of _norm columns.
#     Raw z-score composites (scale_to_score=False in pipeline).
#   - Market: Same structure as Financial.
#     Raw z-score composites (scale_to_score=False in pipeline).
#   - Other factors (operational, risk-adjusted, value, growth, stability):
#     Hierarchically-weighted z-score aggregation using YAML config weights.
#     Two-level weighting: category_weight × indicator_weight per z-score.
#     Raw z-score composites (scale_to_score=False in pipeline).
#
# Stage 3: SINGLE STANDARDIZATION (Step 7b below)
#   - ALL factor scores are standardized ONCE to: mean=50, std=10,
#     clipped to [0, 100].
#   - This is the ONLY transformation from z-scores to the 0-100 scale,
#     avoiding lossy double-transformation (clip at tails then re-standardize).
#   - After this step, one standard deviation in ANY factor equals 10 points.
#
# The preference scores (Step 9) use percentile-rank transforms of the
# re-standardized scores, ensuring approximately equal contribution per
# factor regardless of its original distributional shape.
# ============================================================================


def _get_variable_type(col):
    """Look up variable type from shared constants."""
    if col in BINARY_VARS:
        return "binary"
    if col in ORDINAL_VARS:
        return "ordinal"
    return "continuous"


def _z_score_sub(df, indicators_dict, scale_to_score=True, sector_adjust=False, sector_col="sector"):
    """Compute a composite score from z-scored indicators, handling variable types.

    Binary variables (0/1) are mean-coded directly rather than z-scored,
    because z-scoring binary data produces misleading values and inflated
    influence from rare categories (Agresti, 2002; Hair et al., 2019).

    Ordinal variables use rank-based scoring if a _rank column exists.

    Normalization strategy adapts to sample size N:
      - N < 20:   percentile-rank normalization (z-scores unreliable)
      - N < 100:  robust z-score using median/MAD (outlier-resistant)
      - N >= 100: standard z-score (mean/std)
    The robust method uses MAD * 1.4826 as the scale estimator, which is
    consistent with std for normally distributed data (Leys et al., 2013).

    Parameters
    ----------
    df : pd.DataFrame
    indicators_dict : dict
        {column_name: higher_is_better (bool)}
    scale_to_score : bool, default True
        When True, applies ``50 + z * 20`` transformation and clips to [0, 100].
        When False, returns the raw average z-score without transformation.
    sector_adjust : bool, default False
        When True and ``sector_col`` exists, rank continuous indicators within
        sector first, then normalize cross-sectionally.
    sector_col : str, default "sector"
        Sector column used when ``sector_adjust=True``.

    Returns
    -------
    pd.Series
        Score centered at 50, scaled by 20 (when scale_to_score=True),
        or raw average z-score (when scale_to_score=False).
    """
    available = {k: v for k, v in indicators_dict.items() if k in df.columns}
    if not available:
        return pd.Series(50.0, index=df.index)

    N = len(df)

    # Determine normalization method based on sample size
    if N < 20:
        norm_method = "percentile_rank"
        logger.warning(
            "Sample size N=%d is very small (<20); using percentile-rank "
            "normalization instead of z-scores for numerical stability.", N
        )
    elif N < 100:
        norm_method = "robust_zscore"
        logger.info(
            "Sample size N=%d (<100); using robust z-score (median/MAD) "
            "to reduce outlier sensitivity.", N
        )
    else:
        norm_method = "zscore"
        logger.info(
            "Sample size N=%d (>=100); using standard z-score (mean/std).", N
        )

    z_df = pd.DataFrame(index=df.index)
    for col, higher_better in available.items():
        vtype = _get_variable_type(col)
        vals = pd.to_numeric(df[col], errors="coerce")

        # If all values in column are NaN, keep as NaN so it doesn't
        # contribute 0.0 to the mean (Issue #14 fix)
        if vals.isna().all():
            z_df[col] = np.nan
            continue

        if vtype == "binary":
            # For binary variables: use the proportion directly
            # 1 = has feature -> positive contribution, 0 = does not
            # Map to z-score-like scale: mean-center and scale by std
            # but cap contribution to prevent binary vars from dominating
            proportion = vals.mean()
            if proportion > 0 and proportion < 1:
                z = (vals - proportion) / max(vals.std(), 0.1)
            else:
                z = pd.Series(0.0, index=df.index)
            if not higher_better:
                z = -z
            z = z.clip(-2, 2)  # Tighter clip for binary variables
            z_df[col] = z

        elif vtype == "ordinal":
            # Use rank-based column if available, else rank in place
            rank_col = f"{col}_rank"
            if rank_col in df.columns:
                rank_vals = pd.to_numeric(df[rank_col], errors="coerce")
            else:
                rank_vals = vals.rank(pct=True) * 100
            # Z-score the ranks
            mean, std = rank_vals.mean(), rank_vals.std()
            if std > 1e-10:
                z = (rank_vals - mean) / std
            else:
                z = pd.Series(0.0, index=df.index)
            if not higher_better:
                z = -z
            z = z.clip(-3, 3)
            z_df[col] = z

        else:
            # Continuous variable normalization — method depends on N
            col_key = col.lower()

            # Non-linear transforms before normalization
            if col_key in RATIO_TRANSFORM_COLS:
                vals = np.sign(vals) * np.log1p(np.abs(vals))
            elif col_key in GROWTH_TRANSFORM_COLS:
                vals = np.sign(vals) * np.sqrt(np.abs(vals))
            elif col_key in PROFITABILITY_TRANSFORM_COLS:
                median = vals.median()
                vals = 1.0 / (1.0 + np.exp(-2.0 * (vals - median)))

            # Optional sector-relative preprocessing
            if sector_adjust and sector_col in df.columns:
                vals = vals.groupby(df[sector_col]).rank(pct=True, na_option="keep")

            # Adaptive normality check: non-normal -> rank-based normalization
            use_rank_based = norm_method == "percentile_rank"
            non_na = vals.dropna()
            if not use_rank_based and len(non_na) >= 8 and non_na.nunique() > 3:
                sample = non_na
                if len(sample) > 5000:
                    sample = sample.sample(5000, random_state=42)
                try:
                    _, p_norm = shapiro(sample.values)
                    if np.isfinite(p_norm) and p_norm < 0.05:
                        use_rank_based = True
                except Exception:
                    # Fall back to configured method if Shapiro fails
                    pass

            if use_rank_based:
                # Percentile-rank: map to [0, 1] then to z-score-like scale
                # via inverse normal CDF approximation: (rank - 0.5) / N
                # scaled to have mean≈0, std≈1
                pct = vals.rank(pct=True, na_option="keep")
                # Proper probit transform: inverse normal CDF maps percentile
                # ranks to z-scores.  Clip to [0.01, 0.99] to avoid ±∞.
                z = pd.Series(norm.ppf(pct.clip(0.01, 0.99)), index=df.index)

            elif norm_method == "robust_zscore":
                # Robust z-score: use median and MAD instead of mean/std
                # MAD * 1.4826 is a consistent estimator of std for normal data
                median = vals.median()
                mad = (vals - median).abs().median()
                mad_scaled = mad * 1.4826  # consistency constant
                if mad_scaled > 1e-10:
                    z = (vals - median) / mad_scaled
                else:
                    # MAD is zero (>50% identical values): fall back to std
                    std = vals.std()
                    if std > 1e-10:
                        z = (vals - median) / std
                    else:
                        z = pd.Series(0.0, index=df.index)

            else:
                # Standard z-score (N >= 100)
                mean, std = vals.mean(), vals.std()
                if std > 1e-10:
                    z = (vals - mean) / std
                else:
                    z = pd.Series(0.0, index=df.index)

            if not higher_better:
                z = -z
            # Winsorize extreme z-scores to [-3, 3]
            z = z.clip(-3, 3)
            z_df[col] = z

    # Log warning if many companies have low indicator coverage (Issue #13)
    indicator_count = z_df.notna().sum(axis=1)
    low_coverage = (indicator_count < max(2, len(available) // 2)).sum()
    if low_coverage > 0:
        logger.debug(f"  {low_coverage} companies have <50% indicator coverage for this factor")

    # Average z-score across available indicators, then map to 0-100 scale
    # Note: mean(axis=1) with default skipna=True correctly excludes NaN columns,
    # so all-NaN columns (set to np.nan in Issue #14 fix) don't contribute 0.0.
    avg_z = z_df.mean(axis=1).fillna(0)
    if scale_to_score:
        score = 50 + avg_z * 20  # 1 std = 20 points, centered at 50
        score = score.clip(0, 100)
    else:
        score = avg_z
    # Safety net: fill any remaining NaN with neutral score
    score = score.fillna(0.0 if not scale_to_score else 50.0)
    return score


def _parse_config_scoring_section(config_section):
    """Parse a config-driven scoring section into an indicators dict for _z_score_sub.

    Reads the YAML structure:
        section:
          categories:
            cat_name:
              weight: <float>
              indicators:
                indicator_name:
                  weight: <float>
                  inverse: <bool>  # optional, default false

    Returns
    -------
    dict
        {indicator_name: higher_is_better (bool)} — compatible with _z_score_sub.
        The ``inverse: true`` flag maps to ``higher_is_better=False``.
    """
    indicators = {}
    categories = config_section.get("categories", {})
    for cat_name, cat_cfg in categories.items():
        cat_indicators = cat_cfg.get("indicators", {})
        for ind_name, ind_cfg in cat_indicators.items():
            # inverse: true in config means lower is better → higher_is_better = False
            is_inverse = ind_cfg.get("inverse", False) if isinstance(ind_cfg, dict) else False
            indicators[ind_name] = not is_inverse
    return indicators


def _build_weighted_config_score(
    df,
    config_section,
    fallbacks=None,
    scale_to_score=True,
    hierarchical_zscore=False,
    sector_adjust=False,
    sector_col="sector",
):
    """Compute a hierarchically-weighted composite score from a config section.

    Applies the two-level weight hierarchy specified in the YAML config:

        score = Σ_c [ w_c · Σ_i ( w_i · z_i ) / Σ_i w_i ] / Σ_c w_c

    where:
      - w_c = category weight (e.g., productivity: 0.40)
      - w_i = indicator weight within that category (e.g., revenue_per_employee: 0.50)
      - z_i = z-scored indicator value (computed via the same normalization logic
              as _z_score_sub: robust z-score for N<100, standard z-score for N>=100,
              percentile-rank for N<20)

    This ensures the relative importance of economic factors matches the
    academic literature on mid-cap evaluation, rather than diluting
    theoretically important factors via equal-weight averaging.

    Missing indicators are skipped with a warning, and remaining weights
    within that category are re-normalized.  If an entire category has no
    available indicators, its category weight is redistributed proportionally
    across the remaining categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with indicator columns.
    config_section : dict
        The config section (e.g., index_cfg["operational_quality"]) containing
        ``categories`` with ``weight`` and ``indicators`` sub-keys.
    fallbacks : dict or None
        Optional {missing_indicator: replacement_indicator} mappings.
    scale_to_score : bool, default True
        When True, applies ``50 + z * 20`` transformation and clips to [0, 100].
        When False, returns the raw weighted z-score without transformation.

    Returns
    -------
    pd.Series
        Score centered at 50, scaled by 20, clipped to [0, 100] (when
        scale_to_score=True), or raw weighted z-score (when False).
    """
    categories = config_section.get("categories", {})
    if not categories:
        return pd.Series(50.0, index=df.index)

    # ------------------------------------------------------------------
    # Phase 1: Z-score every indicator (same normalization as _z_score_sub)
    # ------------------------------------------------------------------
    def _zscore_single(vals, higher_better, col_name):
        """Z-score a single indicator series, respecting variable type."""
        temp = pd.DataFrame({col_name: vals}, index=df.index)
        if sector_adjust and sector_col in df.columns:
            temp[sector_col] = df[sector_col]
        return _z_score_sub(
            temp,
            {col_name: higher_better},
            scale_to_score=False,
            sector_adjust=sector_adjust,
            sector_col=sector_col,
        )

    # ------------------------------------------------------------------
    # Phase 2: Hierarchical weighted aggregation
    # ------------------------------------------------------------------
    # For each category, compute the weighted average of its indicators' z-scores.
    # Then combine categories using their category-level weights.
    category_scores = {}   # {cat_name: pd.Series}
    category_weights = {}  # {cat_name: float}

    for cat_name, cat_cfg in categories.items():
        cat_weight = cat_cfg.get("weight", 0.0)
        cat_indicators = cat_cfg.get("indicators", {})
        if not cat_indicators or cat_weight <= 0:
            continue

        # Collect z-scores and weights for available indicators in this category
        ind_z_scores = {}   # {ind_name: pd.Series}
        ind_weights = {}    # {ind_name: float}

        for ind_name, ind_cfg in cat_indicators.items():
            # Resolve indicator name (apply fallbacks if needed)
            resolved_name = ind_name
            if fallbacks and ind_name not in df.columns and ind_name in (fallbacks or {}):
                replacement = fallbacks[ind_name]
                if replacement in df.columns:
                    resolved_name = replacement
                    logger.info("Category '%s': %s not found; falling back to %s",
                                cat_name, ind_name, replacement)

            if resolved_name not in df.columns:
                logger.debug("Category '%s': indicator '%s' not in DataFrame, skipping",
                             cat_name, ind_name)
                continue

            ind_w = ind_cfg.get("weight", 0.0) if isinstance(ind_cfg, dict) else 0.0
            if ind_w <= 0:
                continue

            is_inverse = ind_cfg.get("inverse", False) if isinstance(ind_cfg, dict) else False
            higher_better = not is_inverse

            z = _zscore_single(df[resolved_name], higher_better, resolved_name)
            ind_z_scores[resolved_name] = z
            ind_weights[resolved_name] = ind_w

        if not ind_z_scores:
            logger.warning("Category '%s' in config has no available indicators; "
                           "skipping (weight %.2f will be redistributed)", cat_name, cat_weight)
            continue

        # Re-normalize indicator weights within this category so they sum to 1.0
        # (handles missing indicators gracefully)
        total_ind_w = sum(ind_weights.values())
        if total_ind_w <= 0:
            continue

        # Weighted average of z-scores within this category
        cat_z = pd.Series(0.0, index=df.index)
        for ind_name, z in ind_z_scores.items():
            normalized_w = ind_weights[ind_name] / total_ind_w
            # Where z is NaN, contribute 0 but track for coverage
            cat_z = cat_z + z.fillna(0) * normalized_w

        category_scores[cat_name] = cat_z
        category_weights[cat_name] = cat_weight

    if not category_scores:
        logger.warning("No categories with available indicators; defaulting to 50.0")
        return pd.Series(50.0, index=df.index)

    # Re-normalize category weights (handles missing entire categories)
    total_cat_w = sum(category_weights.values())
    if total_cat_w <= 0:
        return pd.Series(50.0, index=df.index)

    # Optional hierarchical re-standardization: each category contributes on
    # comparable variance scale before category-level weighted aggregation.
    if hierarchical_zscore:
        for cat_name in list(category_scores.keys()):
            cat_vals = category_scores[cat_name]
            cat_std = cat_vals.std()
            if pd.notna(cat_std) and cat_std > 1e-10:
                category_scores[cat_name] = (cat_vals - cat_vals.mean()) / cat_std

    # Weighted sum of category z-scores
    weighted_z = pd.Series(0.0, index=df.index)
    for cat_name, cat_z in category_scores.items():
        normalized_cat_w = category_weights[cat_name] / total_cat_w
        weighted_z = weighted_z + cat_z * normalized_cat_w

    # Map to score space (when scale_to_score is True) or return raw z-score
    if scale_to_score:
        score = (50 + weighted_z * 20).clip(0, 100)
        score = score.fillna(50.0)
    else:
        score = weighted_z.fillna(0.0)
    return score


def build_config_driven_score(
    df,
    index_cfg,
    config_key,
    score_col,
    fallbacks=None,
    scale_to_score=True,
    hierarchical_zscore=False,
    sector_adjust=False,
    sector_col="sector",
):
    """Build a factor score from a config-driven scoring section.

    This generic function replaces the individual build_*_score() functions.
    It reads indicator definitions, directionality (inverse flag), and
    **hierarchical weights** from the specified config section, then applies
    a two-level weighted aggregation:

        score = Σ_c [ w_c · Σ_i ( w_i · z_i ) ] / Σ_c w_c

    where w_c = category weight, w_i = indicator weight, z_i = z-scored value.

    This ensures that YAML-specified weights are respected rather than using
    equal-weight averaging across all indicators (which would dilute the
    contribution of theoretically important factors).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with indicator columns.
    index_cfg : dict
        Full index configuration (from config/index_config.yaml).
    config_key : str
        Top-level key in index_cfg for this scoring section
        (e.g. "operational_quality", "risk_adjusted_scoring").
    score_col : str
        Name of the output score column (e.g. "operational_score").
    fallbacks : dict or None
        Optional dict of {missing_indicator: replacement_indicator} for
        graceful degradation when preferred indicators are absent.
        Example: {"price_volatility_30d": "price_volatility"}
    scale_to_score : bool, default True
        When True, applies ``50 + z * 20`` transformation and clips to [0, 100].
        When False, returns raw z-score composite — useful when a downstream
        step (e.g. Step 7b re-standardization) will apply its own scaling.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with the new score column added.
    """
    config_section = index_cfg.get(config_key, {})
    if not config_section:
        logger.warning("Config section '%s' not found; defaulting %s to 50.0",
                        config_key, score_col)
        df[score_col] = 50.0
        return df

    # Use hierarchically-weighted aggregation instead of equal-weight _z_score_sub
    df[score_col] = _build_weighted_config_score(
        df,
        config_section,
        fallbacks=fallbacks,
        scale_to_score=scale_to_score,
        hierarchical_zscore=hierarchical_zscore,
        sector_adjust=sector_adjust,
        sector_col=sector_col,
    )
    df[score_col] = df[score_col].fillna(0.0 if not scale_to_score else 50.0)
    return df


def _select_return_proxy(df):
    """Create composite quality proxy for monotonicity correction.

    CIRCULARITY FIX (Issue C1): Instead of using price_momentum_6m (which
    overlaps with market_score), we construct a forward quality proxy from
    earnings-based and risk-adjusted metrics that are NOT used in market_score
    construction. This breaks the circular dependency where market_score was
    validated against its own inputs.

    The proxy blends:
      - earnings_growth: forward earnings trajectory
      - revenue_growth: top-line growth signal
      - roa/return_on_assets: asset efficiency (DuPont quality)
      - roe/return_on_equity: equity return quality
      - sharpe_ratio_1y: risk-adjusted performance

    Falls back to price_momentum_6m only if no quality metrics are available.
    """
    quality_candidates = [
        ("earnings_growth", True),
        ("revenue_growth", True),
        ("earnings_quarterly_growth", True),
        ("roa", True),
        ("roe", True),
        ("return_on_assets", True),
        ("return_on_equity", True),
        ("sharpe_ratio_1y", True),
        ("net_income_margin", True),
        ("operating_margins", True),
    ]

    available = []
    for col, higher_better in quality_candidates:
        if col in df.columns and df[col].notna().sum() > 20:
            available.append((col, higher_better))

    if len(available) >= 2:
        # Build composite quality proxy from available metrics
        proxy_parts = []
        for col, higher_better in available:
            vals = pd.to_numeric(df[col], errors="coerce")
            pct_rank = vals.rank(pct=True, na_option="keep") * 100.0
            if not higher_better:
                pct_rank = 100.0 - pct_rank
            proxy_parts.append(pct_rank)

        df["forward_quality_proxy"] = pd.concat(proxy_parts, axis=1).mean(axis=1)
        print(f"    [C1 FIX] Built forward_quality_proxy from {len(available)} metrics: "
              f"{[c for c, _ in available]}")
        return "forward_quality_proxy"

    # Fallback to momentum only if quality metrics unavailable
    momentum_candidates = ["price_momentum_3m", "price_momentum_6m",
                           "price_momentum_1m", "price_momentum_12m"]
    for col in momentum_candidates:
        if col in df.columns and df[col].notna().any():
            print(f"    [WARNING] Falling back to momentum proxy {col} "
                  f"(quality metrics unavailable)")
            return col
    return None


def _quintile_means(df, score_col, return_col):
    """Return quintile mean returns (Q1..Q5) for one factor."""
    valid = df[[score_col, return_col]].dropna().copy()
    if len(valid) < 25:
        return np.array([np.nan] * 5), pd.Series(index=df.index, dtype=float)

    try:
        q = pd.qcut(valid[score_col], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except Exception:
        q = pd.qcut(valid[score_col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")

    valid["_q"] = q.astype(float)
    means = valid.groupby("_q", observed=False)[return_col].mean()
    out = np.array([means.get(float(i), np.nan) for i in [1, 2, 3, 4, 5]])
    return out, valid["_q"]


def _jt_approx_pvalue(quintile_means):
    """Approximate Jonckheere-Terpstra one-sided p-value from quintile means."""
    means = np.asarray(quintile_means, dtype=float)
    if len(means) != 5 or np.isnan(means).any():
        return np.nan
    j_stat = 0.0
    k = len(means)
    for i in range(k):
        for j in range(i + 1, k):
            if means[j] > means[i]:
                j_stat += 1.0
            elif means[j] < means[i]:
                j_stat -= 1.0
    # Normal approximation for ordered-trend sign statistic
    var_j = k * (k - 1) * (2 * k + 5) / 72.0
    if var_j <= 0:
        return np.nan
    z = j_stat / np.sqrt(var_j)
    return float(1.0 - norm.cdf(z))


def _is_monotonic(means):
    """Check monotone non-decreasing quintile means."""
    arr = np.asarray(means, dtype=float)
    if len(arr) != 5 or np.isnan(arr).any():
        return False
    return bool(np.all(np.diff(arr) >= -1e-10))


def _build_indicator_map(index_cfg, df):
    """Build directed indicator map for monotonicity correction."""
    indicator_map = {}

    indicator_map["ESG_composite"] = {c: True for c in ESG_COLS if c in df.columns}

    fin_categories = index_cfg.get("financial_scoring", {}).get("categories", {})
    fin_inverse = set(index_cfg.get("financial_scoring", {}).get("inverse_indicators", []))
    fin_map = {}
    for cat_cfg in fin_categories.values():
        for ind in cat_cfg.get("indicators", []):
            if ind in df.columns:
                fin_map[ind] = ind not in fin_inverse
    indicator_map["financial_score"] = fin_map

    mkt_categories = index_cfg.get("market_factors", {}).get("categories", {})
    mkt_inverse = set(index_cfg.get("market_factors", {}).get("inverse_indicators", []))
    mkt_map = {}
    for cat_cfg in mkt_categories.values():
        for ind in cat_cfg.get("indicators", []):
            if ind in df.columns:
                mkt_map[ind] = ind not in mkt_inverse
    indicator_map["market_score"] = mkt_map

    cfg_factor_map = {
        "operational_score": "operational_quality",
        "risk_adjusted_score": "risk_adjusted_scoring",
        "value_score": "value_scoring",
        "growth_score": "growth_scoring",
        "stability_score": "stability_scoring",
    }
    for factor_col, cfg_key in cfg_factor_map.items():
        raw = _get_config_indicator_directed(index_cfg, cfg_key)
        indicator_map[factor_col] = {k: v for k, v in raw.items() if k in df.columns}

    return indicator_map


def _apply_sector_blend(df, factor_cols, sector_col="sector", w_cross=0.7, w_sector=0.3):
    """Create sector-adjusted columns and blend into final factor scores."""
    for col in factor_cols:
        if col not in df.columns:
            continue
        if sector_col in df.columns:
            sector_adj = df.groupby(sector_col)[col].rank(pct=True, na_option="keep") * 100.0
        else:
            sector_adj = df[col].rank(pct=True, na_option="keep") * 100.0
        sec_col = f"{col}_sector_adj"
        df[sec_col] = sector_adj.fillna(50.0)
        df[col] = (w_cross * df[col] + w_sector * df[sec_col]).clip(0, 100)
    return df


def _apply_monotonicity_correction(
    df,
    factor_cols,
    indicator_map,
    return_col,
    restd_mean,
    restd_std,
    score_clip_min,
    score_clip_max,
):
    """Reweight sub-indicators and isotonic-adjust non-monotonic factors.

    Uses continuous isotonic regression on decile bins for smoother correction,
    with within-bin rank preservation for differentiation.
    """
    # NOTE: The return_col should be a forward_quality_proxy (C1 fix), not
    # price_momentum_6m, to avoid circular validation with market_score.
    if return_col is None:
        return df

    for factor_col in factor_cols:
        if factor_col not in df.columns:
            continue

        base_means, _ = _quintile_means(df, factor_col, return_col)
        if _is_monotonic(base_means):
            continue

        # 1) Reweight sub-indicators by return association (monotonicity-oriented)
        ind_map = indicator_map.get(factor_col, {})
        weighted_components = []
        component_weights = []
        for ind_name, higher_better in ind_map.items():
            if ind_name not in df.columns:
                continue
            z = _z_score_sub(
                df,
                {ind_name: higher_better},
                scale_to_score=False,
                sector_adjust=True,
            )
            mask = z.notna() & df[return_col].notna()
            if mask.sum() < 10:
                continue
            rho, _ = spearmanr(z[mask], df.loc[mask, return_col])
            if not np.isfinite(rho):
                continue
            weighted_components.append(z)
            component_weights.append(abs(float(rho)))

        if weighted_components and np.sum(component_weights) > 1e-12:
            comp = pd.Series(0.0, index=df.index)
            total_w = float(np.sum(component_weights))
            for s, w in zip(weighted_components, component_weights):
                comp = comp + s.fillna(0.0) * (w / total_w)
            cand = comp
            cand_means, _ = _quintile_means(pd.DataFrame({factor_col: cand, return_col: df[return_col]}), factor_col, return_col)
            if np.nansum(np.diff(cand_means) >= 0) >= np.nansum(np.diff(base_means) >= 0):
                df[factor_col] = cand
                base_means = cand_means

        # 2) Continuous isotonic regression using decile bins for smoother correction
        #    with within-bin rank preservation for differentiation
        valid_mask = df[factor_col].notna() & df[return_col].notna()
        valid = df.loc[valid_mask, [factor_col, return_col]].copy()
        if len(valid) < 20:
            continue

        # Use 10 decile bins for finer-grained monotonic fitting
        try:
            valid["_bin"] = pd.qcut(valid[factor_col], 10, labels=False, duplicates="drop")
        except Exception:
            valid["_bin"] = pd.qcut(valid[factor_col].rank(method="first"), 10, labels=False, duplicates="drop")

        bin_means_factor = valid.groupby("_bin", observed=False)[factor_col].mean()
        bin_means_return = valid.groupby("_bin", observed=False)[return_col].mean()

        # Fit isotonic regression: factor score → expected return (monotonic)
        x_bins = bin_means_factor.values
        y_bins = bin_means_return.values
        finite_mask = np.isfinite(x_bins) & np.isfinite(y_bins)
        if finite_mask.sum() < 3:
            continue

        x_finite = x_bins[finite_mask]
        y_finite = y_bins[finite_mask]
        sort_idx = np.argsort(x_finite)
        x_sorted = x_finite[sort_idx]
        y_sorted = y_finite[sort_idx]

        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(x_sorted, y_sorted)

        # Map each company's factor score through isotonic transform
        raw_scores = df.loc[valid_mask, factor_col].values.copy()
        transformed = iso.predict(raw_scores)

        # Re-rank to preserve order and map back to original score scale
        # The isotonic transform ensures monotonicity while the rank-mapping
        # preserves differentiation
        rank_orig = pd.Series(raw_scores).rank(method="average")
        rank_trans = pd.Series(transformed).rank(method="average")
        # Blend: stronger isotonic correction for factors with poor monotonicity
        # Check how non-monotonic the factor is to determine blend strength
        post_means, _ = _quintile_means(
            pd.DataFrame({factor_col: raw_scores, return_col: df.loc[valid_mask, return_col].values}),
            factor_col, return_col,
        )
        n_increasing = int(np.nansum(np.diff(post_means) >= 0))
        # More aggressive correction for less monotonic factors
        iso_weight = 0.6 + 0.1 * max(0, 3 - n_increasing)  # 0.6 to 0.9
        iso_weight = min(iso_weight, 0.9)
        blended_rank = iso_weight * rank_trans + (1 - iso_weight) * rank_orig

        # Map blended ranks to score scale
        n = len(blended_rank)
        corrected_scores = (blended_rank / n) * (score_clip_max - score_clip_min) + score_clip_min

        corrected = df[factor_col].copy()
        corrected.loc[valid_mask] = corrected_scores.values

        # Re-standardize corrected factor to configured 0-100 scale
        if corrected.std() > 1e-10:
            corrected = ((corrected - corrected.mean()) / corrected.std()) * restd_std + restd_mean
            corrected = corrected.clip(score_clip_min, score_clip_max)
        df[factor_col] = corrected.fillna(restd_mean)

    return df


def _save_factor_monotonicity_diagnostic(df, factor_cols, return_col, out_path):
    """Save factor monotonicity diagnostic table."""
    rows = []
    for factor_col in factor_cols:
        if factor_col not in df.columns:
            continue
        if return_col is None:
            means = np.array([np.nan] * 5)
            p_val = np.nan
            is_mono = False
        else:
            means, _ = _quintile_means(df, factor_col, return_col)
            p_val = _jt_approx_pvalue(means)
            is_mono = _is_monotonic(means)
        rows.append({
            "factor": factor_col,
            "q1_mean": means[0],
            "q2_mean": means[1],
            "q3_mean": means[2],
            "q4_mean": means[3],
            "q5_mean": means[4],
            "jonckheere_terpstra_pvalue": p_val,
            "is_monotonic": is_mono,
            "return_proxy": return_col,
        })

    diag_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(out_path, index=False, encoding="utf-8")
    return diag_df


def _get_config_indicator_list(index_cfg, config_key):
    """Extract flat list of indicator names from a config-driven scoring section.

    Used by overlap report and sensitivity analysis to avoid hardcoded lists.
    """
    config_section = index_cfg.get(config_key, {})
    indicators = []
    for cat_cfg in config_section.get("categories", {}).values():
        for ind_name in cat_cfg.get("indicators", {}).keys():
            indicators.append(ind_name)
    return indicators


def _get_config_indicator_directed(index_cfg, config_key):
    """Extract {indicator: higher_is_better} dict from a config-driven scoring section.

    Used by overlap sensitivity analysis to avoid hardcoded dicts.
    """
    config_section = index_cfg.get(config_key, {})
    return _parse_config_scoring_section(config_section)


def build_sector_position(df):
    """Compute sector percentile rank from raw financial indicators.

    Uses raw input indicators (not computed factor scores) to avoid
    circularity — the preference score includes both factor scores and
    sector_position, so sector_position must be independent of them.
    """
    raw_indicators = ["revenue_per_employee", "operating_margins", "market_cap", "total_revenue"]
    avail = [c for c in raw_indicators if c in df.columns]
    if avail and "sector" in df.columns:
        # Z-score within sector for each indicator, then average
        z_scores = pd.DataFrame(index=df.index)
        for col in avail:
            vals = pd.to_numeric(df[col], errors="coerce")
            z_scores[col] = vals.groupby(df["sector"]).transform(
                lambda x: (x - x.median()) / (x.std() + 1e-10)
            )
        composite = z_scores.mean(axis=1)
        df["sector_position"] = composite.groupby(df["sector"]).transform(
            lambda x: x.rank(pct=True)
        )
    else:
        df["sector_position"] = 0.5
    return df


def build_similarity_rank(df, *, k=15):
    """Compute peer-adjusted ESG excess rank from ESG profile similarity.

    CIRCULARITY FIX (Issue M3) + LOW VARIANCE FIX (Issue M3b)
    ----------------------------------------------------------
    Previously this function used ALL 8 factor scores to compute cosine
    similarity, creating a circular dependency (Issue M3).  The initial fix
    restricted similarity to 3 ESG pillar scores (E_score, S_score, G_score),
    which broke the circularity but introduced a low-variance problem: cosine
    similarity in only 3 dimensions produces tightly clustered values (typical
    range 0.85-0.99), leaving near-zero information after min-max scaling.

    The current implementation uses ALL 32 ESG indicator columns (10 Env +
    10 Soc + 12 Gov) as the feature space.  These are raw/normalised input
    indicators -- they do NOT depend on similarity_rank or any composite that
    includes it, so circularity remains broken.  The 32-dimensional cosine
    similarity provides much greater discriminating power (Manning et al.,
    2008, "Introduction to Information Retrieval").

    Feature column selection priority:
      1. Normalised indicator columns ({col}_norm) -- created by
         CompositeIndexBuilder before this function is called.
      2. Raw ESG indicator columns -- fallback if _norm columns are absent.

    Missing values are imputed with per-column median before similarity
    computation to ensure a complete feature matrix.
    """
    # --- Resolve feature columns: prefer _norm variants, fall back to raw ---
    norm_cols = [f"{c}_norm" for c in ESG_COLS if f"{c}_norm" in df.columns]
    raw_cols = [c for c in ESG_COLS if c in df.columns]

    if len(norm_cols) >= 2:
        feature_cols = norm_cols
        source_label = f"{len(norm_cols)} normalised ESG indicators"
    elif len(raw_cols) >= 2:
        feature_cols = raw_cols
        source_label = f"{len(raw_cols)} raw ESG indicators"
    else:
        logger.warning(
            "Fewer than 2 ESG indicator columns available for similarity; "
            "defaulting similarity_rank to 0.5"
        )
        df["similarity_rank"] = 0.5
        return df, None

    logger.info("build_similarity_rank: using %s (%s)", source_label,
                ", ".join(feature_cols[:5]) + (" ..." if len(feature_cols) > 5 else ""))

    # --- Median-impute missing values so cosine similarity gets a full matrix ---
    df_sim = df.copy()
    for col in feature_cols:
        if df_sim[col].isna().any():
            median_val = df_sim[col].median()
            df_sim[col] = df_sim[col].fillna(median_val if pd.notna(median_val) else 0.0)

    sim_matrix = compute_similarity_matrix(
        df_sim, feature_cols=feature_cols, id_col="ticker", metric="cosine"
    )

    # --- Build ESG baseline used for peer-relative excess ---
    if "ESG_composite" in df.columns:
        esg_baseline = pd.to_numeric(df["ESG_composite"], errors="coerce")
        baseline_label = "ESG_composite"
    else:
        proxy_cols = [c for c in norm_cols if c in df_sim.columns]
        if proxy_cols:
            esg_baseline = df_sim[proxy_cols].mean(axis=1)
            baseline_label = f"mean of {len(proxy_cols)} ESG _norm columns"
        else:
            esg_baseline = df_sim[feature_cols].mean(axis=1)
            baseline_label = f"mean of {len(feature_cols)} ESG feature columns"

    if esg_baseline.isna().any():
        base_median = esg_baseline.median()
        esg_baseline = esg_baseline.fillna(base_median if pd.notna(base_median) else 0.0)

    sim_values = sim_matrix.values.copy()
    n = len(df)
    if n <= 1:
        logger.warning("Only one company available; defaulting similarity_rank to 0.5")
        df["similarity_rank"] = 0.5
        return df, sim_matrix

    k_eff = max(1, min(int(k), n - 1))
    logger.info(
        "build_similarity_rank: peer-relative excess using %s with k=%d",
        baseline_label,
        k_eff,
    )

    peer_excess = np.zeros(n, dtype=float)
    esg_values = esg_baseline.to_numpy(dtype=float)
    for i in range(n):
        row = sim_values[i].copy()
        row[i] = -np.inf  # exclude self
        nn_idx = np.argpartition(row, -k_eff)[-k_eff:]
        peer_excess[i] = esg_values[i] - np.mean(esg_values[nn_idx])

    df["similarity_rank"] = peer_excess
    lo, hi = df["similarity_rank"].min(), df["similarity_rank"].max()
    if hi - lo > 1e-10:
        df["similarity_rank"] = (df["similarity_rank"] - lo) / (hi - lo)
    return df, sim_matrix


def derive_pca_weight_rationale(df):
    """Use PCA to derive data-driven weight rationale for the 8 main factor scores.

    The PCA eigenvalue-based proportional contribution shows how much variance
    each factor explains, providing an empirical justification for weight ranges.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    available = [c for c in score_cols if c in df.columns]
    if len(available) < 4:
        return {}

    X = df[available].dropna()
    if len(X) < 10:
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    # Compute each factor's total contribution across all components
    # (absolute loading * explained variance ratio, summed across components)
    loadings = np.abs(pca.components_)  # (n_components, n_features)
    var_ratios = pca.explained_variance_ratio_

    # Weighted contribution of each feature
    contributions = (loadings.T * var_ratios).sum(axis=1)
    contributions = contributions / contributions.sum()

    rationale = {}
    for i, col in enumerate(available):
        rationale[col] = {
            "pca_contribution": float(contributions[i]),
            "suggested_weight_range": (
                max(0.02, float(contributions[i]) - 0.05),
                min(0.40, float(contributions[i]) + 0.10)
            ),
        }

    return rationale


def generate_indicator_overlap_report(index_cfg):
    """Build indicator-to-factor mapping, identify overlaps, and save reports.

    Programmatically maps every indicator to the factor score(s) it feeds into,
    identifies double/triple-counted indicators, and saves:
      - reports/tables/indicator_factor_mapping.csv  (indicator → factor list)
      - reports/tables/factor_overlap_matrix.csv     (factor × factor overlap count)

    This addresses Issue H2 (indicator overlap / double-counting) from the
    audit framework.

    Parameters
    ----------
    index_cfg : dict
        Loaded index configuration (from config/index_config.yaml).
    """
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Build the complete indicator → factor(s) mapping
    # -----------------------------------------------------------------------
    # Config-driven factor definitions: read from index_config.yaml sections
    # for the 5 formerly-hardcoded factors, and from FinancialScorer /
    # MarketFactorScorer config sections for financial and market scores.
    factor_indicators = {}

    # --- Factors from config-driven scoring sections (this script) ---
    _config_factor_map = {
        "operational_score": "operational_quality",
        "risk_adjusted_score": "risk_adjusted_scoring",
        "value_score": "value_scoring",
        "growth_score": "growth_scoring",
        "stability_score": "stability_scoring",
    }
    for factor_name, config_key in _config_factor_map.items():
        factor_indicators[factor_name] = _get_config_indicator_list(index_cfg, config_key)

    # --- financial_score: from config financial_scoring.categories ---
    fin_categories = index_cfg.get("financial_scoring", {}).get("categories", {})
    fin_indicators = []
    for cat_name, cat_cfg in fin_categories.items():
        fin_indicators.extend(cat_cfg.get("indicators", []))
    factor_indicators["financial_score"] = fin_indicators

    # --- market_score: from config market_factors.categories ---
    mkt_categories = index_cfg.get("market_factors", {}).get("categories", {})
    mkt_indicators = []
    for cat_name, cat_cfg in mkt_categories.items():
        mkt_indicators.extend(cat_cfg.get("indicators", []))
    factor_indicators["market_score"] = mkt_indicators

    # NOTE: ESG_composite uses ESG_COLS (E/S/G pillars) — no overlap with
    # financial/market indicators, so excluded from overlap analysis.
    # similarity_rank and sector_position are derived from factor scores,
    # not raw indicators, so also excluded.

    # -----------------------------------------------------------------------
    # 2. Invert to indicator → list of factors
    # -----------------------------------------------------------------------
    indicator_to_factors = {}
    for factor, indicators in factor_indicators.items():
        for ind in indicators:
            indicator_to_factors.setdefault(ind, []).append(factor)

    # Build mapping DataFrame
    rows = []
    for indicator, factors in sorted(indicator_to_factors.items()):
        rows.append({
            "indicator": indicator,
            "factors": "; ".join(sorted(factors)),
            "n_factors": len(factors),
            "is_overlapping": len(factors) > 1,
        })
    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(tables_dir / "indicator_factor_mapping.csv",
                      index=False, encoding="utf-8")

    # -----------------------------------------------------------------------
    # 3. Compute factor × factor overlap matrix
    # -----------------------------------------------------------------------
    all_factors = sorted(factor_indicators.keys())
    overlap_matrix = pd.DataFrame(0, index=all_factors, columns=all_factors)
    for i, f1 in enumerate(all_factors):
        s1 = set(factor_indicators[f1])
        for f2 in all_factors[i:]:
            s2 = set(factor_indicators[f2])
            shared = len(s1 & s2)
            overlap_matrix.loc[f1, f2] = shared
            overlap_matrix.loc[f2, f1] = shared
    overlap_matrix.to_csv(tables_dir / "factor_overlap_matrix.csv", encoding="utf-8")

    # -----------------------------------------------------------------------
    # 4. Log warnings for overlapping indicators
    # -----------------------------------------------------------------------
    n_total = len(indicator_to_factors)
    overlapping = mapping_df[mapping_df["is_overlapping"]]
    n_overlap = len(overlapping)

    print(f"\n--- Indicator Overlap Report (H2) ---")
    print(f"   Total unique indicators across all factors: {n_total}")
    print(f"   Indicators appearing in >1 factor: {n_overlap} "
          f"({100 * n_overlap / max(n_total, 1):.0f}%)")

    if n_overlap > 0:
        for _, row in overlapping.iterrows():
            msg = (f"   WARNING: '{row['indicator']}' appears in "
                   f"{row['n_factors']} factors: {row['factors']}")
            print(msg)
            logger.warning("Indicator overlap: %s -> %s",
                           row["indicator"], row["factors"])

    # Print overlap matrix summary
    print(f"\n   Factor overlap matrix (shared indicator counts):")
    for f1 in all_factors:
        for f2 in all_factors:
            if f1 < f2 and overlap_matrix.loc[f1, f2] > 0:
                print(f"     {f1} <-> {f2}: "
                      f"{overlap_matrix.loc[f1, f2]} shared indicators")

    print(f"\n   [OK] Saved indicator_factor_mapping.csv ({n_total} indicators)")
    print(f"   [OK] Saved factor_overlap_matrix.csv ({len(all_factors)}x{len(all_factors)})")

    return mapping_df, overlap_matrix


def overlap_sensitivity_analysis(df, index_cfg):
    """Compute factor scores with and without shared indicators.

    For each factor, compute an 'exclusive' version using only indicators
    unique to that factor.  Compare rankings via Spearman correlation.

    This demonstrates whether indicator overlap materially affects rankings
    and addresses Issue H1 (overlap sensitivity) from the audit framework.

    Parameters
    ----------
    df : pd.DataFrame
        The indexed DataFrame with all factor scores already computed.
    index_cfg : dict
        Loaded index configuration (from config/index_config.yaml).

    Returns
    -------
    pd.DataFrame
        Sensitivity results with columns: factor, n_total_indicators,
        n_exclusive_indicators, n_shared_indicators, pct_shared,
        spearman_rho, spearman_p, interpretation.
    """
    from scipy.stats import spearmanr

    tables_dir = PROJECT_ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Build factor → {indicator: higher_is_better} mapping
    #    Config-driven for formerly-hardcoded factors; config-based for
    #    financial and market scores.
    # -------------------------------------------------------------------
    factor_indicators_directed: dict[str, dict[str, bool]] = {}

    # --- Config-driven scoring sections (formerly hardcoded) ---
    _config_factor_map = {
        "operational_score": "operational_quality",
        "risk_adjusted_score": "risk_adjusted_scoring",
        "value_score": "value_scoring",
        "growth_score": "growth_scoring",
        "stability_score": "stability_scoring",
    }
    for factor_name, config_key in _config_factor_map.items():
        factor_indicators_directed[factor_name] = _get_config_indicator_directed(
            index_cfg, config_key
        )

    # financial_score — from config, with inverse indicator lookup
    fin_categories = index_cfg.get("financial_scoring", {}).get("categories", {})
    fin_inverse = set(index_cfg.get("financial_scoring", {}).get("inverse_indicators", []))
    fin_directed = {}
    for cat_cfg in fin_categories.values():
        for ind in cat_cfg.get("indicators", []):
            fin_directed[ind] = (ind not in fin_inverse)
    factor_indicators_directed["financial_score"] = fin_directed

    # market_score — from config, with inverse indicator lookup
    mkt_categories = index_cfg.get("market_factors", {}).get("categories", {})
    mkt_inverse = set(index_cfg.get("market_factors", {}).get("inverse_indicators", []))
    mkt_directed = {}
    for cat_cfg in mkt_categories.values():
        for ind in cat_cfg.get("indicators", []):
            mkt_directed[ind] = (ind not in mkt_inverse)
    factor_indicators_directed["market_score"] = mkt_directed

    # -------------------------------------------------------------------
    # 2. Build undirected indicator → set of factors (for overlap detection)
    # -------------------------------------------------------------------
    indicator_to_factors: dict[str, set[str]] = {}
    for factor, ind_dict in factor_indicators_directed.items():
        for ind in ind_dict:
            indicator_to_factors.setdefault(ind, set()).add(factor)

    # -------------------------------------------------------------------
    # 3. For each factor, split indicators into exclusive vs shared
    # -------------------------------------------------------------------
    results = []
    for factor, ind_dict in sorted(factor_indicators_directed.items()):
        # Only consider indicators actually present in the DataFrame
        available = {k: v for k, v in ind_dict.items() if k in df.columns}
        n_total = len(available)

        exclusive = {k: v for k, v in available.items()
                     if len(indicator_to_factors.get(k, set())) == 1}
        shared = {k: v for k, v in available.items()
                  if len(indicator_to_factors.get(k, set())) > 1}
        n_exclusive = len(exclusive)
        n_shared = len(shared)
        pct_shared = (100.0 * n_shared / n_total) if n_total > 0 else 0.0

        # Check if the full factor score column exists in df
        if factor not in df.columns:
            results.append({
                "factor": factor,
                "n_total_indicators": n_total,
                "n_exclusive_indicators": n_exclusive,
                "n_shared_indicators": n_shared,
                "pct_shared": round(pct_shared, 1),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "interpretation": "Factor score column not found in data",
            })
            continue

        # Edge case: no exclusive indicators → cannot compute exclusive version
        if n_exclusive == 0:
            results.append({
                "factor": factor,
                "n_total_indicators": n_total,
                "n_exclusive_indicators": 0,
                "n_shared_indicators": n_shared,
                "pct_shared": round(pct_shared, 1),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "interpretation": (
                    f"ALL {n_total} indicators are shared with other factors; "
                    "exclusive version cannot be computed — this factor is "
                    "entirely redundant with other factors at the indicator level"
                ),
            })
            continue

        # Compute exclusive-only factor score using _z_score_sub
        exclusive_score = _z_score_sub(df, exclusive)

        # Spearman rank correlation between full and exclusive versions
        # Drop rows where either is NaN
        mask = df[factor].notna() & exclusive_score.notna()
        if mask.sum() < 5:
            results.append({
                "factor": factor,
                "n_total_indicators": n_total,
                "n_exclusive_indicators": n_exclusive,
                "n_shared_indicators": n_shared,
                "pct_shared": round(pct_shared, 1),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "interpretation": "Too few valid observations for correlation",
            })
            continue

        rho, p_val = spearmanr(df.loc[mask, factor], exclusive_score[mask])

        # Interpretation
        if rho >= 0.95:
            interp = (f"rho={rho:.3f}: Overlap has NEGLIGIBLE impact; "
                      "shared indicators do not materially change rankings")
        elif rho >= 0.80:
            interp = (f"rho={rho:.3f}: Overlap has MODERATE impact; "
                      "shared indicators shift some rankings but overall "
                      "structure is preserved")
        elif rho >= 0.60:
            interp = (f"rho={rho:.3f}: Overlap has SUBSTANTIAL impact; "
                      "removing shared indicators meaningfully changes "
                      "factor rankings")
        else:
            interp = (f"rho={rho:.3f}: Overlap has SEVERE impact; "
                      "this factor is largely defined by shared indicators "
                      "and may be redundant")

        results.append({
            "factor": factor,
            "n_total_indicators": n_total,
            "n_exclusive_indicators": n_exclusive,
            "n_shared_indicators": n_shared,
            "pct_shared": round(pct_shared, 1),
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p_val, 6),
            "interpretation": interp,
        })

    sensitivity_df = pd.DataFrame(results)
    out_path = tables_dir / "indicator_overlap_sensitivity.csv"
    sensitivity_df.to_csv(out_path, index=False, encoding="utf-8")

    # -------------------------------------------------------------------
    # 4. Print summary
    # -------------------------------------------------------------------
    print(f"\n--- Indicator Overlap Sensitivity Analysis (H1) ---")
    for _, row in sensitivity_df.iterrows():
        rho_str = (f"{row['spearman_rho']:.3f}"
                   if pd.notna(row["spearman_rho"]) else "N/A")
        print(f"   {row['factor']:25s}: "
              f"{row['n_exclusive_indicators']}/{row['n_total_indicators']} exclusive, "
              f"{row['pct_shared']:.0f}% shared, "
              f"Spearman rho={rho_str}")
        if pd.isna(row["spearman_rho"]):
            print(f"     -> {row['interpretation']}")

    # Overall assessment
    computable = sensitivity_df.dropna(subset=["spearman_rho"])
    if len(computable) > 0:
        mean_rho = computable["spearman_rho"].mean()
        min_rho = computable["spearman_rho"].min()
        min_factor = computable.loc[computable["spearman_rho"].idxmin(), "factor"]
        print(f"\n   Mean Spearman rho across factors: {mean_rho:.3f}")
        print(f"   Most affected factor: {min_factor} (rho={min_rho:.3f})")
        if mean_rho >= 0.90:
            print("   CONCLUSION: Indicator overlap does NOT materially affect "
                  "factor rankings on average.")
        elif mean_rho >= 0.70:
            print("   CONCLUSION: Indicator overlap has MODERATE effect on "
                  "factor rankings; consider deduplication for robustness.")
        else:
            print("   CONCLUSION: Indicator overlap SIGNIFICANTLY affects "
                  "factor rankings; the effective number of independent "
                  "factors may be lower than 10.")

    n_no_exclusive = len(sensitivity_df[sensitivity_df["n_exclusive_indicators"] == 0])
    if n_no_exclusive > 0:
        print(f"\n   WARNING: {n_no_exclusive} factor(s) have NO exclusive indicators "
              "and are entirely composed of shared indicators.")

    print(f"\n   [OK] Saved indicator_overlap_sensitivity.csv "
          f"({len(sensitivity_df)} factors)")

    return sensitivity_df


def validate_config_weights(index_cfg, tol=0.01):
    """Validate that all weight configurations satisfy sum-to-one constraints.

    Checks:
      1. ESG pillar weights (E + S + G = 1.0)
      2. ESG category weights within each pillar
      3. Financial scoring category weights and indicator weights
      4. Market factor category weights and indicator weights
      5. Config-driven scoring section weights (operational, risk, value, growth, stability)
      6. Investor profile weights (all 10 factors sum to 1.0)

    Parameters
    ----------
    index_cfg : dict
        Full index configuration loaded from config/index_config.yaml.
    tol : float
        Tolerance for sum-to-one checks (default 0.01).

    Raises
    ------
    ValueError
        If any critical weight constraint is violated.
    """
    failures = []
    passes = []

    def _check_sum(group_name, weights_dict, expected=1.0):
        """Check that values in weights_dict sum to expected within tolerance."""
        total = sum(weights_dict.values())
        if abs(total - expected) <= tol:
            passes.append(f"  PASS: {group_name} = {total:.4f} (expected {expected:.2f})")
            return True
        else:
            msg = f"  FAIL: {group_name} = {total:.4f} (expected {expected:.2f}, delta={total - expected:+.4f})"
            failures.append(msg)
            return False

    print("\n--- Weight Configuration Validation (B7) ---")

    # 1. ESG pillar weights
    pillar_cfg = index_cfg.get("esg_index", {}).get("pillar_weights", {})
    pillar_defaults = {}
    for pillar in ["E", "S", "G"]:
        p = pillar_cfg.get(pillar, {})
        if isinstance(p, dict) and "default" in p:
            pillar_defaults[pillar] = p["default"]
    if pillar_defaults:
        _check_sum("ESG pillar weights (E+S+G)", pillar_defaults)

    # 2. ESG category weights within each pillar
    cat_weights_cfg = index_cfg.get("esg_index", {}).get("category_weights", {})
    for pillar in ["E", "S", "G"]:
        pillar_cats = cat_weights_cfg.get(pillar, {})
        cat_defaults = {}
        for k, v in pillar_cats.items():
            if k == "constraint":
                continue
            if isinstance(v, dict) and "default" in v:
                cat_defaults[k] = v["default"]
        if cat_defaults:
            _check_sum(f"ESG {pillar}-pillar category weights", cat_defaults)

    # 3. Financial scoring: category weights and indicator weights
    fin_cfg = index_cfg.get("financial_scoring", {}).get("categories", {})
    fin_cat_weights = {}
    for cat_name, cat_cfg in fin_cfg.items():
        wr = cat_cfg.get("weight_range", {})
        if isinstance(wr, dict) and "default" in wr:
            fin_cat_weights[cat_name] = wr["default"]
        # Check indicator weights within each category
        ind_weights_cfg = cat_cfg.get("indicator_weights", {})
        ind_defaults = {}
        for ind_name, ind_cfg in ind_weights_cfg.items():
            if isinstance(ind_cfg, dict) and "default" in ind_cfg:
                ind_defaults[ind_name] = ind_cfg["default"]
        if ind_defaults:
            _check_sum(f"Financial > {cat_name} indicator weights", ind_defaults)
    if fin_cat_weights:
        _check_sum("Financial scoring category weights", fin_cat_weights)

    # 4. Market factor: category weights and indicator weights
    mkt_cfg = index_cfg.get("market_factors", {}).get("categories", {})
    mkt_cat_weights = {}
    for cat_name, cat_cfg in mkt_cfg.items():
        wr = cat_cfg.get("weight_range", {})
        if isinstance(wr, dict) and "default" in wr:
            mkt_cat_weights[cat_name] = wr["default"]
        ind_weights_cfg = cat_cfg.get("indicator_weights", {})
        ind_defaults = {}
        for ind_name, ind_cfg in ind_weights_cfg.items():
            if isinstance(ind_cfg, dict) and "default" in ind_cfg:
                ind_defaults[ind_name] = ind_cfg["default"]
        if ind_defaults:
            _check_sum(f"Market > {cat_name} indicator weights", ind_defaults)
    if mkt_cat_weights:
        _check_sum("Market factor category weights", mkt_cat_weights)

    # 5. Config-driven scoring sections (operational, risk, value, growth, stability)
    config_sections = {
        "operational_quality": "Operational",
        "risk_adjusted_scoring": "Risk-adjusted",
        "value_scoring": "Value",
        "growth_scoring": "Growth",
        "stability_scoring": "Stability",
    }
    for config_key, label in config_sections.items():
        section = index_cfg.get(config_key, {})
        categories = section.get("categories", {})
        cat_weights = {}
        for cat_name, cat_cfg in categories.items():
            w = cat_cfg.get("weight", 0.0)
            cat_weights[cat_name] = w
            # Check indicator weights within each category
            indicators = cat_cfg.get("indicators", {})
            ind_weights = {}
            for ind_name, ind_cfg in indicators.items():
                if isinstance(ind_cfg, dict) and "weight" in ind_cfg:
                    ind_weights[ind_name] = ind_cfg["weight"]
            if ind_weights:
                _check_sum(f"{label} > {cat_name} indicator weights", ind_weights)
        if cat_weights:
            _check_sum(f"{label} scoring category weights", cat_weights)

    # 6. Investor profile weights
    profiles = index_cfg.get("preference_scoring", {}).get("investor_profiles", {})
    for profile_name, profile_weights in profiles.items():
        _check_sum(f"Investor profile '{profile_name}'", profile_weights)

    # Summary
    print(f"\n  Checks passed: {len(passes)}")
    for p in passes:
        print(p)

    if failures:
        print(f"\n  Checks FAILED: {len(failures)}")
        for f in failures:
            print(f)
        raise ValueError(
            f"Weight validation failed: {len(failures)} constraint(s) violated. "
            f"Fix config/index_config.yaml before proceeding."
        )
    else:
        print(f"\n  [OK] All {len(passes)} weight constraints validated successfully.")


def main():
    print("=" * 70)
    print("STEP 03: BUILD MULTI-FACTOR INDEX")
    print("=" * 70)

    # Load data
    data_path = PROJECT_ROOT / "data" / "processed" / "clean_data.csv"
    fallback_path = PROJECT_ROOT / "data" / "processed" / "real_data_clean.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError(f"Clean data not found.\nRun: python scripts/02_clean_data.py")
    print(f"[OK] Loaded {len(df)} companies, {len(df.columns)} columns")

    # -----------------------------------------------------------------------
    # IDENTIFY LARGE-CAP BENCHMARKS (scored together with mid-cap, separated after)
    # -----------------------------------------------------------------------
    # The index is designed for MID-CAP companies.  Large-cap benchmarks
    # (from config/index_config.yaml → universe.large_cap_benchmarks)
    # are included for robustness comparison.
    #
    # FIX (Issue C1): Previously, benchmarks were separated BEFORE scoring,
    # scored independently (N=4), then re-standardized against mid-cap stats.
    # This produced a DOUBLE transformation — z-scoring within N=4 distorts
    # distributions, and re-mapping doesn't fix it.
    #
    # CORRECT APPROACH: Score ALL companies together (mid-cap + benchmarks)
    # in a single pass so that the same normalization parameters (mean, std,
    # median, MAD) apply to everyone.  With N_midcap ≈ 56 and N_bench = 4,
    # the benchmarks represent only ~6.7% of the sample, so their influence
    # on normalization statistics is negligible.  After scoring, flag
    # benchmarks via `is_large_cap_benchmark` and separate for reporting.
    # -----------------------------------------------------------------------
    # Load config (needed before benchmark identification)
    index_cfg, _ = load_configs()

    # Validate all weight constraints at startup (B7)
    validate_config_weights(index_cfg)

    # Read large-cap benchmark tickers from config (externalized)
    LARGE_CAP_BENCHMARKS = set(
        index_cfg.get("universe", {}).get("large_cap_benchmarks", [])
    )
    if not LARGE_CAP_BENCHMARKS:
        logger.warning("No large_cap_benchmarks found in config; using empty set")
    if "ticker" in df.columns:
        is_benchmark = df["ticker"].isin(LARGE_CAP_BENCHMARKS)
        n_bench = is_benchmark.sum()
        if n_bench > 0:
            bench_tickers = df.loc[is_benchmark, "ticker"].tolist()
            print(f"[INFO] Found {n_bench} large-cap benchmarks from config "
                  f"({', '.join(bench_tickers)}) — scoring together with mid-cap")
            print(f"       Total universe: {len(df)} companies "
                  f"({len(df) - n_bench} mid-cap + {n_bench} benchmark)")
            print(f"       Benchmark influence on normalization: "
                  f"{100 * n_bench / len(df):.1f}% (negligible)")
    else:
        n_bench = 0

    # 1. ESG Index (with SASB sector-materiality pillar weights)
    print("\n1. Building ESG composite index...")
    esg_indicators = [c for c in ESG_COLS if c in df.columns]
    builder = CompositeIndexBuilder(index_cfg)

    # Log materiality config being applied
    materiality_map = _load_materiality_map()
    if "sector" in df.columns and materiality_map:
        unique_sectors = df["sector"].dropna().unique()
        matched = [s for s in unique_sectors if s in materiality_map]
        print(f"   Sector-materiality weights: {len(matched)}/{len(unique_sectors)} "
              f"sectors matched from SASB config")
        for sec in sorted(matched):
            mw = materiality_map[sec]
            print(f"     {sec:28s}: E={mw['E']:.2f}  S={mw['S']:.2f}  G={mw['G']:.2f}")
        unmatched = [s for s in unique_sectors if s not in materiality_map]
        if unmatched:
            dw = materiality_map.get("default", {})
            print(f"   Using default weights for: {', '.join(sorted(unmatched))}")
            print(f"     {'default':28s}: E={dw.get('E', 0.35):.2f}  "
                  f"S={dw.get('S', 0.35):.2f}  G={dw.get('G', 0.30):.2f}")

    df = builder.build(df, indicator_cols=esg_indicators, sector_column="sector", scale_to_score=False)
    for col in ["ESG_composite", "E_score", "S_score", "G_score"]:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
                  f"range=[{df[col].min():.1f}, {df[col].max():.1f}]")

    # 2. Financial Score
    print("\n2. Computing financial scores...")
    financial_scorer = FinancialScorer(index_cfg)
    df = financial_scorer.compute_financial_score(df, scale_to_score=False)
    if "financial_score" in df.columns:
        print(f"   financial_score: mean={df['financial_score'].mean():.1f}, "
              f"std={df['financial_score'].std():.1f}")

    # 3. Market Score
    print("\n3. Computing market scores...")
    market_scorer = MarketFactorScorer(index_cfg)
    df = market_scorer.compute_market_score(df, scale_to_score=False)
    if "market_score" in df.columns:
        print(f"   market_score: mean={df['market_score'].mean():.1f}, "
              f"std={df['market_score'].std():.1f}")

    # 4. Operational Score (config-driven via operational_quality section)
    print("\n4. Computing operational scores (config-driven)...")
    df = build_config_driven_score(
        df,
        index_cfg,
        "operational_quality",
        "operational_score",
        scale_to_score=False,
        hierarchical_zscore=True,
        sector_adjust=True,
    )
    print(f"   operational_score: mean={df['operational_score'].mean():.1f}, "
          f"std={df['operational_score'].std():.1f}")

    # 5. Additional factor scores (all config-driven)
    print("\n5. Computing additional factor scores (config-driven)...")
    df = build_config_driven_score(
        df,
        index_cfg,
        "risk_adjusted_scoring",
        "risk_adjusted_score",
        scale_to_score=False,
        hierarchical_zscore=True,
        sector_adjust=True,
    )
    df = build_config_driven_score(
        df,
        index_cfg,
        "value_scoring",
        "value_score",
        scale_to_score=False,
        hierarchical_zscore=True,
        sector_adjust=True,
    )
    df = build_config_driven_score(
        df,
        index_cfg,
        "growth_scoring",
        "growth_score",
        scale_to_score=False,
        hierarchical_zscore=True,
        sector_adjust=True,
    )
    # stability_score: fallback price_volatility_30d -> price_volatility
    df = build_config_driven_score(
        df, index_cfg, "stability_scoring", "stability_score",
        fallbacks={"price_volatility_30d": "price_volatility"},
        scale_to_score=False,
        hierarchical_zscore=True,
        sector_adjust=True,
    )
    for col in ["risk_adjusted_score", "value_score", "growth_score", "stability_score"]:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}")

    # 6. Similarity Rank (ESG indicator profile similarity — Issues M3 & M3b)
    print("\n6. Computing ESG indicator profile similarity (all 32 ESG indicators)...")
    df, sim_matrix = build_similarity_rank(df)
    print(f"   similarity_rank (ESG indicator similarity): "
          f"mean={df['similarity_rank'].mean():.3f}, "
          f"std={df['similarity_rank'].std():.3f}")

    # 7. Sector Position
    print("\n7. Computing sector positions...")
    df = build_sector_position(df)
    print(f"   sector_position: mean={df['sector_position'].mean():.3f}")

    # 7b. Standardize all factor scores to consistent scale (SINGLE transformation)
    #     Scores arrive as raw z-score composites (scale_to_score=False upstream).
    #     This is the ONLY place where scores are mapped to the 0-100 scale.
    #     Parameters read from config: universe.scoring.restandardize_mean/std/clip
    scoring_cfg = index_cfg.get("universe", {}).get("scoring", {})
    restd_mean = scoring_cfg.get("restandardize_mean", 50)
    restd_std = scoring_cfg.get("restandardize_std", 10)
    score_clip_min = scoring_cfg.get("score_clip_min", 0)
    score_clip_max = scoring_cfg.get("score_clip_max", 100)
    print(f"\n7b. Re-standardizing all factor scores to mean={restd_mean}, std={restd_std}...")
    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "growth_score", "value_score", "stability_score",
                  "similarity_rank", "sector_position"]
    for col in score_cols:
        if col not in df.columns:
            continue
        if df[col].isna().all():
            logger.warning(f"Column {col} is all NaN, skipping re-standardization")
            continue
        if df[col].std() > 1e-10:
            df[col] = ((df[col] - df[col].mean()) / df[col].std()) * restd_std + restd_mean
            df[col] = df[col].clip(score_clip_min, score_clip_max)
    for col in score_cols:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}, "
                  f"range=[{df[col].min():.1f}, {df[col].max():.1f}]")

    # 7c. Sector-relative blending THEN monotonicity correction
    # (Apply sector blend FIRST so monotonicity correction operates on final factor scores)
    blend_cols = ["financial_score", "operational_score", "growth_score", "value_score", "stability_score"]
    df = _apply_sector_blend(df, blend_cols, sector_col="sector", w_cross=0.7, w_sector=0.3)

    return_proxy_col = _select_return_proxy(df)
    factor_cols_for_mono = [
        "ESG_composite", "financial_score", "market_score", "operational_score",
        "risk_adjusted_score", "value_score", "growth_score", "stability_score",
    ]
    indicator_map = _build_indicator_map(index_cfg, df)
    if return_proxy_col is not None:
        print(f"\n7c. Applying monotonicity correction using return proxy: {return_proxy_col}...")
        df = _apply_monotonicity_correction(
            df,
            factor_cols_for_mono,
            indicator_map,
            return_proxy_col,
            restd_mean,
            restd_std,
            score_clip_min,
            score_clip_max,
        )
    else:
        print("\n7c. Skipping monotonicity correction: no return proxy column found")

    # Save monotonicity diagnostics
    mono_out = PROJECT_ROOT / "reports" / "tables" / "factor_monotonicity_diagnostic.csv"
    mono_df = _save_factor_monotonicity_diagnostic(df, factor_cols_for_mono, return_proxy_col, mono_out)
    n_mono = int(mono_df["is_monotonic"].sum()) if "is_monotonic" in mono_df.columns else 0
    print(f"   [OK] Saved monotonicity diagnostics to {mono_out} ({n_mono}/{len(mono_df)} monotonic)")

    # -----------------------------------------------------------------------
    # FACTOR SCORE INDICATOR SOURCES SUMMARY
    # -----------------------------------------------------------------------
    print("\n--- Factor Score Indicator Sources ---")
    print("  Steps 1-3 (ESG, Financial, Market): Config-driven (index_config.yaml)")
    print("  Steps 4-5 (Operational, Risk, Value, Growth, Stability): Config-driven (index_config.yaml)")
    print("  Step 6 (Similarity): Cosine similarity on all 32 ESG indicators (10 Env + 10 Soc + 12 Gov)")
    print("  Step 7 (Sector Position): Within-sector percentile rank of raw financial indicators")

    # 8. PCA Weight Rationale
    print("\n8. Deriving PCA-based weight rationale...")
    pca_rationale = derive_pca_weight_rationale(df)
    if pca_rationale:
        print("   PCA-derived factor contributions:")
        for factor, info in sorted(pca_rationale.items(), key=lambda x: -x[1]["pca_contribution"]):
            lo, hi = info["suggested_weight_range"]
            print(f"     {factor:25s}: {info['pca_contribution']:.3f}  -> range [{lo:.2f}, {hi:.2f}]")

    # 8b. Indicator Overlap Report (H2 audit)
    print("\n8b. Generating indicator overlap report (H2 audit)...")
    generate_indicator_overlap_report(index_cfg)

    # 8c. Indicator Overlap Sensitivity Analysis (H1 audit)
    print("\n8c. Running indicator overlap sensitivity analysis (H1 audit)...")
    overlap_sensitivity_analysis(df, index_cfg)

    # 9. Preference Scores (3 profiles, using all 10 factors)
    print("\n9. Computing preference scores (3 investor profiles, 10 factors)...")
    scorer = PreferenceScorer(index_cfg)
    for profile in ["esg_first", "balanced", "financial_first"]:
        df[f"pref_{profile}"] = scorer.compute_preference_score(
            df,
            investor_profile=profile,
            financial_score_col="financial_score",
            similarity_rank_col="similarity_rank",
            sector_position_col="sector_position",
        )
        print(f"   pref_{profile}: mean={df[f'pref_{profile}'].mean():.1f}, "
              f"std={df[f'pref_{profile}'].std():.1f}")

    # -----------------------------------------------------------------------
    # 9b. PREFERENCE SCORES EXCLUDING MARKET_SCORE (ex_market variants)
    # -----------------------------------------------------------------------
    # market_score includes momentum sub-scores (price_momentum_1m/3m/6m) that
    # are also used as return proxies in evaluation scripts (05, 06, 08).
    # Using market_score in portfolio selection AND momentum as the return proxy
    # creates a circular dependency that inflates IC, Sharpe, and alpha metrics.
    #
    # These "ex_market" preference scores redistribute market_score's weight
    # proportionally across the remaining 9 factors, preserving the relative
    # weight ratios.  They serve as the PRIMARY selection criterion for
    # unbiased portfolio evaluation.
    # -----------------------------------------------------------------------
    print("\n9b. Computing ex-market preference scores (circularity fix)...")
    _profile_configs = index_cfg.get("preference_scoring", {}).get("investor_profiles", {})
    for profile in ["esg_first", "balanced", "financial_first"]:
        profile_weights = _profile_configs.get(profile, {}).copy()
        mkt_w = profile_weights.pop("market_score", 0.0)
        if mkt_w > 0 and sum(profile_weights.values()) > 0:
            # Redistribute market_score weight proportionally to remaining factors
            remaining_total = sum(profile_weights.values())
            profile_weights = {k: v + mkt_w * (v / remaining_total)
                               for k, v in profile_weights.items()}
        # Defensive assertion: market_score must not survive into the weight dict.
        # Its presence would re-introduce momentum circularity when these
        # ex-market scores are validated against momentum return proxies.
        assert "market_score" not in profile_weights, \
            f"market_score weight was not removed for ex-market variant '{profile}'"
        # Compute score manually using same aggregation logic as PreferenceScorer
        # but with the modified weight dict (no market_score)
        from src.similarity.preference_scoring import SCORE_COLUMN_MAP
        total_w = sum(profile_weights.values())
        if total_w > 0:
            profile_weights = {k: v / total_w for k, v in profile_weights.items()}
        ex_score = pd.Series(0.0, index=df.index)
        agg_mode = index_cfg.get("preference_scoring", {}).get("aggregation_mode", "rank")
        for component, weight in profile_weights.items():
            if weight <= 0:
                continue
            col = SCORE_COLUMN_MAP.get(component, component)
            if col in df.columns:
                vals = df[col].fillna(df[col].median() if df[col].notna().any() else 50)
                if component in ("similarity_rank", "sector_position"):
                    if vals.max() <= 1.0:
                        vals = vals * 100
                vals = PreferenceScorer._normalize_factor(vals, agg_mode)
                ex_score += weight * vals
        df[f"pref_{profile}_ex_market"] = ex_score.clip(0, 100)
        orig_col = f"pref_{profile}"
        ex_col = f"pref_{profile}_ex_market"
        corr = df[[orig_col, ex_col]].corr().iloc[0, 1]
        print(f"   {ex_col}: mean={df[ex_col].mean():.1f}, "
              f"std={df[ex_col].std():.1f}, corr_with_original={corr:.3f}")

    # -----------------------------------------------------------------------
    # CIRCULARITY FIX: Make ex-market scores PRIMARY (Issue C1)
    # -----------------------------------------------------------------------
    # market_score contains price_momentum_1m/3m/6m as sub-indicators.
    # Evaluation scripts (05, 06, 08) use these same momentum columns as
    # return proxies.  Using market_score in portfolio selection AND momentum
    # as the return proxy is circular.
    #
    # Resolution: the ex_market variants (which exclude market_score) become
    # the PRIMARY preference scores.  The originals are kept with a
    # "_with_market" suffix for transparency / backward compatibility.
    # -----------------------------------------------------------------------
    print("\n   Applying circularity fix: ex-market scores become primary...")
    for profile in ["esg_first", "balanced", "financial_first"]:
        orig_col = f"pref_{profile}"
        ex_col = f"pref_{profile}_ex_market"
        contaminated_col = f"pref_{profile}_with_market"
        if orig_col in df.columns and ex_col in df.columns:
            # Rename: original (contaminated) → _with_market
            df.rename(columns={orig_col: contaminated_col}, inplace=True)
            # Rename: ex_market (clean) → primary name
            df.rename(columns={ex_col: orig_col}, inplace=True)
            corr = df[[orig_col, contaminated_col]].corr().iloc[0, 1]
            print(f"   {orig_col}: now ex-market (clean); "
                  f"{contaminated_col}: original (contaminated), corr={corr:.3f}")

    # -----------------------------------------------------------------------
    # FLAG LARGE-CAP BENCHMARKS (Issue C1 fix: scored together, flagged after)
    # -----------------------------------------------------------------------
    # All companies (mid-cap + benchmarks) have now been scored together in a
    # single normalization pass.  Add the flag column so downstream analysis
    # can filter benchmarks vs mid-cap as needed.
    # -----------------------------------------------------------------------
    if "ticker" in df.columns:
        df["is_large_cap_benchmark"] = df["ticker"].isin(LARGE_CAP_BENCHMARKS)
    else:
        df["is_large_cap_benchmark"] = False

    if n_bench > 0:
        bench_rows = df[df["is_large_cap_benchmark"]]
        midcap_rows = df[~df["is_large_cap_benchmark"]]
        print(f"\n--- Benchmark vs Mid-Cap Score Summary (Issue C1: single-pass scoring) ---")
        print(f"     Total dataset: {len(df)} companies "
              f"({len(midcap_rows)} mid-cap + {len(bench_rows)} benchmark)")
        summary_cols = [c for c in score_cols if c in df.columns]
        for col in summary_cols:
            mc_mean = midcap_rows[col].mean()
            mc_std = midcap_rows[col].std()
            bm_vals = bench_rows[col].tolist()
            bm_str = ", ".join(f"{v:.1f}" for v in bm_vals)
            print(f"     {col:25s}: mid-cap mean={mc_mean:.1f} std={mc_std:.1f} | "
                  f"benchmarks=[{bm_str}]")

    # --- Data Provenance Summary ---
    print("\n--- Data Provenance Summary ---")
    if "esg_data_source" in df.columns:
        src_counts = df["esg_data_source"].value_counts()
        print("  ESG data sources:")
        for src, cnt in src_counts.items():
            print(f"    {src}: {cnt} companies ({100*cnt/len(df):.0f}%)")

    # R&D coverage
    rd_cols = ["r_d_intensity", "r_d_expenditure"]
    for rc in rd_cols:
        if rc in df.columns:
            n_real = df[rc].notna().sum()
            print(f"  {rc} coverage: {n_real}/{len(df)} ({100*n_real/len(df):.0f}%)")

    # Large-cap benchmark flag
    if "is_large_cap_benchmark" in df.columns:
        n_bm = df["is_large_cap_benchmark"].sum()
        print(f"  Large-cap benchmarks: {n_bm}")
        print(f"  Mid-cap universe: {len(df) - n_bm}")

    # Save indexed data
    outpath = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    df.to_csv(outpath, index=False, encoding="utf-8")
    print(f"\n[OK] Indexed data saved to {outpath}")
    print(f"     {len(df)} companies, {len(df.columns)} columns")

    # Save similarity matrix
    if sim_matrix is not None:
        sim_path = PROJECT_ROOT / "data" / "processed" / "similarity_matrix.csv"
        sim_matrix.to_csv(sim_path, encoding="utf-8")
        print(f"[OK] Similarity matrix saved to {sim_path}")

    # Save rankings
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    # After the circularity fix rename:
    #   pref_balanced            = clean (ex-market) — PRIMARY sort key
    #   pref_balanced_with_market = original (contaminated) — kept for audit
    ranking_cols = ["ticker", "company_name", "sector", "country",
                    "is_large_cap_benchmark",
                    "ESG_composite", "financial_score", "market_score",
                    "operational_score", "risk_adjusted_score", "value_score",
                    "growth_score", "stability_score",
                    "pref_balanced", "pref_esg_first", "pref_financial_first",
                    "pref_balanced_with_market", "pref_esg_first_with_market",
                    "pref_financial_first_with_market"]
    avail_ranking = [c for c in ranking_cols if c in df.columns]
    rankings = df[avail_ranking].sort_values("pref_balanced", ascending=False)
    rankings["rank"] = range(1, len(rankings) + 1)
    rankings.to_csv(tables_dir / "company_rankings.csv", index=False, encoding="utf-8")
    print(f"[OK] Rankings saved to {tables_dir / 'company_rankings.csv'}")

    # Score summary by sector
    score_cols = [c for c in ["ESG_composite", "financial_score", "market_score",
                               "operational_score", "risk_adjusted_score",
                               "value_score", "growth_score", "stability_score",
                               "pref_balanced"] if c in df.columns]
    if "sector" in df.columns and score_cols:
        sector_summary = df.groupby("sector")[score_cols].agg(["mean", "std", "count"])
        sector_summary.to_csv(tables_dir / "sector_score_summary.csv", encoding="utf-8")

    print("\n[DONE] Next: python scripts/04_statistical_tests.py")


if __name__ == "__main__":
    main()
