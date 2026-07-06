"""Tests for scripts/02_clean_data.py — data cleaning and standardization functions.

Uses importlib to import the module since the filename starts with a digit.
All tests use small synthetic DataFrames (5-10 rows).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script module via importlib (numeric prefix prevents normal import)
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "02_clean_data.py"

spec = importlib.util.spec_from_file_location("clean_data", str(_SCRIPT_PATH))
clean_data = importlib.util.module_from_spec(spec)
# Temporarily suppress the chdir / path manipulation side effects
_orig_chdir = __import__("os").chdir
__import__("os").chdir = lambda *a, **kw: None
spec.loader.exec_module(clean_data)
__import__("os").chdir = _orig_chdir

# Aliases for convenience
classify_variable = clean_data.classify_variable
adaptive_winsorize = clean_data.adaptive_winsorize
detect_outliers_iqr = clean_data.detect_outliers_iqr
detect_outliers_mad = clean_data.detect_outliers_mad
detect_outliers_zscore = clean_data.detect_outliers_zscore
impute_sector_median = clean_data.impute_sector_median
validate_binary_variables = clean_data.validate_binary_variables
log_transform_skewed = clean_data.log_transform_skewed
VARIABLE_TYPES = clean_data.VARIABLE_TYPES


# =====================================================================
# classify_variable
# =====================================================================
class TestClassifyVariable:
    """Tests for classify_variable(col_name)."""

    def test_known_binary(self):
        assert classify_variable("carbon_reduction_target") == "binary"
        assert classify_variable("human_rights_policy") == "binary"
        assert classify_variable("anti_corruption_policy") == "binary"

    def test_known_ordinal(self):
        assert classify_variable("board_size") == "ordinal"
        assert classify_variable("esg_risk_rating") == "ordinal"

    def test_known_bounded_pct(self):
        assert classify_variable("renewable_energy_pct") == "bounded_pct"
        assert classify_variable("gender_diversity_pct") == "bounded_pct"
        assert classify_variable("profit_margins") == "bounded_pct"

    def test_known_ratio(self):
        assert classify_variable("trailing_pe") == "ratio"
        assert classify_variable("debt_to_equity") == "ratio"
        assert classify_variable("payout_ratio") == "ratio"

    def test_known_count(self):
        assert classify_variable("market_cap") == "count"
        assert classify_variable("total_revenue") == "count"
        assert classify_variable("employees") == "count"

    def test_known_rate(self):
        assert classify_variable("revenue_growth") == "rate"
        assert classify_variable("roa") == "rate"
        assert classify_variable("dividend_yield") == "rate"

    def test_known_continuous(self):
        assert classify_variable("beta") == "continuous"
        assert classify_variable("price_volatility") == "continuous"
        assert classify_variable("sharpe_ratio_1y") == "continuous"

    def test_auto_detect_pct_suffix(self):
        """Unknown column ending in _pct should be auto-classified as bounded_pct."""
        assert classify_variable("unknown_metric_pct") == "bounded_pct"

    def test_auto_detect_percent_suffix(self):
        assert classify_variable("some_percent") == "bounded_pct"

    def test_auto_detect_policy_suffix(self):
        """Unknown column ending in _policy should be binary."""
        assert classify_variable("sustainability_policy") == "binary"

    def test_auto_detect_target_suffix(self):
        """Unknown column ending in _target should be binary."""
        assert classify_variable("net_zero_target") == "binary"

    def test_unknown_falls_to_continuous(self):
        """Completely unknown column defaults to continuous."""
        assert classify_variable("some_random_col_xyz") == "continuous"

    def test_all_variable_types_entries_are_valid(self):
        """Every entry in VARIABLE_TYPES maps to a recognised type."""
        valid_types = {"binary", "ordinal", "bounded_pct", "ratio", "count", "rate", "continuous"}
        for col, vtype in VARIABLE_TYPES.items():
            assert vtype in valid_types, f"{col} has unexpected type '{vtype}'"


# =====================================================================
# adaptive_winsorize
# =====================================================================
class TestAdaptiveWinsorize:
    """Tests for adaptive_winsorize(df, cols)."""

    def test_binary_clipped_to_0_1(self):
        """Binary columns should be clipped to 0/1."""
        df = pd.DataFrame({"carbon_reduction_target": [0.0, 1.0, 0.3, 0.8, -0.5, 1.5]})
        result = adaptive_winsorize(df.copy(), ["carbon_reduction_target"])
        vals = result["carbon_reduction_target"]
        assert vals.min() >= 0.0
        assert vals.max() <= 1.0
        assert set(vals.unique()).issubset({0.0, 1.0})

    def test_ordinal_rounded(self):
        """Ordinal columns should be rounded to nearest integer."""
        df = pd.DataFrame({"board_size": [7.3, 8.8, 9.1, 10.6, 11.2]})
        result = adaptive_winsorize(df.copy(), ["board_size"])
        assert all(result["board_size"] == result["board_size"].round())

    def test_bounded_pct_clipped_0_100(self):
        """bounded_pct columns are clipped to [0, 100]."""
        df = pd.DataFrame({"renewable_energy_pct": [-5.0, 50.0, 110.0, 75.0, 25.0]})
        result = adaptive_winsorize(df.copy(), ["renewable_energy_pct"])
        assert result["renewable_energy_pct"].min() >= 0.0
        assert result["renewable_energy_pct"].max() <= 100.0

    def test_ratio_aggressive_winsorize(self):
        """Ratio columns winsorized at 2.5/97.5 percentiles."""
        rng = np.random.RandomState(42)
        values = np.concatenate([rng.uniform(5, 50, 100), [1000.0, -500.0]])
        df = pd.DataFrame({"trailing_pe": values})
        result = adaptive_winsorize(df.copy(), ["trailing_pe"])
        # The extreme outliers should be clipped
        assert result["trailing_pe"].max() < 1000.0
        assert result["trailing_pe"].min() > -500.0

    def test_continuous_standard_winsorize(self):
        """Continuous columns winsorized at 1/99 percentiles."""
        rng = np.random.RandomState(42)
        values = np.concatenate([rng.normal(0, 1, 200), [50.0, -50.0]])
        df = pd.DataFrame({"beta": values})
        result = adaptive_winsorize(df.copy(), ["beta"])
        assert result["beta"].max() < 50.0
        assert result["beta"].min() > -50.0

    def test_count_standard_winsorize(self):
        """Count columns winsorized at 1/99 percentiles."""
        rng = np.random.RandomState(42)
        values = np.concatenate([rng.uniform(1e6, 1e9, 200), [1e15]])
        df = pd.DataFrame({"market_cap": values})
        result = adaptive_winsorize(df.copy(), ["market_cap"])
        assert result["market_cap"].max() < 1e15

    def test_missing_column_skipped(self):
        """Columns not in the DataFrame are silently skipped."""
        df = pd.DataFrame({"beta": [1.0, 2.0, 3.0]})
        result = adaptive_winsorize(df.copy(), ["beta", "nonexistent_col"])
        assert "beta" in result.columns
        assert "nonexistent_col" not in result.columns

    def test_nan_preserved(self):
        """NaN values are preserved through winsorization."""
        df = pd.DataFrame({"beta": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = adaptive_winsorize(df.copy(), ["beta"])
        assert result["beta"].isna().sum() == 2


# =====================================================================
# detect_outliers_iqr
# =====================================================================
class TestDetectOutliersIQR:
    """Tests for detect_outliers_iqr(series, k)."""

    def test_obvious_outlier_detected(self):
        """A value far from the IQR should be flagged."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        mask = detect_outliers_iqr(s)
        assert mask.iloc[-1] is True or mask.iloc[-1] == True  # 100 is an outlier

    def test_no_outliers_in_uniform_data(self):
        """Data within a narrow range should have no IQR outliers."""
        s = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        mask = detect_outliers_iqr(s)
        assert mask.sum() == 0

    def test_both_tails_detected(self):
        """Outliers on both lower and upper tails are detected."""
        s = pd.Series([-100, 1, 2, 3, 4, 5, 6, 7, 8, 100])
        mask = detect_outliers_iqr(s)
        assert mask.iloc[0] == True   # -100
        assert mask.iloc[-1] == True  # 100

    def test_custom_k(self):
        """Stricter k=3 should flag fewer outliers than k=1.5."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 20])
        mask_standard = detect_outliers_iqr(s, k=1.5)
        mask_extreme = detect_outliers_iqr(s, k=3.0)
        assert mask_extreme.sum() <= mask_standard.sum()

    def test_returns_boolean_series(self):
        s = pd.Series([1, 2, 3, 4, 5])
        mask = detect_outliers_iqr(s)
        assert mask.dtype == bool

    def test_constant_series(self):
        """Constant series (IQR=0) should flag nothing (all within bounds)."""
        s = pd.Series([5, 5, 5, 5, 5])
        mask = detect_outliers_iqr(s)
        assert mask.sum() == 0


# =====================================================================
# detect_outliers_mad
# =====================================================================
class TestDetectOutliersMAD:
    """Tests for detect_outliers_mad(series, threshold)."""

    def test_obvious_outlier_detected(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 200])
        mask = detect_outliers_mad(s)
        assert mask.iloc[-1] == True

    def test_no_outliers_in_uniform_data(self):
        s = pd.Series([10, 11, 12, 13, 14, 15])
        mask = detect_outliers_mad(s)
        assert mask.sum() == 0

    def test_constant_series_returns_all_false(self):
        """When MAD ≈ 0, should return all False (no outliers)."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        mask = detect_outliers_mad(s)
        assert mask.sum() == 0

    def test_lower_threshold_detects_more(self):
        """Lower threshold should flag more potential outliers."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 25])
        mask_strict = detect_outliers_mad(s, threshold=3.5)
        mask_loose = detect_outliers_mad(s, threshold=2.0)
        assert mask_loose.sum() >= mask_strict.sum()

    def test_returns_boolean_series(self):
        s = pd.Series([1, 2, 3, 4, 5])
        mask = detect_outliers_mad(s)
        assert mask.dtype == bool

    def test_symmetric_outliers(self):
        """Both positive and negative outliers should be detected."""
        s = pd.Series([-100, 1, 2, 3, 4, 5, 6, 7, 8, 100])
        mask = detect_outliers_mad(s)
        assert mask.iloc[0] == True   # -100
        assert mask.iloc[-1] == True  # 100


# =====================================================================
# detect_outliers_zscore
# =====================================================================
class TestDetectOutliersZscore:
    """Tests for detect_outliers_zscore(series, threshold)."""

    def test_obvious_outlier_detected(self):
        # Need enough normal data so the outlier's z-score exceeds 3.0
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 10000])
        mask = detect_outliers_zscore(s)
        assert mask.iloc[-1] == True

    def test_no_outliers_in_narrow_range(self):
        s = pd.Series([10.0, 10.1, 10.2, 10.3, 10.4, 10.5])
        mask = detect_outliers_zscore(s)
        assert mask.sum() == 0

    def test_constant_series_returns_all_false(self):
        """Zero std should return all False."""
        s = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
        mask = detect_outliers_zscore(s)
        assert mask.sum() == 0

    def test_threshold_sensitivity(self):
        """Lower threshold catches more outliers."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 30])
        mask_strict = detect_outliers_zscore(s, threshold=3.0)
        mask_loose = detect_outliers_zscore(s, threshold=2.0)
        assert mask_loose.sum() >= mask_strict.sum()

    def test_returns_boolean_series(self):
        s = pd.Series([1, 2, 3])
        mask = detect_outliers_zscore(s)
        assert mask.dtype == bool

    def test_both_tails(self):
        """Extreme values on both sides should be flagged."""
        # Use many normal data points so two extreme outliers are clearly > 3 sigma
        rng = np.random.RandomState(42)
        base = rng.normal(50, 5, 100).tolist()
        s = pd.Series([-500] + base + [600])
        mask = detect_outliers_zscore(s)
        assert mask.iloc[0] == True
        assert mask.iloc[-1] == True


# =====================================================================
# impute_sector_median
# =====================================================================
class TestImputeSectorMedian:
    """Tests for impute_sector_median(df, cols)."""

    def test_fills_nan_with_sector_median(self):
        """NaN values should be filled with sector median."""
        df = pd.DataFrame({
            "sector": ["Tech", "Tech", "Tech", "Health", "Health", "Health"],
            "roa": [0.10, np.nan, 0.30, 0.05, np.nan, 0.15],
        })
        result = impute_sector_median(df.copy(), ["roa"])
        assert result["roa"].isna().sum() == 0
        # Tech sector median of [0.10, 0.30] = 0.20 → NaN filled with 0.20
        assert result["roa"].iloc[1] == pytest.approx(0.20, abs=0.01)

    def test_fills_nan_global_median_when_no_sector(self):
        """When no sector column exists, falls back to global median."""
        df = pd.DataFrame({
            "roa": [0.10, np.nan, 0.30, 0.05, np.nan, 0.15],
        })
        result = impute_sector_median(df.copy(), ["roa"])
        assert result["roa"].isna().sum() == 0

    def test_binary_uses_mode(self):
        """Binary columns should be imputed with sector mode, not median."""
        df = pd.DataFrame({
            "sector": ["Tech", "Tech", "Tech", "Tech", "Health", "Health"],
            "carbon_reduction_target": [1.0, 1.0, np.nan, 1.0, 0.0, np.nan],
        })
        result = impute_sector_median(df.copy(), ["carbon_reduction_target"])
        assert result["carbon_reduction_target"].isna().sum() == 0
        # Tech mode is 1.0
        assert result["carbon_reduction_target"].iloc[2] == 1.0

    def test_ordinal_imputed_and_rounded(self):
        """Ordinal columns should be imputed with rounded sector median."""
        df = pd.DataFrame({
            "sector": ["Tech", "Tech", "Tech", "Health", "Health", "Health"],
            "board_size": [9.0, 11.0, np.nan, 7.0, np.nan, 8.0],
        })
        result = impute_sector_median(df.copy(), ["board_size"])
        assert result["board_size"].isna().sum() == 0
        # Values should be integers
        assert all(result["board_size"] == result["board_size"].round())

    def test_no_nans_untouched(self):
        """Columns with no NaN values remain unchanged."""
        df = pd.DataFrame({
            "sector": ["Tech", "Health"],
            "beta": [1.1, 0.9],
        })
        original = df["beta"].copy()
        result = impute_sector_median(df.copy(), ["beta"])
        pd.testing.assert_series_equal(result["beta"], original)

    def test_all_nan_sector_falls_to_global(self):
        """If an entire sector is NaN, global median is used."""
        df = pd.DataFrame({
            "sector": ["Tech", "Tech", "Health", "Health"],
            "roa": [np.nan, np.nan, 0.10, 0.20],
        })
        result = impute_sector_median(df.copy(), ["roa"])
        assert result["roa"].isna().sum() == 0
        # Tech rows should get global median of [0.10, 0.20] = 0.15
        assert result["roa"].iloc[0] == pytest.approx(0.15, abs=0.01)


# =====================================================================
# validate_binary_variables
# =====================================================================
class TestValidateBinaryVariables:
    """Tests for validate_binary_variables(df, cols)."""

    def test_values_become_0_or_1(self):
        """After validation, binary columns contain only 0.0 or 1.0."""
        df = pd.DataFrame({
            "carbon_reduction_target": [0.0, 1.0, 0.3, 0.8, 1.5],
        })
        result = validate_binary_variables(df.copy(), ["carbon_reduction_target"])
        vals = result["carbon_reduction_target"].dropna()
        assert set(vals.unique()).issubset({0.0, 1.0})

    def test_threshold_at_0_5(self):
        """Values > 0.5 map to 1.0, values <= 0.5 map to 0.0."""
        df = pd.DataFrame({
            "carbon_reduction_target": [0.0, 0.49, 0.51, 1.0],
        })
        result = validate_binary_variables(df.copy(), ["carbon_reduction_target"])
        expected = [0.0, 0.0, 1.0, 1.0]
        np.testing.assert_array_equal(result["carbon_reduction_target"].values, expected)

    def test_nan_preserved(self):
        """NaN values in binary columns remain NaN."""
        df = pd.DataFrame({
            "carbon_reduction_target": [0.0, np.nan, 1.0, np.nan],
        })
        result = validate_binary_variables(df.copy(), ["carbon_reduction_target"])
        assert result["carbon_reduction_target"].isna().sum() == 2

    def test_non_binary_columns_unchanged(self):
        """Non-binary columns passed to the function remain untouched."""
        df = pd.DataFrame({
            "beta": [1.2, 0.8, 1.5, 0.3, 2.1],
        })
        original = df["beta"].copy()
        result = validate_binary_variables(df.copy(), ["beta"])
        pd.testing.assert_series_equal(result["beta"], original)

    def test_missing_column_skipped(self):
        """Columns not in DataFrame are silently skipped."""
        df = pd.DataFrame({"carbon_reduction_target": [0.0, 1.0]})
        result = validate_binary_variables(df.copy(), ["carbon_reduction_target", "nonexistent"])
        assert "carbon_reduction_target" in result.columns


# =====================================================================
# log_transform_skewed
# =====================================================================
class TestLogTransformSkewed:
    """Tests for log_transform_skewed(df, cols, skew_threshold)."""

    def test_creates_log_column_for_skewed_count(self):
        """Highly right-skewed count columns get a _log column."""
        rng = np.random.RandomState(42)
        # Create obviously right-skewed data: mostly small, a few very large
        values = np.concatenate([rng.uniform(100, 1000, 90), [1e8, 1e9, 1e10] * 3, [1e12]])
        df = pd.DataFrame({"market_cap": values})
        result = log_transform_skewed(df.copy(), ["market_cap"])
        assert "market_cap_log" in result.columns

    def test_log_values_are_log1p(self):
        """The _log column should equal np.log1p of the original."""
        values = [100, 200, 300, 400, 500, 1e8, 2e8, 3e8, 4e8, 5e8]
        df = pd.DataFrame({"total_revenue": values})
        # Force high skew by using very skewed data
        rng = np.random.RandomState(42)
        skewed = np.concatenate([rng.exponential(1e6, 100), [1e12]])
        df = pd.DataFrame({"total_revenue": skewed})
        result = log_transform_skewed(df.copy(), ["total_revenue"])
        if "total_revenue_log" in result.columns:
            expected = np.log1p(result["total_revenue"])
            np.testing.assert_array_almost_equal(
                result["total_revenue_log"].values, expected.values
            )

    def test_no_log_for_symmetric_data(self):
        """Symmetric (low-skew) data should not get a _log column."""
        rng = np.random.RandomState(42)
        values = rng.normal(50, 5, 100)  # symmetric
        values = np.abs(values)  # ensure non-negative
        df = pd.DataFrame({"beta": values})
        result = log_transform_skewed(df.copy(), ["beta"])
        assert "beta_log" not in result.columns

    def test_no_log_for_negative_data(self):
        """Columns with negative values should not be log-transformed."""
        values = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 100]
        df = pd.DataFrame({"beta": values})
        result = log_transform_skewed(df.copy(), ["beta"])
        assert "beta_log" not in result.columns

    def test_non_count_type_not_transformed(self):
        """Binary/ordinal/bounded_pct types should not be transformed."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "carbon_reduction_target": rng.choice([0, 1], 50),
            "board_size": rng.choice([7, 8, 9, 10, 11], 50),
            "renewable_energy_pct": rng.uniform(0, 100, 50),
        })
        result = log_transform_skewed(
            df.copy(),
            ["carbon_reduction_target", "board_size", "renewable_energy_pct"],
        )
        assert "carbon_reduction_target_log" not in result.columns
        assert "board_size_log" not in result.columns
        assert "renewable_energy_pct_log" not in result.columns

    def test_custom_skew_threshold(self):
        """Higher skew threshold means fewer columns get transformed."""
        rng = np.random.RandomState(42)
        values = rng.exponential(1e6, 100)
        df = pd.DataFrame({"market_cap": values})
        result_low = log_transform_skewed(df.copy(), ["market_cap"], skew_threshold=0.5)
        result_high = log_transform_skewed(df.copy(), ["market_cap"], skew_threshold=100.0)
        assert "market_cap_log" in result_low.columns
        assert "market_cap_log" not in result_high.columns

    def test_original_column_preserved(self):
        """Original column should remain after log transform."""
        rng = np.random.RandomState(42)
        values = rng.exponential(1e6, 100)
        df = pd.DataFrame({"market_cap": values})
        result = log_transform_skewed(df.copy(), ["market_cap"])
        assert "market_cap" in result.columns
