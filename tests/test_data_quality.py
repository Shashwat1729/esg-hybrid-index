"""Tests for src/data_collection/data_quality.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data_collection.data_quality import (
    QualityReport,
    enforce_min_coverage,
    group_median_impute,
    missingness_report,
)


# ── missingness_report ────────────────────────────────────────────────────

class TestMissingnessReport:
    """Tests for missingness_report()."""

    def test_basic_report_structure(self, sample_esg_df):
        """Report returns correct type and metadata."""
        report = missingness_report(sample_esg_df)
        assert isinstance(report, QualityReport)
        assert report.rows == len(sample_esg_df)
        assert report.cols == len(sample_esg_df.columns)

    def test_no_missing_data(self, sample_esg_df):
        """When there are no NaNs, overall missing should be 0."""
        report = missingness_report(sample_esg_df)
        assert report.overall_missing == pytest.approx(0.0)
        for v in report.missing_by_col.values():
            assert v == pytest.approx(0.0)

    def test_with_nan_values(self, df_with_nans):
        """Correctly reports per-column and overall missingness."""
        report = missingness_report(df_with_nans)
        assert report.rows == 6
        # column 'x' has 2 NaN out of 6
        assert report.missing_by_col["x"] == pytest.approx(2 / 6)
        # column 'y' has 2 NaN out of 6
        assert report.missing_by_col["y"] == pytest.approx(2 / 6)
        # column 'z' has 0 NaN
        assert report.missing_by_col["z"] == pytest.approx(0.0)
        # overall_missing should be > 0
        assert report.overall_missing > 0

    def test_empty_dataframe(self):
        """Empty DataFrame returns zero-valued report with NaN overall."""
        report = missingness_report(pd.DataFrame())
        assert report.rows == 0
        assert report.cols == 0
        assert report.missing_by_col == {}
        assert math.isnan(report.overall_missing)

    def test_outlier_detection(self):
        """Extreme values are flagged by z > 4 rule."""
        # Need enough normal values so that mean/std are well-defined and 1000 is truly z>4
        vals = [1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 2.5, 1.0, 2.0,
                1.5, 2.5, 1.0, 2.0, 1.5, 2.5, 1.0, 2.0, 1.5, 1000.0]
        df = pd.DataFrame({"val": vals})
        report = missingness_report(df)
        assert report.outlier_counts_by_col["val"] >= 1

    def test_constant_column_no_outliers(self):
        """Constant column (std=0) should report 0 outliers."""
        df = pd.DataFrame({"const": [5.0] * 10})
        report = missingness_report(df)
        assert report.outlier_counts_by_col["const"] == 0

    def test_all_nan_column(self):
        """All-NaN column should have 1.0 missingness and 0 outliers."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        report = missingness_report(df)
        assert report.missing_by_col["a"] == pytest.approx(1.0)


# ── enforce_min_coverage ──────────────────────────────────────────────────

class TestEnforceMinCoverage:
    """Tests for enforce_min_coverage()."""

    def test_drops_low_coverage_rows(self):
        """Rows with too many NaN required cols are removed."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "x": [1.0, np.nan, 3.0],
                "y": [np.nan, np.nan, 3.0],
                "z": [1.0, np.nan, 3.0],
            }
        )
        # min_coverage=0.7 => need >=70% of x,y,z present
        # A: 2/3 = 0.67 -> removed; B: 0/3 = 0 -> removed; C: 3/3 -> kept
        out = enforce_min_coverage(
            df, required_cols=["x", "y", "z"], min_coverage=0.70
        )
        assert list(out["ticker"]) == ["C"]

    def test_keeps_all_when_full(self, sample_esg_df):
        """When no data is missing, all rows are kept."""
        out = enforce_min_coverage(
            sample_esg_df,
            required_cols=["scope1_emissions", "scope2_emissions"],
            min_coverage=0.5,
        )
        assert len(out) == len(sample_esg_df)

    def test_empty_required_cols_returns_unchanged(self, sample_esg_df):
        """Empty required_cols list returns df unchanged."""
        out = enforce_min_coverage(
            sample_esg_df, required_cols=[], min_coverage=0.9
        )
        assert len(out) == len(sample_esg_df)

    def test_missing_required_cols_returns_unchanged(self, sample_esg_df):
        """If none of the required cols exist, returns df unchanged."""
        out = enforce_min_coverage(
            sample_esg_df, required_cols=["nonexistent_col"], min_coverage=0.5
        )
        assert len(out) == len(sample_esg_df)

    def test_ticker_column_cast_to_str(self):
        """id_col is cast to str after filtering."""
        df = pd.DataFrame(
            {
                "ticker": [1, 2, 3],
                "x": [1.0, 2.0, 3.0],
            }
        )
        out = enforce_min_coverage(
            df, required_cols=["x"], min_coverage=0.0
        )
        # pandas 3.0 uses StringDtype; older uses object — both represent strings
        assert pd.api.types.is_string_dtype(out["ticker"])


# ── group_median_impute ───────────────────────────────────────────────────

class TestGroupMedianImpute:
    """Tests for group_median_impute()."""

    def test_fills_nan_with_group_median(self, df_with_nans):
        """NaN values are filled using within-group medians."""
        out = group_median_impute(
            df_with_nans, group_cols=["sector"], numeric_cols=["x", "y"]
        )
        # No NaN should remain in imputed columns
        assert out["x"].isna().sum() == 0
        assert out["y"].isna().sum() == 0

    def test_no_group_fills_with_global_median(self):
        """Empty group_cols falls back to global median."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 5.0]})
        out = group_median_impute(df, group_cols=[], numeric_cols=["x"])
        assert out["x"].isna().sum() == 0
        # Global median of [1, 3, 5] = 3.0
        assert out["x"].iloc[1] == pytest.approx(3.0)

    def test_preserves_non_nan_values(self, df_with_nans):
        """Non-NaN values are not changed."""
        out = group_median_impute(
            df_with_nans, group_cols=["sector"], numeric_cols=["x"]
        )
        # Original non-NaN values should stay
        assert out.loc[0, "x"] == pytest.approx(1.0)
        assert out.loc[2, "x"] == pytest.approx(3.0)
        assert out.loc[4, "x"] == pytest.approx(5.0)
        assert out.loc[5, "x"] == pytest.approx(6.0)

    def test_auto_detects_numeric_cols(self, df_with_nans):
        """When numeric_cols is None, all numeric columns are imputed."""
        out = group_median_impute(df_with_nans, group_cols=["sector"])
        assert out["x"].isna().sum() == 0
        assert out["y"].isna().sum() == 0

    def test_skips_missing_column(self, df_with_nans):
        """Columns not in DataFrame are silently skipped."""
        out = group_median_impute(
            df_with_nans,
            group_cols=["sector"],
            numeric_cols=["x", "nonexistent"],
        )
        assert out["x"].isna().sum() == 0

    def test_does_not_modify_input(self, df_with_nans):
        """Original DataFrame is not modified."""
        original_nans = df_with_nans["x"].isna().sum()
        _ = group_median_impute(
            df_with_nans, group_cols=["sector"], numeric_cols=["x"]
        )
        assert df_with_nans["x"].isna().sum() == original_nans
