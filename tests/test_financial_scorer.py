"""Tests for src/financial_scoring/financial_scorer.py."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.financial_scoring.financial_scorer import FinancialScorer, MarketFactorScorer


# ── FinancialScorer ───────────────────────────────────────────────────────

class TestFinancialScorer:
    """Tests for FinancialScorer."""

    def test_computes_financial_score(self, sample_financial_df, index_config):
        """financial_score column is added and in [0, 100]."""
        scorer = FinancialScorer(index_config)
        result = scorer.compute_financial_score(sample_financial_df)
        assert "financial_score" in result.columns
        assert result["financial_score"].min() >= 0
        assert result["financial_score"].max() <= 100

    def test_inverse_indicators_are_inverted(self, sample_financial_df, index_config):
        """debt_to_equity, trailing_pe, price_to_book are inverted before normalization."""
        scorer = FinancialScorer(index_config)
        # Store originals
        orig_dte = sample_financial_df["debt_to_equity"].copy()
        result = scorer.compute_financial_score(sample_financial_df)
        # After scoring, original values should be restored
        pd.testing.assert_series_equal(
            result["debt_to_equity"], orig_dte, check_names=False
        )

    def test_category_scores_created(self, sample_financial_df, index_config):
        """Category score columns (e.g. profitability_score) are added."""
        scorer = FinancialScorer(index_config)
        result = scorer.compute_financial_score(sample_financial_df)
        # At least one category score should exist
        cat_score_cols = [c for c in result.columns if c.endswith("_score") and c != "financial_score"]
        assert len(cat_score_cols) > 0

    def test_no_indicators_present(self, index_config):
        """When no financial indicators exist, returns default score of 50."""
        df = pd.DataFrame({"ticker": ["A", "B"], "irrelevant": [1, 2]})
        scorer = FinancialScorer(index_config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = scorer.compute_financial_score(df)
            assert any("No financial indicators" in str(x.message) for x in w)
        assert result["financial_score"].iloc[0] == pytest.approx(50.0)

    def test_partial_indicators(self, index_config):
        """Scorer handles partial indicator availability gracefully."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "roa": [0.1, 0.15, 0.2, 0.05, 0.12],
                "roe": [0.2, 0.3, 0.1, 0.15, 0.25],
                # Missing: net_margin, debt_to_equity, etc.
            }
        )
        scorer = FinancialScorer(index_config)
        result = scorer.compute_financial_score(df)
        assert "financial_score" in result.columns
        assert result["financial_score"].notna().all()

    def test_preserves_row_count(self, sample_financial_df, index_config):
        """Output has same number of rows as input."""
        scorer = FinancialScorer(index_config)
        result = scorer.compute_financial_score(sample_financial_df)
        assert len(result) == len(sample_financial_df)


# ── MarketFactorScorer ────────────────────────────────────────────────────

class TestMarketFactorScorer:
    """Tests for MarketFactorScorer."""

    def test_computes_market_score(self, sample_market_df, index_config):
        """market_score column is added and in [0, 100]."""
        scorer = MarketFactorScorer(index_config)
        result = scorer.compute_market_score(sample_market_df)
        assert "market_score" in result.columns
        assert result["market_score"].min() >= 0
        assert result["market_score"].max() <= 100

    def test_inverse_volatility_indicators(self, sample_market_df, index_config):
        """Volatility indicators are inverted (lower is better) then restored."""
        orig_vol = sample_market_df["price_volatility"].copy()
        scorer = MarketFactorScorer(index_config)
        result = scorer.compute_market_score(sample_market_df)
        pd.testing.assert_series_equal(
            result["price_volatility"], orig_vol, check_names=False
        )

    def test_no_market_indicators(self, index_config):
        """When no market indicators exist, returns default score of 50."""
        df = pd.DataFrame({"ticker": ["A", "B"], "x": [1, 2]})
        scorer = MarketFactorScorer(index_config)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = scorer.compute_market_score(df)
        assert result["market_score"].iloc[0] == pytest.approx(50.0)

    def test_preserves_row_count(self, sample_market_df, index_config):
        """Output has same number of rows as input."""
        scorer = MarketFactorScorer(index_config)
        result = scorer.compute_market_score(sample_market_df)
        assert len(result) == len(sample_market_df)
