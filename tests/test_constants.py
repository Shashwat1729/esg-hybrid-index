"""Tests for src/constants.py."""

from __future__ import annotations

import pytest

from src.constants import (
    ALL_NUMERIC,
    BINARY_VARS,
    DEFAULT_WEIGHTS,
    DEFAULT_WEIGHTS_WITH_MARKET,
    ESG_COLS,
    ESG_ENV_COLS,
    ESG_GOV_COLS,
    ESG_SOC_COLS,
    FINANCIAL_COLS,
    ID_COLS,
    MARKET_COLS,
    OPERATIONAL_COLS,
    ORDINAL_VARS,
    SCORE_COLUMNS,
    load_profiles_from_config,
)


class TestESGColumnLists:
    """Verify ESG column definitions."""

    def test_env_cols_non_empty(self):
        assert len(ESG_ENV_COLS) > 0

    def test_soc_cols_non_empty(self):
        assert len(ESG_SOC_COLS) > 0

    def test_gov_cols_non_empty(self):
        assert len(ESG_GOV_COLS) > 0

    def test_esg_cols_is_union(self):
        """ESG_COLS == ENV + SOC + GOV."""
        assert ESG_COLS == ESG_ENV_COLS + ESG_SOC_COLS + ESG_GOV_COLS

    def test_no_duplicate_esg_cols(self):
        """No duplicates across the three ESG pillar lists."""
        combined = ESG_ENV_COLS + ESG_SOC_COLS + ESG_GOV_COLS
        assert len(combined) == len(set(combined))


class TestFinancialMarketCols:
    """Verify financial / market / operational column definitions."""

    def test_financial_cols_non_empty(self):
        assert len(FINANCIAL_COLS) > 0

    def test_market_cols_non_empty(self):
        assert len(MARKET_COLS) > 0

    def test_operational_cols_non_empty(self):
        assert len(OPERATIONAL_COLS) > 0

    def test_all_numeric_contains_esg_and_financial(self):
        """ALL_NUMERIC should contain elements from ESG, financial, market, operational."""
        for col in ESG_COLS[:3]:
            assert col in ALL_NUMERIC
        for col in FINANCIAL_COLS[:3]:
            assert col in ALL_NUMERIC


class TestDefaultWeights:
    """Verify DEFAULT_WEIGHTS configuration."""

    def test_weights_sum_to_one(self):
        """DEFAULT_WEIGHTS values sum to approximately 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_all_weights_non_negative(self):
        """All weights are non-negative (zero allowed for disabled factors)."""
        for k, v in DEFAULT_WEIGHTS.items():
            assert v >= 0, f"Weight for '{k}' is negative: {v}"

    def test_expected_keys(self):
        """DEFAULT_WEIGHTS has the expected 9 ex-market keys."""
        expected_keys = {
            "ESG_composite", "financial_score", "operational_score",
            "risk_adjusted_score", "growth_score", "value_score", "stability_score",
            "similarity_rank", "sector_position",
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected_keys

    def test_with_market_has_10_keys(self):
        """DEFAULT_WEIGHTS_WITH_MARKET has the original 10 keys including market_score."""
        assert len(DEFAULT_WEIGHTS_WITH_MARKET) == 10
        assert "market_score" in DEFAULT_WEIGHTS_WITH_MARKET

    def test_keys_match_score_columns(self):
        """DEFAULT_WEIGHTS keys are a subset of SCORE_COLUMNS (market_score excluded)."""
        assert set(DEFAULT_WEIGHTS.keys()) < set(SCORE_COLUMNS)
        assert set(SCORE_COLUMNS) - set(DEFAULT_WEIGHTS.keys()) == {"market_score"}


class TestScoreColumns:
    """Verify SCORE_COLUMNS list."""

    def test_score_columns_count(self):
        """SCORE_COLUMNS has exactly 10 entries."""
        assert len(SCORE_COLUMNS) == 10

    def test_esg_composite_present(self):
        assert "ESG_composite" in SCORE_COLUMNS

    def test_financial_score_present(self):
        assert "financial_score" in SCORE_COLUMNS


class TestMetadataSets:
    """Verify variable classification metadata."""

    def test_binary_vars_are_set(self):
        assert isinstance(BINARY_VARS, set)
        assert len(BINARY_VARS) > 0

    def test_ordinal_vars_are_set(self):
        assert isinstance(ORDINAL_VARS, set)
        assert len(ORDINAL_VARS) > 0

    def test_id_cols_contains_ticker(self):
        assert "ticker" in ID_COLS


class TestLoadProfilesFromConfig:
    """Verify config-based profile loading."""

    def test_returns_dict_of_profiles(self):
        """load_profiles_from_config returns at least the balanced profile."""
        profiles = load_profiles_from_config()
        assert isinstance(profiles, dict)
        assert "balanced" in profiles

    def test_balanced_matches_default_weights(self):
        """Config balanced profile matches DEFAULT_WEIGHTS_WITH_MARKET (10 factors)."""
        profiles = load_profiles_from_config()
        balanced = profiles["balanced"]
        assert set(balanced.keys()) == set(DEFAULT_WEIGHTS_WITH_MARKET.keys())
        for k in DEFAULT_WEIGHTS_WITH_MARKET:
            assert balanced[k] == pytest.approx(DEFAULT_WEIGHTS_WITH_MARKET[k], abs=0.001), (
                f"Mismatch for {k}: config={balanced[k]}, constant={DEFAULT_WEIGHTS_WITH_MARKET[k]}"
            )

    def test_profiles_use_column_names(self):
        """All profile keys use DataFrame column names (ESG_composite, not esg_score)."""
        profiles = load_profiles_from_config()
        for pname, weights in profiles.items():
            assert "ESG_composite" in weights, f"Profile '{pname}' missing ESG_composite"
            assert "esg_score" not in weights, f"Profile '{pname}' has raw config key esg_score"

    def test_all_profiles_sum_to_one(self):
        """Each profile's weights sum to ~1.0."""
        profiles = load_profiles_from_config()
        for pname, weights in profiles.items():
            assert sum(weights.values()) == pytest.approx(1.0, abs=0.01), (
                f"Profile '{pname}' sums to {sum(weights.values())}"
            )
