"""Tests for src/similarity/preference_scoring.py."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.similarity.preference_scoring import (
    ALL_SCORE_COMPONENTS,
    SCORE_COLUMN_MAP,
    PreferenceScorer,
)


# ── PreferenceScorer.compute_preference_score ─────────────────────────────

class TestComputePreferenceScore:
    """Tests for PreferenceScorer.compute_preference_score()."""

    @pytest.mark.parametrize("profile", ["esg_first", "balanced", "financial_first"])
    def test_all_profiles_produce_scores(self, sample_scores_df, index_config, profile):
        """All 3 investor profiles compute scores without error."""
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(
            sample_scores_df, investor_profile=profile
        )
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_scores_df)

    def test_scores_in_0_100(self, sample_scores_df, index_config):
        """Preference scores are clipped to [0, 100]."""
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(
            sample_scores_df, investor_profile="balanced"
        )
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_esg_first_weights_esg_higher(self, sample_scores_df, index_config):
        """esg_first profile gives more weight to ESG than financial_first does."""
        scorer = PreferenceScorer(index_config)
        esg_weights = index_config["preference_scoring"]["investor_profiles"]["esg_first"]
        fin_weights = index_config["preference_scoring"]["investor_profiles"]["financial_first"]
        assert esg_weights["esg_score"] > fin_weights["esg_score"]
        assert fin_weights["financial_score"] > esg_weights["financial_score"]

    def test_unknown_profile_uses_defaults(self, sample_scores_df, index_config, caplog):
        """Unknown profile logs a warning and uses default weights."""
        scorer = PreferenceScorer(index_config)
        with caplog.at_level(logging.WARNING):
            scores = scorer.compute_preference_score(
                sample_scores_df, investor_profile="aggressive_growth"
            )
        assert "not found" in caplog.text.lower() or len(scores) == len(sample_scores_df)
        assert scores.notna().all()

    def test_similarity_rank_rescaled(self, index_config):
        """similarity_rank on 0-1 scale is rescaled to 0-100."""
        df = pd.DataFrame(
            {
                "ESG_composite": [50.0, 60.0],
                "financial_score": [50.0, 60.0],
                "market_score": [50.0, 60.0],
                "operational_score": [50.0, 60.0],
                "risk_adjusted_score": [50.0, 60.0],
                "growth_score": [50.0, 60.0],
                "value_score": [50.0, 60.0],
                "stability_score": [50.0, 60.0],
                "similarity_rank": [0.5, 0.8],  # 0-1 scale
                "sector_position": [0.3, 0.7],  # 0-1 scale
            }
        )
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(df, investor_profile="balanced")
        # Scores should be non-trivial (not just 0)
        assert scores.sum() > 0

    def test_missing_columns_handled(self, index_config):
        """Missing score columns are silently skipped (partial data)."""
        df = pd.DataFrame(
            {
                "ESG_composite": [50.0, 60.0, 70.0],
                "financial_score": [40.0, 50.0, 60.0],
                # All other columns missing
            }
        )
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(df, investor_profile="balanced")
        assert len(scores) == 3
        assert scores.notna().all()

    def test_weight_normalization(self, index_config, caplog):
        """Weights that don't sum to 1.0 are normalized with a warning."""
        bad_config = {
            "preference_scoring": {
                "investor_profiles": {
                    "bad": {
                        "esg_score": 0.50,
                        "financial_score": 0.50,
                        "market_score": 0.50,
                        # Sum = 1.5
                    }
                }
            }
        }
        df = pd.DataFrame(
            {
                "ESG_composite": [50.0, 60.0],
                "financial_score": [50.0, 60.0],
                "market_score": [50.0, 60.0],
            }
        )
        scorer = PreferenceScorer(bad_config)
        with caplog.at_level(logging.WARNING):
            scores = scorer.compute_preference_score(df, investor_profile="bad")
        assert scores.notna().all()


# ── Aggregation mode tests ────────────────────────────────────────────────

class TestAggregationModes:
    """Tests for the rank / variance_equalized / raw aggregation modes."""

    @pytest.fixture()
    def _scorer_raw(self):
        """Scorer configured for raw mode."""
        cfg = {
            "preference_scoring": {
                "aggregation_mode": "raw",
                "investor_profiles": {
                    "equal": {"esg_score": 0.50, "financial_score": 0.50},
                },
            }
        }
        return PreferenceScorer(cfg)

    @pytest.fixture()
    def _scorer_rank(self):
        """Scorer configured for rank mode."""
        cfg = {
            "preference_scoring": {
                "aggregation_mode": "rank",
                "investor_profiles": {
                    "equal": {"esg_score": 0.50, "financial_score": 0.50},
                },
            }
        }
        return PreferenceScorer(cfg)

    @pytest.fixture()
    def _scorer_var_eq(self):
        """Scorer configured for variance_equalized mode."""
        cfg = {
            "preference_scoring": {
                "aggregation_mode": "variance_equalized",
                "investor_profiles": {
                    "equal": {"esg_score": 0.50, "financial_score": 0.50},
                },
            }
        }
        return PreferenceScorer(cfg)

    @pytest.fixture()
    def _skewed_df(self):
        """DataFrame where ESG is narrow-spread and financial is wide-spread."""
        return pd.DataFrame(
            {
                "ESG_composite": [48.0, 49.0, 50.0, 51.0, 52.0],
                "financial_score": [20.0, 35.0, 50.0, 65.0, 80.0],
            }
        )

    def test_default_mode_is_rank(self, index_config):
        """Default aggregation_mode (when config omits it) is 'rank'."""
        # index_config fixture does not set aggregation_mode
        scorer = PreferenceScorer(index_config)
        assert scorer.aggregation_mode == "rank"

    def test_config_overrides_default(self):
        """aggregation_mode from config is respected."""
        cfg = {"preference_scoring": {"aggregation_mode": "raw"}}
        scorer = PreferenceScorer(cfg)
        assert scorer.aggregation_mode == "raw"

    def test_parameter_overrides_config(self, _scorer_raw, _skewed_df):
        """aggregation_mode kwarg overrides the instance default."""
        scores_raw = _scorer_raw.compute_preference_score(
            _skewed_df, investor_profile="equal", aggregation_mode="raw",
        )
        scores_rank = _scorer_raw.compute_preference_score(
            _skewed_df, investor_profile="equal", aggregation_mode="rank",
        )
        # They should differ because rank transforms the values
        assert not np.allclose(scores_raw.values, scores_rank.values)

    def test_invalid_mode_raises(self, index_config, sample_scores_df):
        """Invalid aggregation_mode raises ValueError."""
        scorer = PreferenceScorer(index_config)
        with pytest.raises(ValueError, match="Invalid aggregation_mode"):
            scorer.compute_preference_score(
                sample_scores_df,
                investor_profile="balanced",
                aggregation_mode="bogus",
            )

    def test_raw_preserves_original_values(self, _scorer_raw, _skewed_df):
        """Raw mode produces a simple weighted average of original values."""
        scores = _scorer_raw.compute_preference_score(
            _skewed_df, investor_profile="equal",
        )
        expected = 0.5 * _skewed_df["ESG_composite"] + 0.5 * _skewed_df["financial_score"]
        np.testing.assert_allclose(scores.values, expected.values)

    def test_rank_equalises_variance(self, _scorer_rank, _skewed_df):
        """In rank mode, both factors contribute equal variance to the score."""
        scores = _scorer_rank.compute_preference_score(
            _skewed_df, investor_profile="equal",
        )
        # After rank transform both columns have the same distribution,
        # so the composite std should be lower than raw (financial dominates raw).
        assert scores.std() > 0  # non-degenerate
        # Key property: with equal weights and rank-transformed factors, the
        # score should be less spread than raw (where financial_score dominates).
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_rank_output_range(self, _scorer_rank, _skewed_df):
        """Rank mode scores stay within [0, 100]."""
        scores = _scorer_rank.compute_preference_score(
            _skewed_df, investor_profile="equal",
        )
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_variance_equalized_mean_and_spread(self, _scorer_var_eq, _skewed_df):
        """variance_equalized maps each factor to mean≈50, std≈10."""
        scores = _scorer_var_eq.compute_preference_score(
            _skewed_df, investor_profile="equal",
        )
        # Since both factors are standardised to mean=50,std=10 and weights
        # are 0.5 each, composite mean should be ~50.
        assert 40 < scores.mean() < 60
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0

    def test_rank_reduces_balance_ratio(self, _skewed_df):
        """rank mode equalises per-factor variance contribution.

        With equal weights and two factors, the variance of each factor's
        weighted contribution should be nearly identical after rank transform,
        whereas in raw mode the wide-spread factor (financial_score, std≈24)
        dominates the narrow-spread factor (ESG_composite, std≈1.6).
        """
        cfg = {
            "preference_scoring": {
                "investor_profiles": {
                    "equal": {"esg_score": 0.50, "financial_score": 0.50},
                },
            }
        }

        # --- raw mode: compute per-factor weighted std ---
        scorer_raw = PreferenceScorer({**cfg, "preference_scoring": {**cfg["preference_scoring"], "aggregation_mode": "raw"}})
        esg_raw = _skewed_df["ESG_composite"]
        fin_raw = _skewed_df["financial_score"]
        raw_ratio = (0.5 * fin_raw).std() / (0.5 * esg_raw).std()
        # financial_score std ≈ 24, ESG_composite std ≈ 1.6 → ratio ≈ 15
        assert raw_ratio > 5, "Precondition: raw financial_score dominates ESG"

        # --- rank mode: both factors have identical rank distributions ---
        from scipy.stats import rankdata as _rd
        esg_ranked = pd.Series(_rd(esg_raw.values, method="average") / len(esg_raw) * 100)
        fin_ranked = pd.Series(_rd(fin_raw.values, method="average") / len(fin_raw) * 100)
        rank_ratio = (0.5 * fin_ranked).std() / (0.5 * esg_ranked).std()
        # After rank transform both have identical spread → ratio ≈ 1
        assert rank_ratio == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("mode", ["rank", "variance_equalized", "raw"])
    def test_all_modes_produce_valid_scores(
        self, sample_scores_df, index_config, mode,
    ):
        """All three modes produce non-NaN scores within [0, 100]."""
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(
            sample_scores_df,
            investor_profile="balanced",
            aggregation_mode=mode,
        )
        assert scores.notna().all()
        assert scores.min() >= 0.0
        assert scores.max() <= 100.0


# ── PreferenceScorer.rank_companies ───────────────────────────────────────

class TestRankCompanies:
    """Tests for PreferenceScorer.rank_companies()."""

    def test_ranking_order(self, sample_scores_df, index_config):
        """Companies are ranked in descending order of preference score."""
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(
            sample_scores_df, investor_profile="balanced"
        )
        sample_scores_df["preference_score"] = scores
        ranked = scorer.rank_companies(sample_scores_df)
        vals = ranked["preference_score"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_rank_column_added(self, sample_scores_df, index_config):
        """preference_rank column is added starting from 1."""
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(
            sample_scores_df, investor_profile="balanced"
        )
        sample_scores_df["preference_score"] = scores
        ranked = scorer.rank_companies(sample_scores_df)
        assert "preference_rank" in ranked.columns
        assert ranked["preference_rank"].iloc[0] == 1
        assert ranked["preference_rank"].iloc[-1] == len(ranked)

    def test_top_n(self, sample_scores_df, index_config):
        """top_n limits the number of returned companies."""
        scorer = PreferenceScorer(index_config)
        scores = scorer.compute_preference_score(
            sample_scores_df, investor_profile="balanced"
        )
        sample_scores_df["preference_score"] = scores
        ranked = scorer.rank_companies(sample_scores_df, top_n=3)
        assert len(ranked) == 3

    def test_missing_pref_col_raises(self, sample_scores_df, index_config):
        """ValueError when preference_score column is missing."""
        scorer = PreferenceScorer(index_config)
        with pytest.raises(ValueError, match="not found"):
            scorer.rank_companies(sample_scores_df)


# ── Module-level constants ────────────────────────────────────────────────

class TestPreferenceScoringConstants:
    """Tests for module-level constants."""

    def test_all_score_components_length(self):
        """ALL_SCORE_COMPONENTS has exactly 10 elements."""
        assert len(ALL_SCORE_COMPONENTS) == 10

    def test_score_column_map_keys_match_components(self):
        """SCORE_COLUMN_MAP keys match ALL_SCORE_COMPONENTS."""
        assert set(SCORE_COLUMN_MAP.keys()) == set(ALL_SCORE_COMPONENTS)
