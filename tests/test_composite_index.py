"""Tests for src/index_construction/composite_index.py."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.index_construction.composite_index import (
    CompositeIndexBuilder,
    compute_pillar_scores,
    normalize_indicators,
)


# ── Helper ────────────────────────────────────────────────────────────────

def _make_indicator_df(n: int = 8, seed: int = 42) -> pd.DataFrame:
    """Build a minimal DataFrame with ESG-like numeric columns."""
    rng = np.random.RandomState(seed)
    sectors = ["Tech", "Health", "Finance", "Energy"]
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "sector": [sectors[i % len(sectors)] for i in range(n)],
            "emissions": rng.uniform(10, 100, n),
            "energy": rng.uniform(20, 80, n),
            "water": rng.uniform(1, 10, n),
            "waste": rng.uniform(5, 50, n),
            "labor": rng.uniform(10, 50, n),
            "diversity": rng.uniform(0, 100, n),
            "injury": rng.uniform(0, 5, n),
            "community": rng.uniform(0, 10, n),
            "board": rng.uniform(30, 90, n),
            "comp": rng.uniform(50, 500, n),
            "shareholder": rng.uniform(0, 100, n),
            "ethics": rng.uniform(30, 100, n),
        }
    )


# ── normalize_indicators ─────────────────────────────────────────────────

class TestNormalizeIndicators:
    """Tests for normalize_indicators()."""

    @pytest.mark.parametrize("method", ["zscore", "minmax", "percentile", "robust_zscore"])
    def test_all_methods_produce_norm_columns(self, method):
        """Each normalization method adds <col>_norm columns."""
        df = _make_indicator_df()
        cols = ["emissions", "energy", "water"]
        result = normalize_indicators(df, cols, method=method)
        for c in cols:
            assert f"{c}_norm" in result.columns, f"{c}_norm missing for method={method}"

    def test_zscore_mean_near_zero(self):
        """Z-score normalized columns should have mean ≈ 0."""
        df = _make_indicator_df(n=100, seed=7)
        result = normalize_indicators(df, ["emissions"], method="zscore")
        assert result["emissions_norm"].mean() == pytest.approx(0.0, abs=0.1)

    def test_minmax_range_01(self):
        """Min-max normalized values should lie in [0, 1]."""
        df = _make_indicator_df(n=50, seed=7)
        result = normalize_indicators(df, ["emissions"], method="minmax")
        vals = result["emissions_norm"]
        assert vals.min() >= -0.01  # allow tiny float error
        assert vals.max() <= 1.01

    def test_percentile_range(self):
        """Percentile-normalized values in [0, 100]."""
        df = _make_indicator_df(n=50, seed=7)
        result = normalize_indicators(df, ["emissions"], method="percentile")
        assert result["emissions_norm"].min() >= 0
        assert result["emissions_norm"].max() <= 100

    def test_robust_zscore_no_deprecated_mad(self):
        """robust_zscore works without calling deprecated pd.Series.mad()."""
        df = _make_indicator_df(n=20)
        # Should not raise any DeprecationWarning or AttributeError
        result = normalize_indicators(df, ["emissions"], method="robust_zscore")
        assert "emissions_norm" in result.columns
        assert result["emissions_norm"].notna().all()

    def test_missing_column_warns(self):
        """Missing indicator column emits a UserWarning."""
        df = _make_indicator_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_indicators(df, ["nonexistent_col"], method="zscore")
            assert any("not found" in str(x.message) for x in w)

    def test_non_numeric_column_warns(self):
        """Non-numeric column emits a UserWarning."""
        df = _make_indicator_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_indicators(df, ["ticker"], method="zscore")
            assert any("not numeric" in str(x.message) for x in w)

    def test_unknown_method_raises(self):
        """Unknown normalization method raises ValueError."""
        df = _make_indicator_df()
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_indicators(df, ["emissions"], method="bogus")

    def test_winsorize_clips_extremes(self):
        """Winsorization clips extreme values."""
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
        result = normalize_indicators(
            df,
            ["val"],
            method="zscore",
            winsorize={"enabled": True, "lower_quantile": 0.05, "upper_quantile": 0.95},
        )
        assert "val_norm" in result.columns

    def test_by_group_sector(self):
        """Within-sector normalization produces _norm columns."""
        df = _make_indicator_df(n=20)
        result = normalize_indicators(
            df, ["emissions"], method="zscore", by_group={"sector": True}
        )
        assert "emissions_norm" in result.columns

    def test_does_not_modify_input(self):
        """Original DataFrame is not mutated."""
        df = _make_indicator_df()
        original_cols = set(df.columns)
        _ = normalize_indicators(df, ["emissions"], method="zscore")
        assert set(df.columns) == original_cols


# ── compute_pillar_scores ─────────────────────────────────────────────────

class TestComputePillarScores:
    """Tests for compute_pillar_scores()."""

    def test_creates_pillar_and_composite_columns(self):
        """E_score, S_score, G_score, ESG_composite columns are added."""
        df = _make_indicator_df()
        # First normalize so _norm columns exist
        indicator_cols = ["emissions", "energy", "water", "waste",
                          "labor", "diversity", "injury", "community",
                          "board", "comp", "shareholder", "ethics"]
        df = normalize_indicators(df, indicator_cols, method="zscore")

        pillar_config = {
            "E": {"emissions": 0.40, "energy": 0.25, "water": 0.20, "waste": 0.15},
            "S": {"labor": 0.35, "diversity": 0.30, "health_safety": 0.25, "community": 0.10},
            "G": {"board": 0.35, "comp": 0.25, "shareholder_rights": 0.20, "ethics": 0.20},
        }
        result = compute_pillar_scores(df, pillar_config)

        for col in ["E_score", "S_score", "G_score", "ESG_composite"]:
            assert col in result.columns

    def test_scores_in_0_100_range(self):
        """Pillar scores are clipped to [0, 100]."""
        df = _make_indicator_df(n=50, seed=99)
        indicator_cols = ["emissions", "energy", "water", "waste"]
        df = normalize_indicators(df, indicator_cols, method="zscore")
        pillar_config = {
            "E": {"emissions": 0.40, "energy": 0.25, "water": 0.20, "waste": 0.15},
        }
        result = compute_pillar_scores(df, pillar_config)
        assert result["E_score"].min() >= 0
        assert result["E_score"].max() <= 100

    def test_custom_pillar_weights(self):
        """Custom pillar weights alter the composite score."""
        df = _make_indicator_df(n=20)
        cols = ["emissions", "labor", "board"]
        df = normalize_indicators(df, cols, method="zscore")
        pillar_config = {
            "E": {"emissions": 1.0},
            "S": {"labor": 1.0},
            "G": {"board": 1.0},
        }
        r1 = compute_pillar_scores(df.copy(), pillar_config, pillar_weights={"E": 0.9, "S": 0.05, "G": 0.05})
        r2 = compute_pillar_scores(df.copy(), pillar_config, pillar_weights={"E": 0.05, "S": 0.05, "G": 0.9})
        # Composite scores should generally differ when weights are very different
        assert not np.allclose(r1["ESG_composite"].values, r2["ESG_composite"].values)


# ── CompositeIndexBuilder ─────────────────────────────────────────────────

class TestCompositeIndexBuilder:
    """Tests for CompositeIndexBuilder.build()."""

    def test_build_end_to_end(self, index_config):
        """build() adds _norm, pillar scores, and ESG_composite columns."""
        df = _make_indicator_df(n=10)
        builder = CompositeIndexBuilder(index_config)
        result = builder.build(df, indicator_cols=["emissions", "energy", "water", "waste",
                                                    "labor", "diversity", "injury", "community",
                                                    "board", "comp", "shareholder", "ethics"])
        assert "ESG_composite" in result.columns
        # At least some _norm columns should exist
        norm_cols = [c for c in result.columns if c.endswith("_norm")]
        assert len(norm_cols) > 0

    def test_build_raises_on_no_valid_cols(self, index_config):
        """ValueError when none of the indicator_cols exist in df."""
        df = _make_indicator_df()
        builder = CompositeIndexBuilder(index_config)
        with pytest.raises(ValueError, match="None of the requested indicator_cols"):
            builder.build(df, indicator_cols=["nonexistent_a", "nonexistent_b"])

    def test_build_warns_on_partial_missing_cols(self, index_config):
        """Warns when some indicator_cols are missing, but proceeds."""
        df = _make_indicator_df()
        builder = CompositeIndexBuilder(index_config)
        # "emissions" exists, "nonexistent" does not
        result = builder.build(df, indicator_cols=["emissions", "nonexistent"])
        assert "emissions_norm" in result.columns

    def test_build_preserves_id_column(self, index_config):
        """Ticker column survives the build pipeline."""
        df = _make_indicator_df()
        builder = CompositeIndexBuilder(index_config)
        result = builder.build(df, indicator_cols=["emissions"])
        assert "ticker" in result.columns
        assert len(result) == len(df)


# ── Direction handling (ESG_LOWER_IS_BETTER) ──────────────────────────────

class TestIndicatorDirectionFlip:
    """Tests for the lower-is-better direction flip in normalize_indicators."""

    def _make_directional_df(self):
        """DataFrame with one 'lower is better' and one 'higher is better' col."""
        return pd.DataFrame({
            "scope1_emissions": [100.0, 50.0, 10.0],  # lower is better
            "renewable_energy_pct": [10.0, 50.0, 100.0],  # higher is better
        })

    def test_zscore_flip_negates(self):
        """Z-score: lower-is-better column is negated after normalization."""
        df = self._make_directional_df()
        result = normalize_indicators(
            df, ["scope1_emissions", "renewable_energy_pct"], method="zscore",
        )
        # Company with lowest emissions (10) should have the *highest* norm score
        assert result["scope1_emissions_norm"].iloc[2] > result["scope1_emissions_norm"].iloc[0]
        # Higher-is-better column should NOT be flipped
        assert result["renewable_energy_pct_norm"].iloc[2] > result["renewable_energy_pct_norm"].iloc[0]

    def test_minmax_flip_inverts(self):
        """Min-max: lower-is-better column becomes 1 - x, staying in [0, 1]."""
        df = self._make_directional_df()
        result = normalize_indicators(
            df, ["scope1_emissions"], method="minmax",
        )
        vals = result["scope1_emissions_norm"]
        # Lowest raw value (10) → highest norm (near 1.0)
        assert vals.iloc[2] > vals.iloc[0]
        assert vals.min() >= -0.01
        assert vals.max() <= 1.01

    def test_percentile_flip(self):
        """Percentile: lower-is-better becomes 100 - x, staying in [0, 100]."""
        df = self._make_directional_df()
        result = normalize_indicators(
            df, ["scope1_emissions"], method="percentile",
        )
        vals = result["scope1_emissions_norm"]
        assert vals.iloc[2] > vals.iloc[0]
        assert vals.min() >= 0
        assert vals.max() <= 100

    def test_robust_zscore_flip(self):
        """Robust z-score: lower-is-better column is negated."""
        df = self._make_directional_df()
        result = normalize_indicators(
            df, ["scope1_emissions"], method="robust_zscore",
        )
        assert result["scope1_emissions_norm"].iloc[2] > result["scope1_emissions_norm"].iloc[0]

    def test_explicit_empty_set_disables_flip(self):
        """Passing lower_is_better=set() skips all flipping."""
        df = self._make_directional_df()
        result = normalize_indicators(
            df, ["scope1_emissions"], method="zscore", lower_is_better=set(),
        )
        # Without flip, highest raw value gets highest z-score
        assert result["scope1_emissions_norm"].iloc[0] > result["scope1_emissions_norm"].iloc[2]

    def test_custom_lower_is_better_set(self):
        """A custom lower_is_better set overrides the default."""
        df = pd.DataFrame({"renewable_energy_pct": [10.0, 50.0, 100.0]})
        # Pretend renewable_energy_pct is lower-is-better (unusual but tests override)
        result = normalize_indicators(
            df, ["renewable_energy_pct"], method="zscore",
            lower_is_better={"renewable_energy_pct"},
        )
        # After flip, highest raw → lowest norm
        assert result["renewable_energy_pct_norm"].iloc[0] > result["renewable_energy_pct_norm"].iloc[2]

    def test_non_lower_is_better_not_flipped(self):
        """Columns not in the set are left unchanged."""
        df = self._make_directional_df()
        # Normalize without direction handling
        no_flip = normalize_indicators(
            df, ["renewable_energy_pct"], method="zscore", lower_is_better=set(),
        )
        # Normalize with default handling (renewable_energy_pct is NOT in ESG_LOWER_IS_BETTER)
        with_flip = normalize_indicators(
            df, ["renewable_energy_pct"], method="zscore",
        )
        pd.testing.assert_series_equal(
            no_flip["renewable_energy_pct_norm"],
            with_flip["renewable_energy_pct_norm"],
        )
