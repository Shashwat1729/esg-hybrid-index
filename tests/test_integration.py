"""Integration test: raw data -> clean data -> indexed data pipeline.

Uses a small synthetic dataset to verify the full pipeline flow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_collection.data_quality import group_median_impute, missingness_report
from src.data_collection.data_pipeline import standardize_and_clean
from src.index_construction.composite_index import CompositeIndexBuilder, normalize_indicators
from src.financial_scoring.financial_scorer import FinancialScorer
from src.similarity.cosine_similarity import compute_similarity_matrix
from src.similarity.preference_scoring import PreferenceScorer


def _make_full_synthetic_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic dataset with ESG, financial, and market columns."""
    rng = np.random.RandomState(seed)
    sectors = ["Tech", "Health", "Finance", "Energy", "Retail"]
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "company_name": [f"Company_{i}" for i in range(n)],
            "sector": [sectors[i % len(sectors)] for i in range(n)],
            "country": ["US", "IN"] * (n // 2),
            # ESG indicators (with some NaN)
            "scope1_emissions": rng.uniform(10, 100, n),
            "scope2_emissions": _inject_nans(rng.uniform(5, 50, n), rng, frac=0.1),
            "emissions_intensity": rng.uniform(0.1, 1.0, n),
            "renewable_energy_pct": rng.uniform(0, 100, n),
            "energy_efficiency": rng.uniform(50, 100, n),
            "water_usage_intensity": rng.uniform(0.5, 5.0, n),
            "waste_recycling_pct": rng.uniform(10, 90, n),
            "employee_turnover": rng.uniform(5, 30, n),
            "gender_diversity_pct": rng.uniform(20, 60, n),
            "injury_rate": rng.uniform(0, 5, n),
            "board_independence_pct": rng.uniform(40, 90, n),
            "board_diversity_pct": rng.uniform(10, 50, n),
            "exec_comp_esg_linked": rng.uniform(0, 100, n),
            "ethics_compliance_score": rng.uniform(40, 100, n),
            # Financial indicators
            "roa": rng.uniform(-0.05, 0.25, n),
            "roe": rng.uniform(-0.10, 0.40, n),
            "net_margin": rng.uniform(-0.05, 0.30, n),
            "debt_to_equity": rng.uniform(0.1, 3.0, n),
            "trailing_pe": rng.uniform(5, 50, n),
            "price_to_book": rng.uniform(0.5, 10, n),
            "market_cap": rng.uniform(1e9, 1e12, n),
            "free_cashflow": rng.uniform(-1e8, 1e10, n),
        }
    )


def _inject_nans(arr: np.ndarray, rng: np.random.RandomState, frac: float = 0.1) -> np.ndarray:
    """Inject NaN values into a fraction of array elements."""
    arr = arr.copy()
    mask = rng.random(len(arr)) < frac
    arr[mask] = np.nan
    return arr


@pytest.mark.integration
class TestPipelineIntegration:
    """End-to-end integration tests for the data pipeline."""

    @pytest.fixture
    def raw_df(self) -> pd.DataFrame:
        return _make_full_synthetic_df(n=10, seed=42)

    @pytest.fixture
    def pipeline_config(self) -> dict:
        """Minimal config for the full pipeline."""
        return {
            "esg_index": {
                "normalization": {
                    "method": "zscore",
                    "by_group": {"sector": False},
                    "winsorize": {"enabled": False},
                },
                "missing_data": {
                    "min_indicator_coverage": 0.50,
                    "imputation": {"method": "group_median", "group_keys": ["sector"]},
                },
                "pillar_weights": {"E": 0.35, "S": 0.35, "G": 0.30},
                "category_weights": {
                    "E": {"emissions": 0.40, "energy": 0.25, "water": 0.20, "waste": 0.15},
                    "S": {"labor": 0.35, "diversity": 0.30, "health_safety": 0.25, "community": 0.10},
                    "G": {"board": 0.35, "comp": 0.25, "shareholder_rights": 0.20, "ethics": 0.20},
                },
            },
            "financial_scoring": {
                "categories": {
                    "profitability": {
                        "weight_range": {"default": 0.40},
                        "indicators": ["roa", "roe", "net_margin"],
                    },
                    "stability": {
                        "weight_range": {"default": 0.30},
                        "indicators": ["debt_to_equity", "market_cap"],
                    },
                    "valuation": {
                        "weight_range": {"default": 0.30},
                        "indicators": ["trailing_pe", "price_to_book"],
                    },
                },
                "normalization": {"method": "zscore", "by_group": {}},
            },
            "preference_scoring": {
                "investor_profiles": {
                    "balanced": {
                        "esg_score": 0.20,
                        "financial_score": 0.25,
                        "market_score": 0.10,
                        "operational_score": 0.10,
                        "risk_adjusted_score": 0.10,
                        "growth_score": 0.08,
                        "value_score": 0.07,
                        "stability_score": 0.05,
                        "similarity_rank": 0.03,
                        "sector_position": 0.02,
                    },
                },
            },
        }

    def test_step1_data_quality(self, raw_df):
        """Step 1: Raw data quality assessment."""
        report = missingness_report(raw_df)
        assert report.rows == 10
        assert report.cols > 0
        # Overall missing should be small (we injected ~10% in one column)
        assert report.overall_missing < 0.10

    def test_step2_clean_and_impute(self, raw_df, pipeline_config):
        """Step 2: Clean and impute produces a DataFrame with fewer NaNs."""
        esg_cols = [
            "scope1_emissions", "scope2_emissions", "emissions_intensity",
            "renewable_energy_pct", "energy_efficiency",
        ]
        cleaned, diag = standardize_and_clean(
            raw_df, index_cfg=pipeline_config, required_cols=esg_cols
        )
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0
        # NaN count should decrease after imputation
        nans_after = cleaned[esg_cols].isna().sum().sum()
        assert nans_after == 0 or nans_after < raw_df[esg_cols].isna().sum().sum()

    def test_step3_esg_index_construction(self, raw_df, pipeline_config):
        """Step 3: Composite ESG index construction."""
        indicator_cols = [
            "scope1_emissions", "scope2_emissions", "emissions_intensity",
            "renewable_energy_pct", "energy_efficiency",
            "water_usage_intensity", "waste_recycling_pct",
            "employee_turnover", "gender_diversity_pct", "injury_rate",
            "board_independence_pct", "board_diversity_pct",
            "exec_comp_esg_linked", "ethics_compliance_score",
        ]
        builder = CompositeIndexBuilder(pipeline_config)
        indexed = builder.build(raw_df, indicator_cols=indicator_cols)
        assert "ESG_composite" in indexed.columns
        assert indexed["ESG_composite"].notna().all()
        assert len(indexed) == len(raw_df)

    def test_step4_financial_scoring(self, raw_df, pipeline_config):
        """Step 4: Financial scoring."""
        scorer = FinancialScorer(pipeline_config)
        scored = scorer.compute_financial_score(raw_df)
        assert "financial_score" in scored.columns
        assert scored["financial_score"].between(0, 100).all()

    def test_step5_similarity_matrix(self, raw_df, pipeline_config):
        """Step 5: Similarity matrix computation."""
        # First build ESG index to get feature columns
        indicator_cols = ["scope1_emissions", "renewable_energy_pct", "board_independence_pct"]
        builder = CompositeIndexBuilder(pipeline_config)
        indexed = builder.build(raw_df, indicator_cols=indicator_cols)

        norm_cols = [c for c in indexed.columns if c.endswith("_norm")]
        sim = compute_similarity_matrix(indexed, norm_cols)
        assert sim.shape == (len(raw_df), len(raw_df))
        # Diagonal should be ~1.0
        np.testing.assert_array_almost_equal(np.diag(sim.values), np.ones(len(raw_df)), decimal=3)

    def test_step6_preference_scoring(self, raw_df, pipeline_config):
        """Step 6: Preference scoring end-to-end."""
        # Prepare all score columns
        raw_df["ESG_composite"] = np.random.uniform(30, 80, len(raw_df))
        raw_df["financial_score"] = np.random.uniform(30, 80, len(raw_df))
        raw_df["market_score"] = 50.0
        raw_df["operational_score"] = 50.0
        raw_df["risk_adjusted_score"] = 50.0
        raw_df["growth_score"] = 50.0
        raw_df["value_score"] = 50.0
        raw_df["stability_score"] = 50.0
        raw_df["similarity_rank"] = np.random.uniform(0, 1, len(raw_df))
        raw_df["sector_position"] = np.random.uniform(0, 1, len(raw_df))

        scorer = PreferenceScorer(pipeline_config)
        scores = scorer.compute_preference_score(raw_df, investor_profile="balanced")
        assert len(scores) == len(raw_df)
        assert scores.between(0, 100).all()

    def test_full_pipeline_output_shapes(self, raw_df, pipeline_config):
        """Full pipeline preserves row count and adds expected columns."""
        # Step A: Clean
        cleaned, _ = standardize_and_clean(
            raw_df, index_cfg=pipeline_config, required_cols=[]
        )
        assert len(cleaned) == len(raw_df)

        # Step B: ESG index
        indicator_cols = ["scope1_emissions", "renewable_energy_pct", "board_independence_pct"]
        builder = CompositeIndexBuilder(pipeline_config)
        indexed = builder.build(cleaned, indicator_cols=indicator_cols)
        assert "ESG_composite" in indexed.columns

        # Step C: Financial score
        scorer = FinancialScorer(pipeline_config)
        final = scorer.compute_financial_score(indexed)
        assert "financial_score" in final.columns
        assert len(final) == len(raw_df)
