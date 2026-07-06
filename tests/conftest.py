"""Shared pytest fixtures for the multi-factor ESG test suite.

Provides synthetic DataFrames, config dicts, and path helpers used across
all test modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup – make sure `import src.*` works regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures: paths
# ---------------------------------------------------------------------------
@pytest.fixture
def project_root() -> Path:
    """Return the absolute path to the project root."""
    return PROJECT_ROOT


@pytest.fixture
def config_dir(project_root) -> Path:
    """Return the path to config/ directory."""
    return project_root / "config"


# ---------------------------------------------------------------------------
# Fixtures: config dicts
# ---------------------------------------------------------------------------
@pytest.fixture
def index_config() -> dict:
    """Minimal index config dict matching index_config.yaml structure."""
    return {
        "esg_index": {
            "normalization": {
                "method": "zscore",
                "by_group": {"sector": False},
                "winsorize": {"enabled": False},
            },
            "missing_data": {
                "min_indicator_coverage": 0.60,
                "imputation": {
                    "method": "group_median",
                    "group_keys": ["sector"],
                },
            },
            "pillar_weights": {
                "E": 0.35,
                "S": 0.35,
                "G": 0.30,
            },
            "category_weights": {
                "E": {"emissions": 0.40, "energy": 0.25, "water": 0.20, "waste": 0.15},
                "S": {"labor": 0.35, "diversity": 0.30, "health_safety": 0.25, "community": 0.10},
                "G": {"board": 0.35, "comp": 0.25, "shareholder_rights": 0.20, "ethics": 0.20},
            },
        },
        "financial_scoring": {
            "categories": {
                "profitability": {
                    "weight_range": {"default": 0.30},
                    "indicators": ["roa", "roe", "net_margin"],
                },
                "stability": {
                    "weight_range": {"default": 0.20},
                    "indicators": ["debt_to_equity", "market_cap"],
                },
                "valuation": {
                    "weight_range": {"default": 0.15},
                    "indicators": ["trailing_pe", "price_to_book"],
                },
            },
            "inverse_indicators": ["debt_to_equity", "trailing_pe", "price_to_book"],
            "normalization": {"method": "zscore", "by_group": {}},
        },
        "market_factors": {
            "categories": {
                "liquidity": {
                    "weight_range": {"default": 0.40},
                    "indicators": ["avg_daily_volume"],
                },
                "volatility": {
                    "weight_range": {"default": 0.30},
                    "indicators": ["price_volatility", "beta"],
                },
                "momentum": {
                    "weight_range": {"default": 0.30},
                    "indicators": ["price_momentum_1m", "price_momentum_3m"],
                },
            },
            "inverse_indicators": ["price_volatility", "beta"],
            "normalization": {"method": "zscore", "by_group": {}},
        },
        "preference_scoring": {
            "investor_profiles": {
                "esg_first": {
                    "esg_score": 0.40,
                    "financial_score": 0.10,
                    "market_score": 0.05,
                    "operational_score": 0.08,
                    "risk_adjusted_score": 0.05,
                    "growth_score": 0.05,
                    "value_score": 0.05,
                    "stability_score": 0.07,
                    "similarity_rank": 0.10,
                    "sector_position": 0.05,
                },
                "balanced": {
                    "esg_score": 0.20,
                    "financial_score": 0.20,
                    "market_score": 0.10,
                    "operational_score": 0.10,
                    "risk_adjusted_score": 0.08,
                    "growth_score": 0.10,
                    "value_score": 0.08,
                    "stability_score": 0.05,
                    "similarity_rank": 0.05,
                    "sector_position": 0.04,
                },
                "financial_first": {
                    "esg_score": 0.05,
                    "financial_score": 0.30,
                    "market_score": 0.15,
                    "operational_score": 0.10,
                    "risk_adjusted_score": 0.10,
                    "growth_score": 0.10,
                    "value_score": 0.10,
                    "stability_score": 0.05,
                    "similarity_rank": 0.03,
                    "sector_position": 0.02,
                },
            },
        },
    }


@pytest.fixture
def data_sources_config() -> dict:
    """Minimal data sources config dict."""
    return {
        "data_sources": {
            "public": {
                "financial": {
                    "yahoo_finance": {"enabled": False},
                },
                "filings": {
                    "sec_edgar": {"enabled": False},
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Fixtures: synthetic DataFrames
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_esg_df() -> pd.DataFrame:
    """Small synthetic ESG dataset (8 companies, mix of sectors)."""
    rng = np.random.RandomState(42)
    n = 8
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "company_name": [f"Company_{i}" for i in range(n)],
            "sector": ["Tech", "Tech", "Health", "Health", "Finance", "Finance", "Energy", "Energy"],
            "country": ["US", "IN", "US", "IN", "US", "IN", "US", "IN"],
            # ESG indicators
            "scope1_emissions": rng.uniform(10, 100, n),
            "scope2_emissions": rng.uniform(5, 50, n),
            "emissions_intensity": rng.uniform(0.1, 1.0, n),
            "renewable_energy_pct": rng.uniform(0, 100, n),
            "energy_efficiency": rng.uniform(50, 100, n),
            "water_usage_intensity": rng.uniform(0.5, 5.0, n),
            "waste_recycling_pct": rng.uniform(10, 90, n),
            "employee_turnover": rng.uniform(5, 30, n),
            "gender_diversity_pct": rng.uniform(20, 60, n),
            "women_management_pct": rng.uniform(10, 50, n),
            "injury_rate": rng.uniform(0, 5, n),
            "board_independence_pct": rng.uniform(40, 90, n),
            "board_diversity_pct": rng.uniform(10, 50, n),
            "exec_comp_esg_linked": rng.uniform(0, 100, n),
            "ceo_pay_ratio": rng.uniform(50, 500, n),
            "ethics_compliance_score": rng.uniform(40, 100, n),
        }
    )


@pytest.fixture
def sample_financial_df() -> pd.DataFrame:
    """Small synthetic financial dataset (8 companies)."""
    rng = np.random.RandomState(42)
    n = 8
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "sector": ["Tech", "Tech", "Health", "Health", "Finance", "Finance", "Energy", "Energy"],
            "roa": rng.uniform(-0.05, 0.25, n),
            "roe": rng.uniform(-0.10, 0.40, n),
            "net_margin": rng.uniform(-0.05, 0.30, n),
            "operating_margin": rng.uniform(0.0, 0.35, n),
            "debt_to_equity": rng.uniform(0.1, 3.0, n),
            "trailing_pe": rng.uniform(5, 50, n),
            "price_to_book": rng.uniform(0.5, 10, n),
            "market_cap": rng.uniform(1e9, 1e12, n),
            "free_cashflow": rng.uniform(-1e8, 1e10, n),
            "total_revenue": rng.uniform(1e8, 1e11, n),
            "ebitda": rng.uniform(1e7, 1e10, n),
        }
    )


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Small synthetic market data (8 companies)."""
    rng = np.random.RandomState(42)
    n = 8
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "sector": ["Tech", "Tech", "Health", "Health", "Finance", "Finance", "Energy", "Energy"],
            "avg_daily_volume": rng.uniform(1e5, 1e7, n),
            "price_volatility": rng.uniform(0.1, 0.6, n),
            "beta": rng.uniform(0.5, 2.0, n),
            "price_momentum_1m": rng.normal(0, 5, n),
            "price_momentum_3m": rng.normal(2, 10, n),
            "price_momentum_6m": rng.normal(5, 15, n),
            # Removed (Issue M6): bid_ask_spread, free_float_pct — synthetic noise
        }
    )


@pytest.fixture
def sample_scores_df() -> pd.DataFrame:
    """Synthetic dataset with all 10 preference scoring components."""
    rng = np.random.RandomState(42)
    n = 8
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "sector": ["Tech", "Tech", "Health", "Health", "Finance", "Finance", "Energy", "Energy"],
            "ESG_composite": rng.uniform(30, 80, n),
            "financial_score": rng.uniform(30, 80, n),
            "market_score": rng.uniform(30, 80, n),
            "operational_score": rng.uniform(30, 80, n),
            "risk_adjusted_score": rng.uniform(30, 80, n),
            "growth_score": rng.uniform(30, 80, n),
            "value_score": rng.uniform(30, 80, n),
            "stability_score": rng.uniform(30, 80, n),
            "similarity_rank": rng.uniform(0, 1, n),
            "sector_position": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """DataFrame with intentional NaN values for imputation tests."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "sector": ["Tech", "Tech", "Health", "Health", "Tech", "Health"],
            "x": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
            "y": [10.0, 20.0, np.nan, 40.0, np.nan, 60.0],
            "z": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        }
    )
