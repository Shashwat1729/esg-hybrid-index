"""Shared constants for the multi-factor ESG codebase.

Centralises column definitions, variable-type metadata, and default
configuration values so that they are defined in exactly one place.

The single source of truth for weight profiles is config/index_config.yaml.
DEFAULT_WEIGHTS mirrors the *balanced* profile from that file.  Any update
to the YAML must be reflected here (or loaded dynamically via
``load_profiles_from_config``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Identifier columns
# ---------------------------------------------------------------------------
ID_COLS = ["ticker", "company_name", "currency", "sector", "industry", "country"]

# ---------------------------------------------------------------------------
# ESG indicator columns
# ---------------------------------------------------------------------------
ESG_ENV_COLS = [
    "scope1_emissions", "scope2_emissions", "scope3_emissions",
    "emissions_intensity", "renewable_energy_pct", "energy_efficiency",
    "water_usage_intensity", "waste_recycling_pct", "carbon_reduction_target",
    "environmental_fines",
    "r_d_intensity",  # Innovation proxy (Porter & van der Linde, 1995)
]
ESG_SOC_COLS = [
    "employee_turnover", "gender_diversity_pct", "women_management_pct",
    "pay_gap_ratio", "injury_rate", "safety_training_hours",
    "employee_satisfaction", "community_investment_pct",
    "supply_chain_audit_pct", "human_rights_policy",
    "revenue_per_employee",  # Human capital proxy (Edmans, 2011)
]
ESG_GOV_COLS = [
    "board_independence_pct", "board_diversity_pct", "board_size",
    "exec_comp_esg_linked", "ceo_pay_ratio", "shareholder_rights_score",
    "ethics_compliance_score", "anti_corruption_policy",
    "data_privacy_score", "tax_transparency_score",
    "esg_controversy_score", "esg_risk_rating",
]
ESG_COLS = ESG_ENV_COLS + ESG_SOC_COLS + ESG_GOV_COLS

# ---------------------------------------------------------------------------
# Category → indicator mapping for ESG pillar score computation
# Keys match the category names used in config/index_config.yaml
# (esg_index.category_weights).  Values are the *exact* base column names
# whose normalised versions ("{col}_norm") will be looked up at runtime.
# ---------------------------------------------------------------------------
CATEGORY_INDICATOR_MAP: dict[str, list[str]] = {
    # Environmental
    "emissions": [
        "scope1_emissions", "scope2_emissions", "scope3_emissions",
        "emissions_intensity", "carbon_reduction_target", "environmental_fines",
    ],
    "energy": ["renewable_energy_pct", "energy_efficiency"],
    "water": ["water_usage_intensity"],
    "waste": ["waste_recycling_pct"],
    # Environmental — innovation proxy (Porter & van der Linde, 1995;
    # Hall et al., 2005): R&D spending signals investment in cleaner
    # processes and green product development.
    "innovation": ["r_d_intensity"],
    # Social
    "labor": ["employee_turnover", "employee_satisfaction", "supply_chain_audit_pct"],
    "diversity": ["gender_diversity_pct", "women_management_pct", "pay_gap_ratio"],
    "health_safety": ["injury_rate", "safety_training_hours"],
    "community": ["community_investment_pct", "human_rights_policy"],
    # Social — human capital proxy (Becker, 1964; Edmans, 2011):
    # Revenue per employee captures workforce productivity and the
    # value created per unit of human capital deployed.
    "human_capital": ["revenue_per_employee"],
    # Governance
    "board": ["board_independence_pct", "board_diversity_pct", "board_size"],
    "comp": ["exec_comp_esg_linked", "ceo_pay_ratio"],
    "shareholder_rights": ["shareholder_rights_score"],
    "ethics": [
        "ethics_compliance_score", "anti_corruption_policy",
        "data_privacy_score", "tax_transparency_score",
        "esg_controversy_score", "esg_risk_rating",
    ],
}

# ---------------------------------------------------------------------------
# Financial / market / operational columns
# ---------------------------------------------------------------------------
FINANCIAL_COLS = [
    "market_cap", "total_revenue", "ebitda", "net_income", "gross_profit",
    "total_debt", "total_cash",
    "roa", "roe", "debt_to_equity", "current_ratio", "quick_ratio",
    "free_cashflow", "operating_cashflow",
    "trailing_pe", "forward_pe", "price_to_book", "price_to_sales",
    "enterprise_to_revenue", "enterprise_to_ebitda",
    "dividend_yield", "payout_ratio",
    "revenue_growth", "earnings_growth", "earnings_quarterly_growth",
    # Yahoo raw margins (decimal ratios from yf.Ticker.info, e.g. 0.15 = 15%):
    #   profit_margins, gross_margins, operating_margins
    # Computed margins (percentage, derived in 01_download_data.py from financials):
    #   net_margin = net_income / total_revenue * 100
    #   operating_margin = ebitda / total_revenue * 100
    #   gross_margin = gross_profit / total_revenue * 100
    # Both sets are kept because Yahoo raw fields may be available even when the
    # underlying line items (net_income, ebitda, gross_profit) are missing.
    "profit_margins", "gross_margins", "operating_margins",
    "net_margin", "operating_margin", "gross_margin",
    "debt_to_ebitda", "cash_flow_to_debt", "fcf_margin",
]

MARKET_COLS = [
    "price", "avg_daily_volume", "avg_daily_volume_30d", "avg_daily_volume_90d",
    "price_volatility", "price_volatility_30d",
    "price_momentum_1m", "price_momentum_3m", "price_momentum_6m", "price_momentum_12m",
    "beta",
    # Removed (Issue M6): free_float_pct, bid_ask_spread — synthetic random
    # noise with zero discriminating power; see scripts/01_download_data.py
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
# Variable-type classification sets
# ---------------------------------------------------------------------------
BINARY_VARS = {"carbon_reduction_target", "human_rights_policy", "anti_corruption_policy"}

ORDINAL_VARS = {"board_size", "esg_risk_rating"}

# ---------------------------------------------------------------------------
# Zero-calibration proxy indicators — these have NO real data for validation.
# They are constructed entirely from financial proxies and should be
# heavily down-weighted or excluded from ESG pillar scoring.
# Source: reports/tables/proxy_calibration_report.csv (n_real = 0 for all)
# ---------------------------------------------------------------------------
ZERO_CALIBRATION_PROXIES: set[str] = {
    # Environmental
    "renewable_energy_pct",     # proxy: energy_efficiency_proxy
    "scope1_emissions",         # proxy: emissions_intensity_proxy
    "energy_efficiency",        # proxy: capital_efficiency_proxy
    "waste_recycling_pct",      # proxy: waste_efficiency_proxy
    # Social
    "employee_satisfaction",    # proxy: employee_productivity_proxy
    "gender_diversity_pct",     # proxy: workforce_investment_proxy
    "safety_training_hours",    # proxy: workforce_scale_proxy
    "supply_chain_audit_pct",   # proxy: supply_chain_proxy
    "community_investment_pct", # proxy: community_proxy
    # Governance
    "anti_corruption_policy",   # proxy: financial_transparency_proxy
    "board_diversity_pct",      # proxy: board_quality_proxy
}

# ---------------------------------------------------------------------------
# ESG indicator direction – indicators where *lower* raw values are better.
# After normalization these must be flipped so that a low raw value produces
# a *high* normalised score (e.g. low emissions → high E score).
# ---------------------------------------------------------------------------
ESG_LOWER_IS_BETTER: set[str] = {
    # Environmental – emission / resource-use metrics
    "scope1_emissions",
    "scope2_emissions",
    "scope3_emissions",
    "emissions_intensity",
    "water_usage_intensity",
    "environmental_fines",
    # Social – workforce-risk metrics
    "employee_turnover",
    "injury_rate",
    # NOTE: pay_gap_ratio is NOT lower-is-better in our encoding.
    # Our synthetic data generates values near 1.0 = equal pay (good),
    # values near 0.60 = large gap (bad).  Higher ratio = better.
    # Governance – risk ratings where higher raw = worse
    # NOTE: esg_controversy_score is NOT lower-is-better in our encoding.
    # Our synthetic data: better base_quality → higher controversy_score
    # (= fewer controversies).  Higher score = better.
    "esg_risk_rating",          # lower risk rating = better (100 - bq*50)
    "ceo_pay_ratio",            # lower CEO-to-median pay ratio = better
}

# ---------------------------------------------------------------------------
# Score columns used in preference scoring
# ---------------------------------------------------------------------------
SCORE_COLUMNS = [
    "ESG_composite", "financial_score", "market_score", "operational_score",
    "risk_adjusted_score", "value_score", "growth_score", "stability_score",
    "similarity_rank", "sector_position",
]

# Same list without market_score (used in ex-market evaluation paths)
SCORE_COLUMNS_EX_MARKET = [c for c in SCORE_COLUMNS if c != "market_score"]

# ---------------------------------------------------------------------------
# Default balanced investor-profile weights
# ---------------------------------------------------------------------------
# Source of truth: config/index_config.yaml → preference_scoring.investor_profiles.balanced
# Keys use DataFrame column names (ESG_composite, not esg_score).
# The config YAML uses esg_score; the mapping is handled by
# ``load_profiles_from_config`` and the ``_CONFIG_KEY_TO_COLUMN`` map.
#
# CIRCULARITY FIX (Issue C1):
# market_score includes price_momentum_1m/3m/6m as sub-indicators.
# Evaluation scripts use these same momentum columns as return proxies.
# Including market_score in portfolio selection AND momentum as the return
# measure creates circular dependency that inflates IC, Sharpe, and alpha.
#
# Resolution:
#   DEFAULT_WEIGHTS           = ex-market weights (PRIMARY — no circularity)
#   DEFAULT_WEIGHTS_WITH_MARKET = original weights (kept for audit/backward compat)
# ---------------------------------------------------------------------------

# Original weights including market_score (CONTAMINATED — kept for reference)
DEFAULT_WEIGHTS_WITH_MARKET: dict[str, float] = {
    "ESG_composite": 0.22,
    "financial_score": 0.18,
    "market_score": 0.05,
    "operational_score": 0.06,
    "risk_adjusted_score": 0.15,
    "growth_score": 0.14,
    "value_score": 0.04,
    "stability_score": 0.08,
    "similarity_rank": 0.00,
    "sector_position": 0.08,
}

# Ex-market weights: market_score's weight redistributed proportionally to the
# remaining 9 factors, preserving their relative ratios.
# Each factor gets: w_new = w_old / sum(non-market weights)
_non_market = {k: v for k, v in DEFAULT_WEIGHTS_WITH_MARKET.items() if k != "market_score"}
_non_market_sum = sum(_non_market.values())
DEFAULT_WEIGHTS_EX_MARKET: dict[str, float] = {
    k: round(v / _non_market_sum, 4) for k, v in _non_market.items()
}
# Ensure weights sum to exactly 1.0 by adjusting the largest weight
_wsum = sum(DEFAULT_WEIGHTS_EX_MARKET.values())
if abs(_wsum - 1.0) > 1e-6:
    _max_key = max(DEFAULT_WEIGHTS_EX_MARKET, key=DEFAULT_WEIGHTS_EX_MARKET.get)
    DEFAULT_WEIGHTS_EX_MARKET[_max_key] = round(DEFAULT_WEIGHTS_EX_MARKET[_max_key] + (1.0 - _wsum), 4)

# PRIMARY weights — used by all downstream scripts (clean, no circularity)
DEFAULT_WEIGHTS: dict[str, float] = DEFAULT_WEIGHTS_EX_MARKET

# ---------------------------------------------------------------------------
# Config key ↔ DataFrame column mapping
# ---------------------------------------------------------------------------
# The YAML config uses ``esg_score`` while the DataFrame column is
# ``ESG_composite``.  All other keys are identical.
_CONFIG_KEY_TO_COLUMN: dict[str, str] = {
    "esg_score": "ESG_composite",
}


def _map_keys_to_columns(profile: dict[str, float]) -> dict[str, float]:
    """Translate config-level key names to DataFrame column names."""
    return {
        _CONFIG_KEY_TO_COLUMN.get(k, k): v
        for k, v in profile.items()
    }


def _find_project_root() -> Path:
    """Walk up from CWD / this file to locate the project root (has config/)."""
    candidates = [Path.cwd()] + list(Path.cwd().parents)
    this_file = Path(__file__)
    candidates += [this_file.parent.parent]  # src/ -> project root
    for p in candidates:
        if (p / "config" / "index_config.yaml").exists():
            return p
    return Path.cwd()


def load_profiles_from_config(
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Load all investor profiles from *config/index_config.yaml*.

    Returns a dict mapping profile name → {column_name: weight}.
    Keys are translated to DataFrame column names (``ESG_composite`` etc.).

    Falls back to a dict containing only the ``balanced`` profile from
    ``DEFAULT_WEIGHTS`` if the file cannot be read.
    """
    import yaml  # deferred so the module loads even without pyyaml

    if config_path is None:
        config_path = _find_project_root() / "config" / "index_config.yaml"
    else:
        config_path = Path(config_path)

    try:
        with open(config_path, "r") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh) or {}
        raw_profiles: dict[str, dict[str, float]] = (
            cfg.get("preference_scoring", {})
               .get("investor_profiles", {})
        )
        if not raw_profiles:
            raise ValueError("No investor_profiles found in config")
        return {
            name: _map_keys_to_columns(weights)
            for name, weights in raw_profiles.items()
        }
    except Exception:
        # Graceful fallback – at minimum expose the balanced profile
        return {"balanced": DEFAULT_WEIGHTS.copy()}
