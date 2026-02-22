"""Financial Metrics Scoring.

Computes financial scores from profitability, growth, efficiency, stability, and valuation metrics.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..index_construction.composite_index import normalize_indicators


class FinancialScorer:
    """Compute comprehensive financial scores from multiple factor categories."""

    def __init__(self, config: dict[str, Any]):
        """Initialize with financial scoring configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Financial scoring config from index_config.yaml
        """
        self.config = config
        self.financial_cfg = config.get("financial_scoring", {})

    def compute_financial_score(
        self,
        df: pd.DataFrame,
        *,
        id_col: str = "ticker",
    ) -> pd.DataFrame:
        """Compute composite financial score.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with financial indicator columns
        id_col : str, default "ticker"
            Identifier column name

        Returns
        -------
        pd.DataFrame
            DataFrame with added financial category scores and composite financial_score
        """
        df = df.copy()
        categories = self.financial_cfg.get("categories", {})
        norm_cfg = self.financial_cfg.get("normalization", {})

        # Collect all financial indicators
        all_indicators = []
        for cat_name, cat_config in categories.items():
            indicators = cat_config.get("indicators", [])
            all_indicators.extend(indicators)

        # Normalize financial indicators
        available_indicators = [c for c in all_indicators if c in df.columns]
        if not available_indicators:
            warnings.warn("No financial indicators found in dataframe", UserWarning)
            df["financial_score"] = 50.0  # Default neutral score
            return df

        # For stability/valuation indicators, lower is often better (inverse)
        # Save original values to restore after normalization
        inverse_financial = ["debt_to_equity", "trailing_pe", "price_to_book"]
        originals = {}
        for col in inverse_financial:
            if col in available_indicators and col in df.columns:
                originals[col] = df[col].copy()
                df[col] = -df[col].astype(float)  # Invert so higher-is-better

        df = normalize_indicators(
            df,
            available_indicators,
            method=norm_cfg.get("method", "zscore"),
            by_group=norm_cfg.get("by_group", {}),
            winsorize={"enabled": True, "lower_quantile": 0.01, "upper_quantile": 0.99},
        )

        # Compute category scores
        category_weights = {}
        for cat_name, cat_config in categories.items():
            # Handle weight ranges (dict with min/max/default) or direct weight value
            weight_val = cat_config.get("weight_range", cat_config.get("weight", 0.0))
            if isinstance(weight_val, dict):
                cat_weight = weight_val.get("default", weight_val.get("max", 0.0))
            else:
                cat_weight = weight_val
            
            cat_indicators = cat_config.get("indicators", [])
            category_weights[cat_name] = cat_weight

            # Find normalized columns for this category
            norm_cols = [f"{ind}_norm" for ind in cat_indicators if f"{ind}_norm" in df.columns]
            if not norm_cols:
                # Try alternative patterns
                for ind in cat_indicators:
                    matching = [c for c in df.columns if c.endswith("_norm") and ind.lower() in c.lower()]
                    norm_cols.extend(matching)

            if norm_cols:
                # Compute category score (weighted average of normalized indicators)
                df[f"{cat_name}_score"] = df[norm_cols].mean(axis=1)
            else:
                df[f"{cat_name}_score"] = 0.0
                warnings.warn(f"No indicators found for financial category '{cat_name}'", UserWarning)

        # Compute composite financial score
        total_weight = sum(category_weights.values())
        if total_weight == 0:
            df["financial_score"] = 50.0
            return df

        df["financial_score"] = 0.0
        for cat_name, cat_weight in category_weights.items():
            cat_score_col = f"{cat_name}_score"
            if cat_score_col in df.columns:
                df["financial_score"] += (cat_weight / total_weight) * df[cat_score_col].fillna(0)

        # Scale to 0-100 for interpretability
        df["financial_score"] = 50 + (df["financial_score"] * 20)
        df["financial_score"] = df["financial_score"].clip(0, 100)

        # Restore original values for inverted columns
        for col, orig_vals in originals.items():
            df[col] = orig_vals

        return df


class MarketFactorScorer:
    """Compute market factor scores (liquidity, volatility, momentum)."""

    def __init__(self, config: dict[str, Any]):
        """Initialize with market factors configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Market factors config from index_config.yaml
        """
        self.config = config
        self.market_cfg = config.get("market_factors", {})

    def compute_market_score(
        self,
        df: pd.DataFrame,
        *,
        id_col: str = "ticker",
    ) -> pd.DataFrame:
        """Compute composite market factor score.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with market factor columns
        id_col : str, default "ticker"
            Identifier column name

        Returns
        -------
        pd.DataFrame
            DataFrame with added market category scores and composite market_score
        """
        df = df.copy()
        categories = self.market_cfg.get("categories", {})
        norm_cfg = self.market_cfg.get("normalization", {})

        # Collect all market indicators
        all_indicators = []
        for cat_name, cat_config in categories.items():
            indicators = cat_config.get("indicators", [])
            all_indicators.extend(indicators)

        # Normalize market indicators
        available_indicators = [c for c in all_indicators if c in df.columns]
        if not available_indicators:
            warnings.warn("No market factor indicators found in dataframe", UserWarning)
            df["market_score"] = 50.0
            return df

        # For volatility-type indicators, lower is better (inverse)
        # Save original values to restore after normalization
        inverse_cols = ["price_volatility", "beta", "bid_ask_spread"]
        market_originals = {}
        for col in inverse_cols:
            if col in df.columns:
                market_originals[col] = df[col].copy()
                df[col] = -df[col].astype(float)  # Invert so higher-is-better

        df = normalize_indicators(
            df,
            available_indicators,
            method=norm_cfg.get("method", "zscore"),
            by_group=norm_cfg.get("by_group", {}),
            winsorize={"enabled": True, "lower_quantile": 0.01, "upper_quantile": 0.99},
        )

        # Compute category scores
        category_weights = {}
        for cat_name, cat_config in categories.items():
            # Handle weight ranges (dict with min/max/default) or direct weight value
            weight_val = cat_config.get("weight_range", cat_config.get("weight", 0.0))
            if isinstance(weight_val, dict):
                cat_weight = weight_val.get("default", weight_val.get("max", 0.0))
            else:
                cat_weight = weight_val
            
            cat_indicators = cat_config.get("indicators", [])
            category_weights[cat_name] = cat_weight

            norm_cols = [f"{ind}_norm" for ind in cat_indicators if f"{ind}_norm" in df.columns]
            if not norm_cols:
                for ind in cat_indicators:
                    matching = [c for c in df.columns if c.endswith("_norm") and ind.lower() in c.lower()]
                    norm_cols.extend(matching)

            if norm_cols:
                df[f"market_{cat_name}_score"] = df[norm_cols].mean(axis=1)
            else:
                df[f"market_{cat_name}_score"] = 0.0

        # Compute composite market score
        total_weight = sum(category_weights.values())
        if total_weight == 0:
            df["market_score"] = 50.0
            return df

        df["market_score"] = 0.0
        for cat_name, cat_weight in category_weights.items():
            cat_score_col = f"market_{cat_name}_score"
            if cat_score_col in df.columns:
                df["market_score"] += (cat_weight / total_weight) * df[cat_score_col].fillna(0)

        # Scale to 0-100
        df["market_score"] = 50 + (df["market_score"] * 20)
        df["market_score"] = df["market_score"].clip(0, 100)

        # Restore original values for inverted columns
        for col, orig_vals in market_originals.items():
            df[col] = orig_vals

        return df
