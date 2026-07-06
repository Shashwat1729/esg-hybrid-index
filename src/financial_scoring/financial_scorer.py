"""Financial Metrics Scoring.

Computes financial scores from profitability, growth, efficiency, stability, and valuation metrics.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..index_construction.composite_index import normalize_indicators

logger = logging.getLogger(__name__)


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
        scale_to_score: bool = True,
    ) -> pd.DataFrame:
        """Compute composite financial score.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with financial indicator columns
        id_col : str, default "ticker"
            Identifier column name
        scale_to_score : bool, default True
            When True, applies ``50 + score * 20`` transformation and clips to
            [0, 100].  When False, returns raw z-score composite — useful when a
            downstream step will apply its own scaling.

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

        # Validate: check which indicators are present
        available_indicators = [c for c in all_indicators if c in df.columns]
        missing_indicators = [c for c in all_indicators if c not in df.columns]
        if missing_indicators:
            logger.info(
                "Financial indicators not in dataframe (skipped): %s",
                missing_indicators[:10],
            )
        if not available_indicators:
            warnings.warn("No financial indicators found in dataframe", UserWarning)
            logger.warning("No valid indicators for financial_score; defaulting to 50.0")
            df["financial_score"] = 50.0  # Default neutral score
            return df

        # ---------------------------------------------------------------
        # INVERSION / WINSORIZATION INTERACTION
        # ---------------------------------------------------------------
        # 1. We save originals for "lower_is_better" indicators (e.g.
        #    trailing_pe, price_to_book, debt_to_equity, price_volatility).
        # 2. We NEGATE these columns: df[col] = -df[col].
        # 3. normalize_indicators() then:
        #    a. Winsorizes at 1st/99th percentiles on the NEGATED scale
        #       (clips extreme-low original values, which are extreme-high
        #       after negation).
        #    b. Z-scores on the NEGATED scale, so a high original value
        #       maps to a low (bad) z-score.
        # 4. We restore originals AFTER scoring (line ~148), so the output
        #    DataFrame keeps original raw values but the *_norm columns
        #    reflect the inverted scoring direction.
        #
        # This is CORRECT: winsorization should limit extreme raw values
        # regardless of scoring direction, and the z-score should reflect
        # that (e.g.) lower P/E is better.  The key subtlety is that
        # winsorization clips on the INVERTED scale, which correctly
        # limits the *same* extreme original values — just from the
        # opposite tail of the distribution.
        # ---------------------------------------------------------------
        # Load inverse indicator list from config; fall back to empty list
        inverse_financial = self.financial_cfg.get("inverse_indicators", [])
        originals = {}
        for col in inverse_financial:
            if col in available_indicators and col in df.columns:
                originals[col] = df[col].copy()
                df[col] = -df[col].astype(float)  # Invert so higher-is-better

        # Pass lower_is_better=set() to prevent normalize_indicators from
        # applying the ESG_LOWER_IS_BETTER flip.  The financial scorer already
        # handles inversion above (df[col] = -df[col]) for its own
        # inverse_indicators.  Without this override, any indicator that
        # appears in BOTH inverse_financial AND ESG_LOWER_IS_BETTER would be
        # double-flipped (negated here, then flipped again during
        # normalization).  See audit: 2025-03 double-inversion fix.
        df = normalize_indicators(
            df,
            available_indicators,
            method=norm_cfg.get("method", "zscore"),
            by_group=norm_cfg.get("by_group", {}),
            winsorize={"enabled": True, "lower_quantile": 0.01, "upper_quantile": 0.99},
            lower_is_better=set(),  # inversion already handled above
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

            # Read per-indicator weights from config (if present)
            ind_weights_cfg = cat_config.get("indicator_weights", {})
            if norm_cols and ind_weights_cfg:
                # Build weight array aligned with norm_cols
                w_arr = []
                for ind in cat_indicators:
                    nc = f"{ind}_norm"
                    if nc in norm_cols:
                        iw = ind_weights_cfg.get(ind, {})
                        if isinstance(iw, dict):
                            w_arr.append(iw.get("default", 1.0))
                        else:
                            w_arr.append(float(iw) if iw else 1.0)
                w_arr = np.array(w_arr, dtype=float)
                if w_arr.sum() > 0:
                    df[f"{cat_name}_score"] = (df[norm_cols] * w_arr).sum(axis=1) / w_arr.sum()
                else:
                    df[f"{cat_name}_score"] = df[norm_cols].mean(axis=1)
            elif norm_cols:
                # Fallback: equal weighting when no indicator_weights defined
                df[f"{cat_name}_score"] = df[norm_cols].mean(axis=1)
            else:
                df[f"{cat_name}_score"] = 0.0
                warnings.warn(f"No indicators found for financial category '{cat_name}'", UserWarning)

        # Compute composite financial score
        total_weight = sum(category_weights.values())
        if total_weight == 0:
            logger.warning("No valid indicators for financial_score; defaulting to 50.0")
            df["financial_score"] = 50.0
            return df

        df["financial_score"] = 0.0
        for cat_name, cat_weight in category_weights.items():
            cat_score_col = f"{cat_name}_score"
            if cat_score_col in df.columns:
                df["financial_score"] += (cat_weight / total_weight) * df[cat_score_col].fillna(0)

        # Scale to 0-100 for interpretability (when scale_to_score is True)
        if scale_to_score:
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
        scale_to_score: bool = True,
    ) -> pd.DataFrame:
        """Compute composite market factor score.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with market factor columns
        id_col : str, default "ticker"
            Identifier column name
        scale_to_score : bool, default True
            When True, applies ``50 + score * 20`` transformation and clips to
            [0, 100].  When False, returns raw z-score composite — useful when a
            downstream step will apply its own scaling.

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
        missing_market = [c for c in all_indicators if c not in df.columns]
        if missing_market:
            logger.info(
                "Market indicators not in dataframe (skipped): %s",
                missing_market[:10],
            )
        if not available_indicators:
            warnings.warn("No market factor indicators found in dataframe", UserWarning)
            logger.warning("No valid indicators for market_score; defaulting to 50.0")
            df["market_score"] = 50.0
            return df

        # ---------------------------------------------------------------
        # INVERSION / WINSORIZATION INTERACTION (same pattern as
        # FinancialScorer — see detailed comment there)
        # For volatility-type indicators, lower is better (inverse).
        # Negation happens BEFORE normalize_indicators(), so winsorization
        # clips on the inverted scale.  Originals are restored after scoring.
        # ---------------------------------------------------------------
        # Load inverse indicator list from config; fall back to empty list
        inverse_cols = self.market_cfg.get("inverse_indicators", [])
        market_originals = {}
        for col in inverse_cols:
            if col in df.columns:
                market_originals[col] = df[col].copy()
                df[col] = -df[col].astype(float)  # Invert so higher-is-better

        # Pass lower_is_better=set() to prevent normalize_indicators from
        # applying the ESG_LOWER_IS_BETTER flip.  The market scorer already
        # handles inversion above (df[col] = -df[col]) for its own
        # inverse_indicators.  Without this override, any indicator that
        # appears in BOTH inverse_cols AND ESG_LOWER_IS_BETTER would be
        # double-flipped.  See audit: 2025-03 double-inversion fix.
        df = normalize_indicators(
            df,
            available_indicators,
            method=norm_cfg.get("method", "zscore"),
            by_group=norm_cfg.get("by_group", {}),
            winsorize={"enabled": True, "lower_quantile": 0.01, "upper_quantile": 0.99},
            lower_is_better=set(),  # inversion already handled above
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

            # Read per-indicator weights from config (if present)
            ind_weights_cfg = cat_config.get("indicator_weights", {})
            if norm_cols and ind_weights_cfg:
                # Build weight array aligned with norm_cols
                w_arr = []
                for ind in cat_indicators:
                    nc = f"{ind}_norm"
                    if nc in norm_cols:
                        iw = ind_weights_cfg.get(ind, {})
                        if isinstance(iw, dict):
                            w_arr.append(iw.get("default", 1.0))
                        else:
                            w_arr.append(float(iw) if iw else 1.0)
                w_arr = np.array(w_arr, dtype=float)
                if w_arr.sum() > 0:
                    df[f"market_{cat_name}_score"] = (df[norm_cols] * w_arr).sum(axis=1) / w_arr.sum()
                else:
                    df[f"market_{cat_name}_score"] = df[norm_cols].mean(axis=1)
            elif norm_cols:
                # Fallback: equal weighting when no indicator_weights defined
                df[f"market_{cat_name}_score"] = df[norm_cols].mean(axis=1)
            else:
                df[f"market_{cat_name}_score"] = 0.0

        # Compute composite market score
        total_weight = sum(category_weights.values())
        if total_weight == 0:
            logger.warning("No valid indicators for market_score; defaulting to 50.0")
            df["market_score"] = 50.0
            return df

        df["market_score"] = 0.0
        for cat_name, cat_weight in category_weights.items():
            cat_score_col = f"market_{cat_name}_score"
            if cat_score_col in df.columns:
                df["market_score"] += (cat_weight / total_weight) * df[cat_score_col].fillna(0)

        # Scale to 0-100 (when scale_to_score is True)
        if scale_to_score:
            df["market_score"] = 50 + (df["market_score"] * 20)
            df["market_score"] = df["market_score"].clip(0, 100)

        # Restore original values for inverted columns
        for col, orig_vals in market_originals.items():
            df[col] = orig_vals

        return df
