"""Investment Preference Scoring Module.

Combines ESG scores, financial metrics, market factors, operational quality,
risk-adjusted returns, growth, value, stability, similarity, and sector position
into a composite investment preference score using 10 sub-factors.

Weight rationale (from literature and empirical analysis):
- Financial quality factors explain most cross-sectional return variation (Fama-French)
- ESG integration adds risk mitigation and downside protection (Giese et al. 2019)
- Market momentum captures short-term price dynamics (Jegadeesh & Titman 1993)
- Operational quality is a proxy for sustainable competitive advantage (Novy-Marx 2013)
- Risk-adjusted metrics (Sharpe/Sortino) reward efficient return generation
- Similarity and sector position add peer-relative context

Aggregation modes (variance-equalized scoring):
- "rank": Convert each factor to percentile ranks (0-100) before weighting.
  Guarantees each factor contributes equally per unit of weight regardless of
  the original score distribution's variance.
- "variance_equalized": Standardize each factor to mean=50, std=10 before
  weighting.  Preserves cardinal distance information while equalizing spread.
- "raw": Use raw 0-100 scores as-is (legacy behaviour).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# All 10 score components used in preference scoring
ALL_SCORE_COMPONENTS = [
    "esg_score", "financial_score", "market_score", "operational_score",
    "risk_adjusted_score", "growth_score", "value_score", "stability_score",
    "similarity_rank", "sector_position",
]

# Mapping from config key -> DataFrame column name
SCORE_COLUMN_MAP = {
    "esg_score": "ESG_composite",
    "financial_score": "financial_score",
    "market_score": "market_score",
    "operational_score": "operational_score",
    "risk_adjusted_score": "risk_adjusted_score",
    "growth_score": "growth_score",
    "value_score": "value_score",
    "stability_score": "stability_score",
    "similarity_rank": "similarity_rank",
    "sector_position": "sector_position",
}


class PreferenceScorer:
    """Compute investment preference scores from 10 sub-factor components."""

    _VALID_MODES = frozenset({"rank", "variance_equalized", "raw"})

    def __init__(self, config: dict[str, Any]):
        self.config = config
        pref_cfg = config.get("preference_scoring", {})
        self.profiles = pref_cfg.get("investor_profiles", {})
        self.aggregation_mode: str = pref_cfg.get("aggregation_mode", "rank")
        self.quality_gate_quantile: float = float(pref_cfg.get("quality_gate_quantile", 0.30))
        self.quality_gate_min_universe: int = int(pref_cfg.get("quality_gate_min_universe", 20))
        self.power_transform_exponent: float = float(pref_cfg.get("power_transform_exponent", 1.5))
        self.momentum_tilt_weight: float = float(pref_cfg.get("momentum_tilt_weight", 0.15))

    def compute_preference_score(
        self,
        df: pd.DataFrame,
        *,
        esg_score_col: str = "ESG_composite",
        financial_score_col: str | None = None,
        similarity_rank_col: str | None = None,
        sector_position_col: str | None = None,
        investor_profile: str = "balanced",
        aggregation_mode: str | None = None,
    ) -> pd.Series:
        """Compute composite investment preference score.

        Uses ALL 10 sub-factor components with profile-specific weights.
        Components on 0-100 scale are used directly.
        Components on 0-1 scale (similarity_rank, sector_position) are rescaled.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with component scores
        investor_profile : str
            Investor profile: "esg_first", "balanced", "financial_first"
        aggregation_mode : str | None
            How to normalise factor columns before weighting:
            - ``"rank"`` (default): percentile-rank transform to 0-100 scale.
              Guarantees each factor contributes equally per unit of weight.
            - ``"variance_equalized"``: standardise to mean=50, std=10.
            - ``"raw"``: use raw scores as-is (legacy behaviour).
            When *None*, falls back to ``self.aggregation_mode`` (from config,
            default ``"rank"``).

        Returns
        -------
        pd.Series
            Investment preference scores (0-100 scale)
        """
        mode = aggregation_mode if aggregation_mode is not None else self.aggregation_mode
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid aggregation_mode '{mode}'. "
                f"Must be one of {sorted(self._VALID_MODES)}"
            )
        apply_momentum_tilt = investor_profile in {"momentum_tilt", "pref_momentum_tilt"}
        profile_key = "balanced" if apply_momentum_tilt else investor_profile

        # Get weights for profile
        if profile_key in self.profiles:
            weights = self.profiles[profile_key].copy()
        else:
            available_profiles = list(self.profiles.keys())
            logger.warning(
                "Profile '%s' not found in config (available: %s). Using default balanced weights.",
                profile_key,
                available_profiles,
            )
            # Default balanced weights — sourced from constants.DEFAULT_WEIGHTS
            # to stay consistent with config/index_config.yaml
            from ..constants import DEFAULT_WEIGHTS
            # Reverse map: DEFAULT_WEIGHTS uses DataFrame column names (ESG_composite),
            # but self.profiles/weights expects config keys (esg_score).
            _COL_TO_CONFIG_KEY = {v: k for k, v in SCORE_COLUMN_MAP.items()}
            weights = {
                _COL_TO_CONFIG_KEY.get(col, col): w
                for col, w in DEFAULT_WEIGHTS.items()
            }

        # Validate weights sum approximately to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                "Profile '%s' weights sum to %.4f (not 1.0). Normalizing.",
                investor_profile,
                total_weight,
            )
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        score = pd.Series(0.0, index=df.index)

        for component, weight in weights.items():
            if weight <= 0:
                continue

            col = SCORE_COLUMN_MAP.get(component, component)

            if col in df.columns:
                vals = df[col].fillna(df[col].median() if df[col].notna().any() else 50)

                # similarity_rank and sector_position are on 0-1 scale -> rescale
                if component in ("similarity_rank", "sector_position"):
                    if vals.max() <= 1.0:
                        vals = vals * 100

                # --- Apply aggregation normalization BEFORE weighting ---
                vals = self._normalize_factor(vals, mode)

                score += weight * vals

        # Keep raw mode as strict legacy behaviour.
        if mode != "raw":
            score = score.clip(0, 100)

            # Improve score differentiation while preserving ordering.
            if mode == "rank" and self.power_transform_exponent != 1.0:
                score = ((score / 100.0) ** self.power_transform_exponent) * 100.0

            # Quality gate: penalize the lowest financial quality names.
            fin_col = financial_score_col or SCORE_COLUMN_MAP["financial_score"]
            if (
                len(df) >= self.quality_gate_min_universe
                and fin_col in df.columns
                and df[fin_col].notna().any()
            ):
                threshold = df[fin_col].quantile(self.quality_gate_quantile)
                quality_mask = df[fin_col].fillna(-np.inf) >= threshold
                score = score.where(quality_mask, other=0.0)

            # Momentum-tilted profile: blend base preference with momentum percentile.
            if apply_momentum_tilt:
                momentum_pct = self._compute_momentum_percentile(df)
                score = (1.0 - self.momentum_tilt_weight) * score + self.momentum_tilt_weight * momentum_pct

        return score.clip(0, 100)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_factor(vals: pd.Series, mode: str) -> pd.Series:
        """Normalise a single factor column according to *mode*.

        Parameters
        ----------
        vals : pd.Series
            Factor values (already rescaled to ~0-100 range and NaN-filled).
        mode : str
            ``"rank"``, ``"variance_equalized"``, or ``"raw"``.

        Returns
        -------
        pd.Series
            Normalised values on a 0-100 compatible scale.
        """
        if mode == "raw":
            return vals

        if mode == "rank":
            # Percentile-rank: each value mapped to its percentile in [0, 100].
            # rankdata(method="average") handles ties; pct=True → 0-1 range.
            arr = vals.to_numpy(dtype=float, na_value=np.nan)
            ranked = rankdata(arr, method="average")
            # Convert to 0-100 percentile scale
            return pd.Series(ranked / len(ranked) * 100, index=vals.index)

        if mode == "variance_equalized":
            # Standardise to mean=50, std=10 (keeps cardinal distances, equalises spread)
            std = vals.std()
            if std == 0 or np.isnan(std):
                return pd.Series(50.0, index=vals.index)
            return ((vals - vals.mean()) / std) * 10 + 50

        # Should be unreachable due to validation in compute_preference_score
        raise ValueError(f"Unknown aggregation mode: {mode}")  # pragma: no cover

    @staticmethod
    def _compute_momentum_percentile(df: pd.DataFrame) -> pd.Series:
        """Compute momentum percentile (0-100), with robust fallbacks."""
        if "momentum_percentile" in df.columns:
            momentum = df["momentum_percentile"]
        else:
            momentum_cols = [
                "price_momentum_1m", "price_momentum_3m", "price_momentum_6m", "price_momentum_12m",
            ]
            available = [col for col in momentum_cols if col in df.columns]
            if available:
                momentum = df[available].mean(axis=1, skipna=True)
            elif "market_score" in df.columns:
                momentum = df["market_score"]
            else:
                return pd.Series(50.0, index=df.index)

        momentum = momentum.fillna(momentum.median() if momentum.notna().any() else 0.0)
        ranked = rankdata(momentum.to_numpy(dtype=float), method="average")
        return pd.Series(ranked / len(ranked) * 100, index=df.index)

    def rank_companies(
        self,
        df: pd.DataFrame,
        *,
        preference_score_col: str = "preference_score",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Rank companies by investment preference score."""
        if preference_score_col not in df.columns:
            raise ValueError(f"Preference score column '{preference_score_col}' not found")

        ranked = df.sort_values(preference_score_col, ascending=False)
        if top_n:
            ranked = ranked.head(top_n)

        ranked["preference_rank"] = range(1, len(ranked) + 1)
        return ranked
