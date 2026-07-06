"""Composite ESG Index Construction.

Implements three-level framework: Scope → Measurement → Weighting & Aggregation.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.constants import CATEGORY_INDICATOR_MAP, ESG_LOWER_IS_BETTER
from src.utils import robust_zscore

logger = logging.getLogger(__name__)


def normalize_indicators(
    df: pd.DataFrame,
    indicator_cols: list[str],
    *,
    method: str = "zscore",
    by_group: dict[str, bool] | None = None,
    winsorize: dict[str, Any] | None = None,
    lower_is_better: set[str] | None = None,
    orthogonalize_against: str | None = None,
) -> pd.DataFrame:
    """Normalize ESG indicators using specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with indicator columns
    indicator_cols : list[str]
        Column names to normalize
    method : str, default "zscore"
        Normalization method: "zscore", "minmax", "percentile", "robust_zscore"
    by_group : dict[str, bool] | None, optional
        Group-by keys for within-group normalization (e.g., {"sector": True})
    winsorize : dict[str, Any] | None, optional
        Winsorization config: {"enabled": bool, "lower_quantile": float, "upper_quantile": float}
    lower_is_better : set[str] | None, optional
        Set of indicator column names where lower raw values are desirable
        (e.g. emissions, injury_rate).  After normalization the ``_norm``
        column for these indicators is flipped so that a *low* raw value
        yields a *high* normalized score.  Defaults to
        ``ESG_LOWER_IS_BETTER`` from ``src.constants``.
    orthogonalize_against : str | None, optional
        Column name to orthogonalize each indicator against before
        normalization. When provided, each indicator ``x`` is transformed as
        ``x - beta * z`` where ``z`` is the reference column and
        ``beta = Cov(x, z) / Var(z)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized indicator columns (suffix "_norm")
    """
    by_group = by_group or {}
    winsorize = winsorize or {}
    if lower_is_better is None:
        lower_is_better = ESG_LOWER_IS_BETTER
    df = df.copy()

    # Filter to only numeric columns
    numeric_cols = []
    for col in indicator_cols:
        if col not in df.columns:
            warnings.warn(f"Indicator column '{col}' not found, skipping", UserWarning)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.warn(f"Indicator column '{col}' is not numeric, skipping", UserWarning)
            continue
        numeric_cols.append(col)
    
    if not numeric_cols:
        warnings.warn("No numeric indicator columns found for normalization", UserWarning)
        return df

    # ------------------------------------------------------------------
    # Winsorize if enabled
    # ------------------------------------------------------------------
    # NOTE: Winsorization modifies values on a COPY of the dataframe
    # (see df = df.copy() at line 60 above — the caller's data is safe).
    #
    # For ESG indicators, "lower_is_better" columns (e.g. emissions,
    # injury_rate) have NOT yet been inverted at this point — inversion
    # happens later in the "Flip direction" block below.  So
    # winsorization clips on the ORIGINAL (un-flipped) scale.
    #
    # For financial/market indicators, the caller (FinancialScorer /
    # MarketFactorScorer) NEGATES "lower_is_better" columns BEFORE
    # calling this function, and passes lower_is_better=set() to
    # suppress the flip block.  Therefore winsorization clips on the
    # NEGATED scale — which correctly limits the same extreme original
    # values, just from the opposite tail.
    # ------------------------------------------------------------------
    if winsorize.get("enabled", False):
        lower_q = winsorize.get("lower_quantile", 0.01)
        upper_q = winsorize.get("upper_quantile", 0.99)
        for col in numeric_cols:
            if col in df.columns:
                lower = df[col].quantile(lower_q)
                upper = df[col].quantile(upper_q)
                df[col] = df[col].clip(lower=lower, upper=upper)

    # Optionally remove linear dependency on a reference column
    if orthogonalize_against is not None:
        if orthogonalize_against not in df.columns:
            warnings.warn(
                f"Orthogonalization column '{orthogonalize_against}' not found, skipping",
                UserWarning,
            )
        elif not pd.api.types.is_numeric_dtype(df[orthogonalize_against]):
            warnings.warn(
                f"Orthogonalization column '{orthogonalize_against}' is not numeric, skipping",
                UserWarning,
            )
        else:
            ref = df[orthogonalize_against]
            for col in numeric_cols:
                valid = df[col].notna() & ref.notna()
                if valid.sum() < 2:
                    continue
                ref_var = ref.loc[valid].var()
                if pd.isna(ref_var) or ref_var <= 1e-12:
                    continue
                beta = df.loc[valid, col].cov(ref.loc[valid]) / ref_var
                df.loc[valid, col] = df.loc[valid, col] - beta * ref.loc[valid]

    # Apply normalization
    norm_cols = []
    for col in numeric_cols:

        if method == "zscore":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )
            else:
                df[f"{col}_norm"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
        elif method == "minmax":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
                )
            else:
                df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        elif method == "percentile":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: x.rank(pct=True) * 100
                )
            else:
                df[f"{col}_norm"] = df[col].rank(pct=True) * 100
        elif method == "robust_zscore":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    robust_zscore
                )
            else:
                df[f"{col}_norm"] = robust_zscore(df[col])
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        norm_cols.append(f"{col}_norm")

    # ------------------------------------------------------------------
    # Flip direction for "lower is better" indicators.
    # After flipping, a low raw value (good) yields a high norm score.
    #   zscore / robust_zscore : negate  (high z → low z)
    #   minmax                 : 1 − x   (keeps [0, 1] range)
    #   percentile             : 100 − x (keeps [0, 100] range)
    # ------------------------------------------------------------------
    for col in numeric_cols:
        if col not in lower_is_better:
            continue
        norm_col = f"{col}_norm"
        if norm_col not in df.columns:
            continue
        if method in ("zscore", "robust_zscore"):
            df[norm_col] = -df[norm_col]
        elif method == "minmax":
            df[norm_col] = 1.0 - df[norm_col]
        elif method == "percentile":
            df[norm_col] = 100.0 - df[norm_col]

    return df


def _load_materiality_map(path: str | None = None) -> dict[str, dict[str, float]]:
    """Load SASB sector-materiality E/S/G weights from YAML.

    Parameters
    ----------
    path : str | None
        Path to the materiality YAML file.  When *None* the default
        ``config/sasb_materiality.yaml`` relative to the project root is used.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of sector name → {"E": w_e, "S": w_s, "G": w_g}.
        Always includes a ``"default"`` key.
    """
    import yaml
    from pathlib import Path as _Path

    if path is None:
        # Walk up from this file to find the project root (contains config/)
        _here = _Path(__file__).resolve()
        for parent in (_here.parent, _here.parent.parent, _here.parent.parent.parent):
            candidate = parent / "config" / "sasb_materiality.yaml"
            if candidate.exists():
                path = str(candidate)
                break

    if path is None or not _Path(path).exists():
        logger.warning("Materiality YAML not found (%s); using default weights", path)
        return {"default": {"E": 0.35, "S": 0.35, "G": 0.30}}

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    materiality: dict[str, dict[str, float]] = raw.get("sector_materiality", {})
    # Ensure a default entry always exists
    if "default" not in materiality:
        materiality["default"] = {"E": 0.35, "S": 0.35, "G": 0.30}
    return materiality


def shrink_low_confidence_scores(
    score: pd.Series,
    sector: pd.Series,
    confidence: pd.Series,
    *,
    min_confidence: float = 0.4,
) -> pd.Series:
    """Shrink low-confidence scores toward the sector median."""
    adjusted = score.copy()
    if adjusted.empty:
        return adjusted

    sector_median = score.groupby(sector).transform("median")
    global_median = score.median()
    sector_median = sector_median.fillna(global_median)

    low_conf_mask = confidence < min_confidence
    adjusted.loc[low_conf_mask] = (
        confidence.loc[low_conf_mask] * score.loc[low_conf_mask]
        + (1.0 - confidence.loc[low_conf_mask]) * sector_median.loc[low_conf_mask]
    )
    return adjusted


def _harmonize_pillar_by_source(
    df: pd.DataFrame,
    pillar_col: str,
    source_col: str = "esg_data_source",
) -> pd.DataFrame:
    """Harmonize pillar scores across data sources using quantile normalization.

    M5 FIX: Different ESG data sources (Yahoo Finance vs SEC filings) produce
    systematically different score distributions for the same pillar. This
    creates a confound where the ranking reflects data source, not actual
    quality. We fix this by quantile-normalizing within each source group,
    then mapping to a common distribution.
    """
    if source_col not in df.columns or pillar_col not in df.columns:
        return df

    sources = df[source_col].dropna().unique()
    if len(sources) <= 1:
        return df  # No harmonization needed with single source

    # Quantile-normalize within each source, then map to overall percentile
    harmonized = df[pillar_col].copy()
    for src in sources:
        mask = df[source_col] == src
        if mask.sum() < 5:
            continue
        vals = df.loc[mask, pillar_col]
        # Convert to percentile rank within this source
        pct_rank = vals.rank(pct=True, na_option="keep")
        harmonized.loc[mask] = pct_rank

    # Now all sources are on 0-1 percentile scale; convert back to z-score scale
    # matching the overall distribution
    overall_mean = df[pillar_col].mean()
    overall_std = df[pillar_col].std()
    if overall_std > 1e-10:
        from scipy.stats import norm

        # Map percentile ranks to z-scores, then to original scale
        harmonized = harmonized.clip(0.001, 0.999)  # avoid infinite z
        z_scores = pd.Series(norm.ppf(harmonized.values), index=harmonized.index)
        harmonized = z_scores * overall_std + overall_mean

    df[pillar_col] = harmonized
    return df


def compute_pillar_scores(
    df: pd.DataFrame,
    pillar_config: dict[str, dict[str, float]],
    *,
    pillar_weights: dict[str, float] | None = None,
    sector_column: str | None = None,
    materiality_map: dict[str, dict[str, float]] | None = None,
    scale_to_score: bool = True,
    min_confidence: float = 0.4,
    pillar_weighting_method: str = "fixed",
) -> pd.DataFrame:
    """Compute ESG pillar scores (E, S, G) from normalized indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with normalized indicator columns (suffix "_norm")
    pillar_config : dict[str, dict[str, float]]
        Category weights within each pillar, e.g.:
        {
            "E": {"emissions": 0.40, "energy": 0.25, "water": 0.20, "waste": 0.15},
            "S": {"labor": 0.35, "diversity": 0.30, "health_safety": 0.25, "community": 0.10},
            "G": {"board": 0.35, "comp": 0.25, "shareholder_rights": 0.20, "ethics": 0.20}
        }
    pillar_weights : dict[str, float] | None, optional
        *Default* pillar weights for the composite score (used when
        ``sector_column`` is not provided or a company's sector has no
        entry in the materiality map).  Defaults to equal weights.
    sector_column : str | None, optional
        Column in *df* that contains the sector label for each company.
        When provided together with *materiality_map*, each company receives
        sector-specific E/S/G pillar weights instead of the flat defaults.
    materiality_map : dict[str, dict[str, float]] | None, optional
        Sector → {"E": w, "S": w, "G": w} mapping (e.g. loaded from
        ``config/sasb_materiality.yaml``).  Must include a ``"default"`` key.
    scale_to_score : bool, default True
        When True (default), applies the ``50 + z * 20`` transformation and
        clips to [0, 100] for human-readable scores.  When False, returns raw
        z-score composites without transformation — useful when a downstream
        step (e.g. Step 7b re-standardization) will apply its own scaling,
        avoiding lossy double-transformation at the tails.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: "E_score", "S_score", "G_score", "ESG_composite"
    """
    df = df.copy()
    pillar_weights = pillar_weights or {"E": 1/3, "S": 1/3, "G": 1/3}

    # ------------------------------------------------------------------
    # Determine whether sector-materiality weighting is active
    # ------------------------------------------------------------------
    use_materiality = (
        sector_column is not None
        and sector_column in df.columns
        and materiality_map is not None
        and len(materiality_map) > 0
    )
    if use_materiality:
        default_mw = materiality_map.get("default", pillar_weights)  # type: ignore[union-attr]
        logger.info(
            "Sector-materiality weighting active (%d sectors configured)",
            len(materiality_map) - 1,  # type: ignore[arg-type]
        )

    for pillar, cat_weights in pillar_config.items():
        score_col = f"{pillar}_score"
        coverage_col = f"{pillar}_coverage"
        confidence_col = f"{pillar}_confidence"

        weighted_score = pd.Series(0.0, index=df.index, dtype=float)
        effective_weight = pd.Series(0.0, index=df.index, dtype=float)

        total_weight = sum(cat_weights.values())
        if total_weight == 0:
            warnings.warn(f"No weights defined for pillar {pillar}, skipping", UserWarning)
            df[score_col] = np.nan
            df[coverage_col] = 0.0
            df[confidence_col] = 0.0
            continue

        for category, weight in cat_weights.items():
            # Look up the explicit indicator list for this category,
            # then check for the corresponding "_norm" columns in the df.
            known_indicators = CATEGORY_INDICATOR_MAP.get(category, [])

            matching_cols: list[str] = []

            # 1) Exact match: "{category}_norm" (covers short/test column names)
            if f"{category}_norm" in df.columns:
                matching_cols.append(f"{category}_norm")

            # 2) Mapped indicators: "{indicator}_norm"
            for indicator in known_indicators:
                norm_col = f"{indicator}_norm"
                if norm_col in df.columns and norm_col not in matching_cols:
                    matching_cols.append(norm_col)
            
            if not matching_cols:
                warnings.warn(f"No indicator found for category '{category}' in pillar {pillar}", UserWarning)
                continue

            # Filter out zero-variance (constant) indicators — they add
            # noise without discriminating power (e.g., binary columns
            # that are all-True after imputation).
            informative_cols = [
                c for c in matching_cols
                if df[c].notna().sum() > 1 and df[c].dropna().nunique() > 1
            ]
            if not informative_cols:
                logger.info(
                    "Skipping category '%s' in pillar %s: all %d indicator(s) "
                    "are constant (zero variance)",
                    category, pillar, len(matching_cols),
                )
                continue

            # Use first matching column (or average if multiple)
            if len(informative_cols) == 1:
                col_data = df[informative_cols[0]]
            else:
                col_data = df[informative_cols].mean(axis=1)

            # Down-weight indicators that have zero real calibration data.
            # These are entirely proxy-constructed and should contribute
            # less to the pillar score to reduce synthetic noise.
            from src.constants import ZERO_CALIBRATION_PROXIES
            zero_cal_count = sum(
                1 for c in informative_cols
                if c.replace("_norm", "") in ZERO_CALIBRATION_PROXIES
            )
            calibration_penalty = 1.0
            if zero_cal_count > 0:
                # If ALL indicators in this category are zero-calibration,
                # apply 0.3x weight; if mixed, apply proportional penalty
                frac_uncalibrated = zero_cal_count / len(informative_cols)
                calibration_penalty = 1.0 - 0.7 * frac_uncalibrated

            effective_cat_weight = weight * calibration_penalty
            available_mask = col_data.notna()
            weighted_score.loc[available_mask] += effective_cat_weight * col_data.loc[available_mask]
            effective_weight.loc[available_mask] += effective_cat_weight

        df[score_col] = weighted_score.div(effective_weight.where(effective_weight > 0))
        df[coverage_col] = effective_weight / total_weight
        df[confidence_col] = df[coverage_col].clip(0.0, 1.0)

    df = _harmonize_pillar_by_source(df, "E_score", source_col="data_source")
    df = _harmonize_pillar_by_source(df, "S_score", source_col="data_source")
    df = _harmonize_pillar_by_source(df, "G_score", source_col="data_source")

    # ------------------------------------------------------------------
    # Decouple E_score from financial_score and apply sector-relative
    # normalization for Environmental score.
    # ------------------------------------------------------------------
    if "E_score" in df.columns and "financial_score" in df.columns:
        valid = df["E_score"].notna() & df["financial_score"].notna()
        if valid.sum() >= 2:
            fin_var = df.loc[valid, "financial_score"].var()
            if not pd.isna(fin_var) and fin_var > 1e-12:
                beta = df.loc[valid, "E_score"].cov(df.loc[valid, "financial_score"]) / fin_var
                df.loc[valid, "E_score"] = (
                    df.loc[valid, "E_score"] - beta * df.loc[valid, "financial_score"]
                )

    if "E_score" in df.columns:
        if sector_column is not None and sector_column in df.columns:
            e_sector_rank = df.groupby(sector_column)["E_score"].rank(pct=True)
        else:
            e_sector_rank = df["E_score"].rank(pct=True)
        e_rank_std = e_sector_rank.std()
        if pd.isna(e_rank_std) or e_rank_std <= 1e-12:
            df["E_score"] = 0.0
        else:
            df["E_score"] = (e_sector_rank - e_sector_rank.mean()) / (e_rank_std + 1e-10)

    # ------------------------------------------------------------------
    # Compute composite ESG score
    # ------------------------------------------------------------------
    use_empirical_weights = pillar_weighting_method == "empirical"
    empirical_weights: dict[str, float] | None = None
    if use_empirical_weights and "financial_score" in df.columns:
        inv_corr: dict[str, float] = {}
        for pillar_key in ["E", "S", "G"]:
            score_col = f"{pillar_key}_score"
            if score_col not in df.columns:
                continue
            valid = df[score_col].notna() & df["financial_score"].notna()
            if valid.sum() < 3:
                continue
            corr = df.loc[valid, score_col].corr(df.loc[valid, "financial_score"])
            if pd.isna(corr):
                continue
            inv_corr[pillar_key] = 1.0 / (abs(corr) + 1e-6)

        inv_sum = sum(inv_corr.values())
        if inv_sum > 0:
            empirical_weights = {k: v / inv_sum for k, v in inv_corr.items()}

    stable_empirical = empirical_weights is not None and abs(sum(empirical_weights.values()) - 1.0) < 1e-6

    if stable_empirical:
        w_e = pd.Series(empirical_weights.get("E", 0.0), index=df.index, dtype=float)
        w_s = pd.Series(empirical_weights.get("S", 0.0), index=df.index, dtype=float)
        w_g = pd.Series(empirical_weights.get("G", 0.0), index=df.index, dtype=float)
        w_total = w_e + w_s + w_g

        esg_weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
        esg_effective_weight = pd.Series(0.0, index=df.index, dtype=float)
        for pillar_key, w_series in [("E", w_e), ("S", w_s), ("G", w_g)]:
            score_col = f"{pillar_key}_score"
            if score_col in df.columns:
                available_mask = df[score_col].notna()
                esg_weighted_sum.loc[available_mask] += (
                    w_series.loc[available_mask] * df.loc[available_mask, score_col]
                )
                esg_effective_weight.loc[available_mask] += w_series.loc[available_mask]

        esg_raw = esg_weighted_sum.div(esg_effective_weight.where(esg_effective_weight > 0))
        df["ESG_confidence"] = esg_effective_weight.div(w_total.where(w_total > 0)).clip(0.0, 1.0)
        df["materiality_E_weight"] = w_e
        df["materiality_S_weight"] = w_s
        df["materiality_G_weight"] = w_g
    elif use_materiality:
        # Per-company pillar weights based on sector materiality
        sectors = df[sector_column].fillna("")  # type: ignore[arg-type]
        w_e = sectors.map(lambda s: materiality_map.get(s, default_mw).get("E", default_mw["E"]))  # type: ignore[union-attr]
        w_s = sectors.map(lambda s: materiality_map.get(s, default_mw).get("S", default_mw["S"]))  # type: ignore[union-attr]
        w_g = sectors.map(lambda s: materiality_map.get(s, default_mw).get("G", default_mw["G"]))  # type: ignore[union-attr]
        w_total = w_e + w_s + w_g

        esg_weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
        esg_effective_weight = pd.Series(0.0, index=df.index, dtype=float)
        for pillar_key, w_series in [("E", w_e), ("S", w_s), ("G", w_g)]:
            score_col = f"{pillar_key}_score"
            if score_col in df.columns:
                available_mask = df[score_col].notna()
                esg_weighted_sum.loc[available_mask] += (
                    w_series.loc[available_mask] * df.loc[available_mask, score_col]
                )
                esg_effective_weight.loc[available_mask] += w_series.loc[available_mask]

        esg_raw = esg_weighted_sum.div(esg_effective_weight.where(esg_effective_weight > 0))
        df["ESG_confidence"] = esg_effective_weight.div(w_total.where(w_total > 0)).clip(0.0, 1.0)

        # Store which materiality weights were applied (useful for auditing)
        df["materiality_E_weight"] = w_e
        df["materiality_S_weight"] = w_s
        df["materiality_G_weight"] = w_g
    else:
        total_pillar_weight = sum(pillar_weights.values())
        esg_weighted_sum = pd.Series(0.0, index=df.index, dtype=float)
        esg_effective_weight = pd.Series(0.0, index=df.index, dtype=float)
        for pillar, weight in pillar_weights.items():
            score_col = f"{pillar}_score"
            if score_col in df.columns:
                available_mask = df[score_col].notna()
                esg_weighted_sum.loc[available_mask] += weight * df.loc[available_mask, score_col]
                esg_effective_weight.loc[available_mask] += weight

        esg_raw = esg_weighted_sum.div(esg_effective_weight.where(esg_effective_weight > 0))
        df["ESG_confidence"] = esg_effective_weight.div(total_pillar_weight).clip(0.0, 1.0)

    if "data_source" in df.columns:
        source = df["data_source"].astype(str).str.lower()
        provenance_weight = pd.Series(1.0, index=df.index, dtype=float)
        # Stronger penalty: proxy-only data gets 0.5 (was 0.8), imputed gets 0.4
        provenance_weight = provenance_weight.where(
            ~source.str.contains("proxy|calibrated", regex=True), 0.5
        )
        provenance_weight = provenance_weight.where(
            ~source.str.contains("imput", regex=True), 0.4
        )
        # Real data sources retain full weight
        provenance_weight = provenance_weight.where(
            ~source.str.contains("epa|sec|real|yahoo|sbti|cdp|ungc", regex=True), 1.0
        )
        df["ESG_confidence"] = (df["ESG_confidence"] * provenance_weight).clip(0.0, 1.0)
        df["ESG_provenance_weight"] = provenance_weight

    if sector_column is not None and sector_column in df.columns:
        sector_series = df[sector_column]
    else:
        sector_series = pd.Series("__all__", index=df.index)
    df["ESG_composite"] = shrink_low_confidence_scores(
        esg_raw,
        sector_series,
        df["ESG_confidence"],
        min_confidence=min_confidence,
    )

    # Scale to 0-100 for interpretability (when scale_to_score is True)
    if scale_to_score:
        for col in ["E_score", "S_score", "G_score", "ESG_composite"]:
            if col in df.columns:
                # Rescale from z-score-like to 0-100 (assuming roughly normal distribution)
                df[col] = 50 + (df[col] * 20)  # Center at 50, scale by 20
                df[col] = df[col].clip(0, 100)

    return df


class CompositeIndexBuilder:
    """Build composite ESG index from raw indicators."""

    def __init__(self, config: dict[str, Any]):
        """Initialize with index configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Index configuration dict (from index_config.yaml)
        """
        self.config = config
        self.esg_cfg = config.get("esg_index", {})

    def build(
        self,
        df: pd.DataFrame,
        indicator_cols: list[str],
        *,
        id_col: str = "ticker",
        sector_column: str | None = "sector",
        materiality_path: str | None = None,
        scale_to_score: bool = True,
    ) -> pd.DataFrame:
        """Build composite ESG index.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with indicator columns
        indicator_cols : list[str]
            List of indicator column names to include
        id_col : str, default "ticker"
            Identifier column name
        sector_column : str | None, default "sector"
            Column containing sector labels.  When provided (and a materiality
            config is found), sector-specific E/S/G pillar weights from
            ``config/sasb_materiality.yaml`` are applied.  Pass ``None`` to
            disable materiality weighting.
        materiality_path : str | None, optional
            Explicit path to SASB materiality YAML.  When ``None`` the default
            ``config/sasb_materiality.yaml`` is auto-discovered.
        scale_to_score : bool, default True
            When True, applies ``50 + z * 20`` transformation and clips to
            [0, 100].  When False, returns raw z-score composites.

        Returns
        -------
        pd.DataFrame
            DataFrame with added normalized indicators, pillar scores, and composite score
        """
        # Validate indicator_cols exist in dataframe
        missing_cols = [c for c in indicator_cols if c not in df.columns]
        if missing_cols:
            logger.warning(
                "indicator_cols not found in dataframe (will be skipped): %s",
                missing_cols,
            )
            indicator_cols = [c for c in indicator_cols if c in df.columns]
        if not indicator_cols:
            raise ValueError(
                "None of the requested indicator_cols exist in the dataframe"
            )

        # Step 1: Normalize indicators
        norm_cfg = self.esg_cfg.get("normalization", {})
        df = normalize_indicators(
            df,
            indicator_cols,
            method=norm_cfg.get("method", "zscore"),
            by_group=norm_cfg.get("by_group", {}),
            winsorize=norm_cfg.get("winsorize", {}),
            orthogonalize_against=norm_cfg.get("orthogonalize_against"),
        )

        # Step 2: Compute pillar scores
        category_weights_raw = self.esg_cfg.get("category_weights", {})
        pillar_weights_raw = self.esg_cfg.get("pillar_weights", {})
        
        # Extract default weights from weight ranges (if they're dicts with min/max/default)
        def extract_default_weight(weight_value):
            """Extract default weight from weight range dict or return value directly."""
            if isinstance(weight_value, dict):
                return weight_value.get("default", weight_value.get("max", 0.0))
            return weight_value
        
        # Process category weights
        category_weights = {}
        for pillar, cat_dict in category_weights_raw.items():
            if isinstance(cat_dict, dict):
                category_weights[pillar] = {
                    cat: extract_default_weight(weight_val)
                    for cat, weight_val in cat_dict.items()
                    if cat != "constraint"  # Skip constraint keys
                }
            else:
                category_weights[pillar] = cat_dict
        
        # Process pillar weights
        pillar_weights = {}
        for pillar, weight_val in pillar_weights_raw.items():
            if pillar != "constraint":  # Skip constraint keys
                pillar_weights[pillar] = extract_default_weight(weight_val)
        
        # Load sector-materiality weights (if sector column is available)
        materiality_map = None
        if sector_column and sector_column in df.columns:
            materiality_map = _load_materiality_map(materiality_path)

        df = compute_pillar_scores(
            df,
            category_weights,
            pillar_weights=pillar_weights,
            sector_column=sector_column,
            materiality_map=materiality_map,
            scale_to_score=scale_to_score,
            pillar_weighting_method=self.esg_cfg.get("pillar_weighting_method", "fixed"),
        )

        return df
