"""Shared utility functions for the multi-factor ESG pipeline.

Provides common helpers used across multiple scripts to eliminate
boilerplate and code duplication.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

import numpy as np
import pandas as pd


def robust_zscore(
    series: pd.Series,
    clip_bounds: tuple[float, float] = (-3, 3),
) -> pd.Series:
    """Compute a robust z-score using Median Absolute Deviation (MAD).

    Formula: ``(x - median) / (MAD * 1.4826)`` where 1.4826 is the
    consistency constant that makes MAD an unbiased estimator of the
    standard deviation for normally distributed data (Iglewicz & Hoaglin,
    1993).

    Falls back to the standard z-score ``(x - mean) / std`` when the MAD
    is near zero (constant or near-constant data).

    Parameters
    ----------
    series : pd.Series
        Input numeric series.
    clip_bounds : tuple[float, float], default (-3, 3)
        Lower and upper bounds for clipping the resulting z-scores.

    Returns
    -------
    pd.Series
        Robust z-scored (and clipped) series, preserving the original index.
    """
    median = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - median)) * 1.4826

    if mad < 1e-10:
        # MAD ≈ 0 → fall back to standard z-score
        std = np.nanstd(series, ddof=0)
        if std < 1e-10:
            return pd.Series(0.0, index=series.index)
        z = (series - np.nanmean(series)) / std
    else:
        z = (series - median) / mad

    return z.clip(lower=clip_bounds[0], upper=clip_bounds[1])


# Mapping from config profile keys to DataFrame column names.
# Mirrors SCORE_COLUMN_MAP in src/similarity/preference_scoring.py.
_CONFIG_TO_COLUMN = {
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


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Works whether called from ``src/`` or ``scripts/``.
    """
    # Walk up from this file (src/utils.py) to the project root
    return Path(__file__).resolve().parent.parent


def setup_paths() -> Path:
    """Add the project root to ``sys.path`` and ``os.chdir`` there.

    Returns the project root path for convenience.  This replaces the
    three-line boilerplate at the top of every script::

        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(PROJECT_ROOT))
        os.chdir(PROJECT_ROOT)
    """
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    return project_root


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it does not exist, then return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_indexed_data(
    project_root: Path | None = None,
    *,
    include_benchmarks: bool = False,
) -> pd.DataFrame:
    """Load ``data/processed/indexed_data.csv`` with standard error handling.

    Parameters
    ----------
    project_root : Path, optional
        Project root directory.  Detected automatically if *None*.
    include_benchmarks : bool, default False
        If *False* (default), rows with ``is_large_cap_benchmark == True``
        are dropped so that downstream analysis operates on the mid-cap
        universe only.  Set to *True* for robustness testing that explicitly
        needs the large-cap comparison companies (AAPL, MSFT, GOOGL, AMZN).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the indexed-data file cannot be found.
    """
    if project_root is None:
        project_root = get_project_root()
    path = project_root / "data" / "processed" / "indexed_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"indexed_data.csv not found at {path}.\n"
            "Run: python scripts/03_build_index.py"
        )
    df = pd.read_csv(path)

    # Filter out large-cap benchmarks unless explicitly requested
    if not include_benchmarks and "is_large_cap_benchmark" in df.columns:
        n_before = len(df)
        df = df[~df["is_large_cap_benchmark"].fillna(False).astype(bool)].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            import logging
            logging.getLogger(__name__).info(
                "Excluded %d large-cap benchmarks from analysis (mid-cap focus). "
                "Use include_benchmarks=True for robustness testing.",
                n_dropped,
            )

    return df


def load_profile_weights(
    profile_name: str = "balanced",
    *,
    project_root: Path | None = None,
    as_column_names: bool = True,
) -> dict[str, float]:
    """Load investor-profile weights from ``config/index_config.yaml``.

    Parameters
    ----------
    profile_name : str
        Profile key under ``preference_scoring.investor_profiles``
        (e.g. ``"balanced"``, ``"esg_first"``, ``"financial_first"``).
    project_root : Path, optional
        Project root directory.  Detected automatically if *None*.
    as_column_names : bool, default True
        If *True*, translate config keys (e.g. ``"esg_score"``) to the
        corresponding DataFrame column names (e.g. ``"ESG_composite"``)
        using :data:`_CONFIG_TO_COLUMN`.  If *False*, return config keys
        as-is.

    Returns
    -------
    dict[str, float]
        Mapping of factor (column) name to weight for all 10 factors.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found.
    KeyError
        If the requested profile does not exist in the config.
    """
    if project_root is None:
        project_root = get_project_root()
    cfg_path = project_root / "config" / "index_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    profiles = (
        cfg.get("preference_scoring", {}).get("investor_profiles", {})
    )
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise KeyError(
            f"Profile '{profile_name}' not found in config. "
            f"Available profiles: {available}"
        )

    raw_weights: dict[str, float] = profiles[profile_name]

    if as_column_names:
        return {
            _CONFIG_TO_COLUMN.get(k, k): v
            for k, v in raw_weights.items()
        }
    return dict(raw_weights)
