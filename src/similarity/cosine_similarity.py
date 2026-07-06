"""Cosine Similarity Implementation for ESG Profiles.

CIRCULARITY FIX (Issue M3)
--------------------------
Previously, similarity was computed on ALL 8 factor scores (ESG, financial,
market, operational, risk_adjusted, value, growth, stability), then included
as a 9th factor in the preference score.  This created circularity: the
similarity score depended on the other composite scores, and the final
preference depended on the similarity score.

The fix restricts similarity computation to ESG pillar scores (E_score,
S_score, G_score) by default.  These are independent inputs derived from
raw ESG indicators, not from any composite that includes similarity itself.
This breaks the circular dependency and redefines the metric as "ESG peer
similarity" — how closely a company's ESG profile matches its peers —
rather than "multi-factor conformity".
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default columns for similarity computation: ESG pillar scores only.
# Using these breaks the circularity that arises from computing similarity
# on the same composite scores that later include similarity as a factor.
DEFAULT_ESG_FEATURE_SUBSET = ["E_score", "S_score", "G_score"]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    vec_a : np.ndarray
        First vector
    vec_b : np.ndarray
        Second vector
    eps : float, default 1e-12
        Small epsilon to avoid division by zero

    Returns
    -------
    float
        Cosine similarity in [0, 1] range (for non-negative vectors)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < eps or norm_b < eps:
        return 0.0

    dot_product = np.dot(vec_a, vec_b)
    return dot_product / (norm_a * norm_b)


def compute_similarity_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    id_col: str = "ticker",
    metric: str = "cosine",
    feature_weights: dict[str, float] | None = None,
    output_scale: dict[str, float] | None = None,
    feature_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Compute pairwise similarity matrix for companies.

    CIRCULARITY NOTE (Issue M3)
    ---------------------------
    When this function is used to produce a similarity_rank that will be
    included as a factor in a composite preference score, the ``feature_cols``
    (or ``feature_subset``) MUST NOT include scores that themselves depend on
    similarity_rank.  The default ``feature_subset`` uses only the three ESG
    pillar scores (E_score, S_score, G_score), which are independent inputs
    and therefore break the circular dependency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns and id_col
    feature_cols : list[str] | None
        Column names to use as feature vector.  If *None*, falls back to
        ``feature_subset`` (which defaults to ESG pillar scores).
    id_col : str, default "ticker"
        Identifier column name
    metric : str, default "cosine"
        Similarity metric: "cosine", "euclidean", "jaccard"
    feature_weights : dict[str, float] | None, optional
        Optional weights for each feature column
    output_scale : dict[str, float] | None, optional
        Scale output to [min, max] range (default: [0, 1])
    feature_subset : list[str] | None, optional
        Explicit subset of columns to use.  Takes effect only when
        ``feature_cols`` is *None*.  Defaults to
        ``DEFAULT_ESG_FEATURE_SUBSET`` (E_score, S_score, G_score).

    Returns
    -------
    pd.DataFrame
        Similarity matrix with id_col as index and columns
    """
    # Resolve which columns to use: explicit feature_cols > feature_subset > default
    if feature_cols is None:
        feature_cols = feature_subset or DEFAULT_ESG_FEATURE_SUBSET
    output_scale = output_scale or {"min": 0.0, "max": 1.0}

    # Extract feature vectors
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        warnings.warn(f"Missing feature columns: {missing}", UserWarning)

    if not available_cols:
        raise ValueError("No feature columns available")

    X = df[available_cols].fillna(df[available_cols].mean()).values
    ids = df[id_col].values

    # Apply feature weights if provided
    if feature_weights:
        weights = np.array([feature_weights.get(col, 1.0) for col in available_cols])
        X = X * weights[np.newaxis, :]

    # Compute similarity matrix
    n = len(ids)
    sim_matrix = np.zeros((n, n))

    if metric == "cosine":
        # Normalize vectors
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-12)
        sim_matrix = X_norm @ X_norm.T
    elif metric == "euclidean":
        # Convert to similarity (1 / (1 + distance))
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(X, metric="euclidean"))
        sim_matrix = 1 / (1 + distances)
    elif metric == "jaccard":
        # Vectorized Jaccard similarity using matrix operations
        # For binary/categorical features
        X_binary = (X > 0).astype(float)
        # Intersection: X_binary @ X_binary.T gives element-wise AND count
        intersection = X_binary @ X_binary.T
        # Row sums for union computation
        row_sums = X_binary.sum(axis=1)
        # Union: |A| + |B| - |A ∩ B|
        union = row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - intersection
        sim_matrix = np.where(union > 0, intersection / union, 0.0)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    # Scale to output range
    if output_scale["min"] != 0.0 or output_scale["max"] != 1.0:
        min_val = sim_matrix.min()
        max_val = sim_matrix.max()
        if max_val > min_val:
            sim_matrix = (sim_matrix - min_val) / (max_val - min_val)
            sim_matrix = sim_matrix * (output_scale["max"] - output_scale["min"]) + output_scale["min"]

    # Create DataFrame
    result = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    return result


def rank_by_similarity(
    similarity_matrix: pd.DataFrame,
    target_id: str,
    *,
    top_n: int = 10,
    exclude_self: bool = True,
) -> pd.Series:
    """Rank companies by similarity to a target company.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Pairwise similarity matrix
    target_id : str
        Target company identifier
    top_n : int, default 10
        Number of top similar companies to return
    exclude_self : bool, default True
        Exclude the target company from results

    Returns
    -------
    pd.Series
        Ranked similarity scores (descending order)
    """
    if target_id not in similarity_matrix.index:
        raise ValueError(f"Target ID '{target_id}' not found in similarity matrix")

    similarities = similarity_matrix.loc[target_id].sort_values(ascending=False)

    if exclude_self:
        similarities = similarities.drop(target_id)

    return similarities.head(top_n)
