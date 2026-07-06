"""Tests for src/similarity/cosine_similarity.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.similarity.cosine_similarity import (
    compute_similarity_matrix,
    cosine_similarity,
    rank_by_similarity,
)


# ── cosine_similarity (function) ──────────────────────────────────────────

class TestCosineSimilarity:
    """Tests for the low-level cosine_similarity() function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity ≈ 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity ≈ 0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_zero_vector(self):
        """Zero vector returns 0.0 similarity."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_negative_vectors(self):
        """Opposite vectors have similarity ≈ -1.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)

    def test_non_negative_vectors_in_unit_range(self):
        """Non-negative vectors produce similarity in [0, 1]."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        sim = cosine_similarity(v1, v2)
        assert 0.0 <= sim <= 1.0


# ── compute_similarity_matrix ─────────────────────────────────────────────

class TestComputeSimilarityMatrix:
    """Tests for compute_similarity_matrix()."""

    @pytest.fixture
    def feature_df(self) -> pd.DataFrame:
        """Small DataFrame for similarity tests."""
        return pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feat2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "feat3": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_output_is_square(self, feature_df):
        """Similarity matrix is n x n."""
        sim = compute_similarity_matrix(feature_df, ["feat1", "feat2", "feat3"])
        assert sim.shape == (5, 5)

    def test_output_is_symmetric(self, feature_df):
        """sim(i,j) == sim(j,i)."""
        sim = compute_similarity_matrix(feature_df, ["feat1", "feat2", "feat3"])
        np.testing.assert_array_almost_equal(sim.values, sim.values.T)

    def test_diagonal_is_one(self, feature_df):
        """Self-similarity is 1.0."""
        sim = compute_similarity_matrix(feature_df, ["feat1", "feat2", "feat3"])
        diag = np.diag(sim.values)
        np.testing.assert_array_almost_equal(diag, np.ones(5), decimal=5)

    def test_values_in_range(self, feature_df):
        """All values in [-1, 1] for cosine metric."""
        sim = compute_similarity_matrix(feature_df, ["feat1", "feat2", "feat3"])
        assert sim.values.min() >= -1.01
        assert sim.values.max() <= 1.01

    def test_index_and_columns_match_tickers(self, feature_df):
        """Index and columns of result match ticker column."""
        sim = compute_similarity_matrix(feature_df, ["feat1", "feat2"])
        assert list(sim.index) == ["A", "B", "C", "D", "E"]
        assert list(sim.columns) == ["A", "B", "C", "D", "E"]

    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "jaccard"])
    def test_all_metrics(self, feature_df, metric):
        """All supported metrics produce valid square matrices."""
        sim = compute_similarity_matrix(feature_df, ["feat1", "feat2"], metric=metric)
        assert sim.shape == (5, 5)

    def test_unknown_metric_raises(self, feature_df):
        """Unknown metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            compute_similarity_matrix(feature_df, ["feat1"], metric="manhattan")

    def test_no_feature_cols_raises(self, feature_df):
        """No valid feature columns raises ValueError."""
        with pytest.raises(ValueError, match="No feature columns"):
            compute_similarity_matrix(feature_df, ["nonexistent_a", "nonexistent_b"])

    def test_feature_weights(self, feature_df):
        """Feature weights change the similarity matrix."""
        sim1 = compute_similarity_matrix(feature_df, ["feat1", "feat2"])
        sim2 = compute_similarity_matrix(
            feature_df, ["feat1", "feat2"],
            feature_weights={"feat1": 10.0, "feat2": 0.1},
        )
        # Weighted and unweighted should generally differ
        assert not np.allclose(sim1.values, sim2.values)

    def test_single_company(self):
        """Single company produces 1x1 matrix with value 1.0."""
        df = pd.DataFrame({"ticker": ["X"], "f": [3.0]})
        sim = compute_similarity_matrix(df, ["f"])
        assert sim.shape == (1, 1)
        assert sim.iloc[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_identical_companies(self):
        """Identical feature vectors should yield similarity 1.0."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "f1": [1.0, 1.0],
                "f2": [2.0, 2.0],
            }
        )
        sim = compute_similarity_matrix(df, ["f1", "f2"])
        assert sim.loc["A", "B"] == pytest.approx(1.0, abs=1e-5)

    def test_jaccard_vectorized(self):
        """Jaccard similarity is computed correctly (vectorized)."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "f1": [1, 1, 0],
                "f2": [1, 0, 1],
                "f3": [0, 1, 1],
            }
        )
        sim = compute_similarity_matrix(df, ["f1", "f2", "f3"], metric="jaccard")
        # Jaccard(A, B) = |{f1}| / |{f1, f2, f3}| = 1/3
        assert sim.loc["A", "B"] == pytest.approx(1 / 3, abs=0.05)

    def test_missing_feature_cols_warns(self, feature_df):
        """Warns when some feature columns are missing."""
        import warnings as _w
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            compute_similarity_matrix(feature_df, ["feat1", "nonexistent"])
            assert any("Missing feature columns" in str(x.message) for x in w)


# ── rank_by_similarity ────────────────────────────────────────────────────

class TestRankBySimilarity:
    """Tests for rank_by_similarity()."""

    @pytest.fixture
    def sim_matrix(self) -> pd.DataFrame:
        """Pre-computed similarity matrix."""
        ids = ["A", "B", "C", "D"]
        data = np.array(
            [
                [1.0, 0.9, 0.5, 0.1],
                [0.9, 1.0, 0.6, 0.2],
                [0.5, 0.6, 1.0, 0.8],
                [0.1, 0.2, 0.8, 1.0],
            ]
        )
        return pd.DataFrame(data, index=ids, columns=ids)

    def test_returns_series(self, sim_matrix):
        """Output is a pandas Series."""
        result = rank_by_similarity(sim_matrix, "A")
        assert isinstance(result, pd.Series)

    def test_excludes_self(self, sim_matrix):
        """Target company is excluded from results."""
        result = rank_by_similarity(sim_matrix, "A", exclude_self=True)
        assert "A" not in result.index

    def test_includes_self_when_requested(self, sim_matrix):
        """Target company is included when exclude_self=False."""
        result = rank_by_similarity(sim_matrix, "A", exclude_self=False)
        assert "A" in result.index

    def test_descending_order(self, sim_matrix):
        """Results are sorted in descending order."""
        result = rank_by_similarity(sim_matrix, "A")
        values = result.values
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_top_n(self, sim_matrix):
        """top_n limits the number of results."""
        result = rank_by_similarity(sim_matrix, "A", top_n=2)
        assert len(result) == 2

    def test_unknown_target_raises(self, sim_matrix):
        """Unknown target ID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            rank_by_similarity(sim_matrix, "UNKNOWN")

    def test_most_similar_to_A_is_B(self, sim_matrix):
        """For the given matrix, B is most similar to A."""
        result = rank_by_similarity(sim_matrix, "A", top_n=1)
        assert result.index[0] == "B"
