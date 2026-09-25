"""Tests for src/utils.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils import ensure_dir, get_project_root, load_indexed_data


class TestGetProjectRoot:
    """Tests for get_project_root()."""

    def test_returns_path(self):
        """Returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_src(self):
        """Project root contains a src/ directory."""
        root = get_project_root()
        assert (root / "src").is_dir()

    def test_root_contains_config(self):
        """Project root contains a config/ directory."""
        root = get_project_root()
        assert (root / "config").is_dir()

    def test_root_is_absolute(self):
        """Returned path is absolute."""
        root = get_project_root()
        assert root.is_absolute()


class TestEnsureDir:
    """Tests for ensure_dir()."""

    def test_creates_new_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "new_dir" / "sub_dir"
            result = ensure_dir(target)
            assert target.is_dir()
            assert result == target

    def test_existing_directory_no_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "existing"
            target.mkdir()
            result = ensure_dir(target)
            assert target.is_dir()
            assert result == target

    def test_returns_path_object(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            result = ensure_dir(Path(td) / "test")
            assert isinstance(result, Path)

    def test_accepts_string(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = str(Path(td) / "str_dir")
            result = ensure_dir(target)
            assert Path(target).is_dir()
            assert isinstance(result, Path)


class TestLoadIndexedData:
    """Tests for load_indexed_data()."""

    def test_raises_file_not_found(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError, match="indexed_data.csv not found"):
                load_indexed_data(project_root=Path(td))

    def test_loads_existing_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data" / "processed"
            data_dir.mkdir(parents=True)
            csv_path = data_dir / "indexed_data.csv"
            csv_path.write_text("ticker,score\nA,50\nB,60\n", encoding="utf-8")
            df = load_indexed_data(project_root=Path(td))
            assert len(df) == 2
            assert "ticker" in df.columns
            assert "score" in df.columns

    def test_returns_dataframe(self):
        import tempfile
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "data" / "processed"
            data_dir.mkdir(parents=True)
            csv_path = data_dir / "indexed_data.csv"
            csv_path.write_text("ticker\nA\n", encoding="utf-8")
            result = load_indexed_data(project_root=Path(td))
            assert isinstance(result, pd.DataFrame)


class TestPortfolioTopN:
    """Dynamic top-N rule: max(10, min(50, ceil(0.07 N)))."""

    def test_documented_values(self):
        from src.utils import get_portfolio_top_n
        assert get_portfolio_top_n(276) == 20
        assert get_portfolio_top_n(100) == 10
        assert get_portfolio_top_n(500) == 35
        assert get_portfolio_top_n(1000) == 50

    def test_never_exceeds_universe(self):
        from src.utils import get_portfolio_top_n
        assert get_portfolio_top_n(5) == 5
