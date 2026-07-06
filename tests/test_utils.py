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

    def test_creates_new_directory(self, tmp_path):
        """Creates a directory that doesn't exist."""
        target = tmp_path / "new_dir" / "sub_dir"
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_existing_directory_no_error(self, tmp_path):
        """Does not raise for already-existing directory."""
        target = tmp_path / "existing"
        target.mkdir()
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_returns_path_object(self, tmp_path):
        """Returns a Path object."""
        result = ensure_dir(tmp_path / "test")
        assert isinstance(result, Path)

    def test_accepts_string(self, tmp_path):
        """Accepts string paths."""
        target = str(tmp_path / "str_dir")
        result = ensure_dir(target)
        assert Path(target).is_dir()
        assert isinstance(result, Path)


class TestLoadIndexedData:
    """Tests for load_indexed_data()."""

    def test_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when indexed_data.csv doesn't exist."""
        with pytest.raises(FileNotFoundError, match="indexed_data.csv not found"):
            load_indexed_data(project_root=tmp_path)

    def test_loads_existing_csv(self, tmp_path):
        """Loads a valid CSV file."""
        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "indexed_data.csv"
        csv_path.write_text("ticker,score\nA,50\nB,60\n", encoding="utf-8")

        df = load_indexed_data(project_root=tmp_path)
        assert len(df) == 2
        assert "ticker" in df.columns
        assert "score" in df.columns

    def test_returns_dataframe(self, tmp_path):
        """Return type is pd.DataFrame."""
        import pandas as pd

        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)
        csv_path = data_dir / "indexed_data.csv"
        csv_path.write_text("ticker\nA\n", encoding="utf-8")

        result = load_indexed_data(project_root=tmp_path)
        assert isinstance(result, pd.DataFrame)
