"""Tests for src/data_collection/data_pipeline.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_collection.data_pipeline import (
    PipelinePaths,
    load_configs,
    load_yaml,
    standardize_and_clean,
)


# ── load_yaml ─────────────────────────────────────────────────────────────

class TestLoadYaml:
    """Tests for load_yaml()."""

    def test_loads_valid_yaml(self, tmp_path):
        """Correctly parses a YAML file."""
        cfg = {"key": "value", "nested": {"a": 1}}
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(cfg), encoding="utf-8")
        result = load_yaml(path)
        assert result == cfg

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file returns {}."""
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        result = load_yaml(path)
        assert result == {}

    def test_missing_file_raises(self, tmp_path):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "does_not_exist.yaml")


# ── load_configs ──────────────────────────────────────────────────────────

class TestLoadConfigs:
    """Tests for load_configs()."""

    def test_loads_from_explicit_paths(self, tmp_path):
        """Returns dicts when given explicit paths."""
        idx_cfg = {"esg_index": {"normalization": {"method": "zscore"}}}
        ds_cfg = {"data_sources": {"public": {}}}
        idx_path = tmp_path / "idx.yaml"
        ds_path = tmp_path / "ds.yaml"
        idx_path.write_text(yaml.dump(idx_cfg), encoding="utf-8")
        ds_path.write_text(yaml.dump(ds_cfg), encoding="utf-8")

        index_config, data_sources = load_configs(
            index_config_path=idx_path, data_sources_path=ds_path
        )
        assert "esg_index" in index_config
        assert "data_sources" in data_sources

    def test_loads_from_project_config_dir(self, config_dir):
        """Loads configs from the project's actual config/ directory if it exists."""
        idx_path = config_dir / "index_config.yaml"
        ds_path = config_dir / "data_sources.yaml"
        if idx_path.exists() and ds_path.exists():
            index_config, data_sources = load_configs(
                index_config_path=idx_path, data_sources_path=ds_path
            )
            assert isinstance(index_config, dict)
            assert isinstance(data_sources, dict)
        else:
            pytest.skip("Config files not found in project")


# ── PipelinePaths ─────────────────────────────────────────────────────────

class TestPipelinePaths:
    """Tests for PipelinePaths dataclass."""

    def test_default_paths(self):
        """Default paths are set correctly."""
        pp = PipelinePaths()
        assert pp.raw_dir == Path("data/raw")
        assert pp.processed_dir == Path("data/processed")
        assert pp.external_dir == Path("data/external")

    def test_frozen(self):
        """PipelinePaths is immutable (frozen dataclass)."""
        pp = PipelinePaths()
        with pytest.raises(AttributeError):
            pp.raw_dir = Path("/tmp")


# ── standardize_and_clean ─────────────────────────────────────────────────

class TestStandardizeAndClean:
    """Tests for standardize_and_clean()."""

    def test_returns_cleaned_df_and_diagnostics(self, df_with_nans, index_config):
        """Returns a tuple of (DataFrame, dict) with expected keys."""
        out_df, diag = standardize_and_clean(
            df_with_nans, index_cfg=index_config, required_cols=["x", "y", "z"]
        )
        assert isinstance(out_df, pd.DataFrame)
        assert isinstance(diag, dict)
        assert "missingness_before" in diag
        assert "missingness_after" in diag

    def test_imputation_reduces_nans(self, df_with_nans, index_config):
        """NaN count decreases after standardize_and_clean."""
        nans_before = df_with_nans[["x", "y"]].isna().sum().sum()
        out_df, _ = standardize_and_clean(
            df_with_nans, index_cfg=index_config, required_cols=["x", "y"]
        )
        nans_after = out_df[["x", "y"]].isna().sum().sum()
        assert nans_after <= nans_before

    def test_no_required_cols(self, df_with_nans, index_config):
        """Works with empty required_cols (no coverage filter)."""
        out_df, diag = standardize_and_clean(
            df_with_nans, index_cfg=index_config, required_cols=[]
        )
        # All rows kept (no coverage filter)
        assert len(out_df) == len(df_with_nans)

    def test_imputation_method_none(self, df_with_nans):
        """When method='none', no imputation is done."""
        cfg = {
            "esg_index": {
                "missing_data": {
                    "min_indicator_coverage": 0.0,
                    "imputation": {"method": "none", "group_keys": ["sector"]},
                },
            }
        }
        out_df, _ = standardize_and_clean(
            df_with_nans, index_cfg=cfg, required_cols=[]
        )
        # NaN count unchanged since method=none
        assert out_df["x"].isna().sum() == df_with_nans["x"].isna().sum()
