"""Tests for the Paper 1 extensions (step 27) and the pre-registration (step 28).

Unit tests use synthetic data; artifact tests check that the generated paper
macros agree with the pipeline outputs they are built from, and are skipped
when those outputs have not been produced.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "reports" / "tables"
NUMBERS = ROOT / "Paper" / "generated" / "numbers.tex"
PREREG = ROOT / "preregistration" / "window2"


def _load(name, filename):
    cwd = os.getcwd()
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # scripts chdir to the project root
    os.chdir(cwd)
    return mod


ext = _load("ext27", "27_paper1_extensions.py")
prereg = _load("prereg28", "28_preregistration.py")


def _macros():
    out = {}
    for line in NUMBERS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}$", line)
        if m:
            out[m.group(1)] = m.group(2).replace(r"\ensuremath{-}", "-")
    return out


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_pct_is_rank_in_unit_interval_and_fills_missing():
    s = pd.Series([3.0, np.nan, 1.0, 2.0])
    p = ext.pct(s)
    assert p.notna().all()
    assert p.between(0, 1).all()
    assert p.iloc[2] < p.iloc[3] < p.iloc[0]


def test_one_way_costs_cover_both_countries():
    assert set(ext.ONE_WAY_COST_BPS) == {"US", "India"}
    assert ext.ONE_WAY_COST_BPS["India"] > ext.ONE_WAY_COST_BPS["US"] > 0


def test_provenance_classes_are_disjoint():
    assert not (ext.MEASURED & ext.PROXY)
    assert not (ext.MEASURED & ext.IMPUTED)
    assert not (ext.PROXY & ext.IMPUTED)


def test_prereg_window_is_after_window1_and_ordered():
    assert prereg.SNAPSHOT < prereg.DRY_ANCHOR < prereg.DRY_END <= prereg.ANCHOR < prereg.END


def test_sha256_matches_hashlib(tmp_path):
    import hashlib
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 1000)
    assert prereg.sha256(f) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_h4_regression_recovers_negative_esg_effect():
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame({
        "ESG_composite": rng.normal(size=n), "trail_vol": rng.normal(size=n),
        "trail_beta": rng.normal(size=n), "market_cap": rng.lognormal(size=n),
        "profitability_score": rng.normal(size=n), "debt_to_equity": rng.normal(size=n),
        "log_dollar_volume": rng.normal(size=n), "trail_ret": rng.normal(size=n),
        "price_to_book": rng.normal(size=n),
        "sector": rng.choice(["A", "B", "C"], n), "country": rng.choice(["US", "India"], n),
    })
    d["realized_vol_oos"] = 2 * d["trail_vol"] - 0.8 * d["ESG_composite"] + rng.normal(size=n)
    coef, lo, hi, p_one, nobs = prereg.h4_regression(d)
    assert coef < 0 and hi < 0 and p_one < 0.01 and nobs == n


# ---------------------------------------------------------------------------
# Artifact consistency (skipped until the pipeline has produced outputs)
# ---------------------------------------------------------------------------
needs_outputs = pytest.mark.skipif(
    not (NUMBERS.exists() and (TABLES / "ext_risk_controls.csv").exists()),
    reason="pipeline outputs not generated")


@needs_outputs
def test_h4_macro_matches_table():
    t = pd.read_csv(TABLES / "ext_risk_controls.csv")
    r = t[(t.outcome == "realized_vol_oos") & (t.spec == "full_controls")
          & (t.esg_measure == "ESG_composite")].iloc[0]
    assert float(_macros()["HFourEst"]) == pytest.approx(r["coef"], abs=5e-4)


@needs_outputs
def test_provenance_cells_sum_to_used_indicators():
    prov = pd.read_csv(TABLES / "ext_provenance_split.csv").set_index("measure")
    total = sum(prov.loc[m, "mean_cells_per_firm"] for m in ("ESG_measured", "ESG_proxy", "ESG_imputed"))
    assert total == pytest.approx(prov.loc["ESG_composite", "mean_cells_per_firm"], abs=0.1)


@needs_outputs
def test_weight_sweep_endpoints_and_deployed_flag():
    sw = pd.read_csv(TABLES / "ext_esg_weight_sweep.csv")
    assert sw["w_esg"].iloc[0] == 0 and sw["w_esg"].iloc[-1] == 1
    assert sw["is_deployed"].sum() == 1


@needs_outputs
def test_monthly_windows_cover_oos_period():
    mon = pd.read_csv(TABLES / "ext_monthly.csv", parse_dates=["start", "end"])
    meta = pd.read_csv(TABLES / "oos_metadata.csv").iloc[0]
    assert str(mon["start"].iloc[0].date()) == meta["snapshot_date"]
    assert str(mon["end"].iloc[-1].date()) == meta["end_date"]
    assert (mon["start"].iloc[1:].to_numpy() == mon["end"].iloc[:-1].to_numpy()).all()


@needs_outputs
def test_share_of_negative_ics_is_complement():
    m = _macros()
    assert float(m["WSharePos"]) + float(m["WShareNeg"]) == pytest.approx(100, abs=0.11)


@pytest.mark.skipif(not (PREREG / "manifest.json").exists(), reason="registration not frozen")
def test_registration_hashes_match():
    man = json.loads((PREREG / "manifest.json").read_text(encoding="utf-8"))
    for path, h in man["sha256"].items():
        assert prereg.sha256(ROOT / path) == h, f"{path} changed since registration"
    assert man["anchor"] == str(prereg.ANCHOR.date())
    assert man["end"] == str(prereg.END.date())


@pytest.mark.skipif(not (PREREG / "manifest.json").exists(), reason="registration not frozen")
def test_confirmatory_evaluation_refuses_before_window_closes():
    if pd.Timestamp.today() >= prereg.END + pd.Timedelta(days=3):
        pytest.skip("window has closed")
    with pytest.raises(SystemExit):
        prereg.evaluate(dry_run=False)
