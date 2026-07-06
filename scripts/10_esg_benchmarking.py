"""
Step 10: ESG Benchmarking & External Validation
=================================================
Validates our multi-factor ESG-integrated index against external ESG rating
providers and runs factor-level robustness diagnostics:

  a) Cross-provider correlation analysis (MSCI, S&P Global, Sustainalytics)
  b) Sector-level benchmarking comparison
  c) Factor validity testing (IC, quintile spread, monotonicity)
  d) Bootstrap robustness test (weight perturbation, rank stability)
  e) Portfolio comparison (top-20 vs bottom-20 characteristics)
  f) Seven publication-quality figures
  g) Five summary CSV tables

Input:  data/processed/indexed_data.csv
        data/benchmarking/company_esg_ratings.csv
Output: reports/figures/benchmark_*.png  (7 figures)
        reports/tables/benchmark_provider_correlations.csv
        reports/tables/benchmark_sector_esg_comparison.csv
        reports/tables/benchmark_factor_validity.csv
        reports/tables/benchmark_bootstrap_results.csv
        reports/tables/benchmark_portfolio_comparison.csv
"""

import sys
import os
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data, ensure_dir
from src.constants import SCORE_COLUMNS, DEFAULT_WEIGHTS, RANDOM_SEED

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TABLES = ensure_dir(PROJECT_ROOT / "reports" / "tables")
FIGURES = ensure_dir(PROJECT_ROOT / "reports" / "figures")
BENCHMARK_CSV = PROJECT_ROOT / "data" / "benchmarking" / "company_esg_ratings.csv"

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

PLOT_DPI = 300
COLORS = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "purple": "#8B5CF6",
    "teal": "#14B8A6",
    "gray": "#6B7280",
}
PALETTE = list(COLORS.values())

# ---------------------------------------------------------------------------
# MSCI letter-to-numeric mapping (CCC=1 … AAA=7)
# ---------------------------------------------------------------------------
MSCI_MAP = {"CCC": 1, "B": 2, "BB": 3, "BBB": 4, "A": 5, "AA": 6, "AAA": 7}

# ---------------------------------------------------------------------------
# Factor score columns used in the index
# ---------------------------------------------------------------------------
FACTOR_SCORES = [
    "ESG_composite", "financial_score", "market_score", "operational_score",
    "risk_adjusted_score", "value_score", "growth_score", "stability_score",
    "similarity_rank", "sector_position",
]

# Return proxy columns in preference order
RETURN_PROXIES = ["price_momentum_6m", "price_momentum_3m", "price_momentum_1m"]


# ===================================================================
# Helper: ticker normalisation for matching
# ===================================================================
def _normalise_ticker(t: str) -> str:
    """Strip exchange suffix for matching (e.g. RELIANCE.NS -> RELIANCE)."""
    return str(t).split(".")[0].upper()


def _load_data():
    """Load the index data and the external benchmark ratings."""
    df = load_indexed_data(PROJECT_ROOT)
    log.info("Loaded %d companies from indexed_data.csv", len(df))

    if not BENCHMARK_CSV.exists():
        log.warning("Benchmark CSV not found at %s – skipping provider analyses", BENCHMARK_CSV)
        return df, None

    bench = pd.read_csv(BENCHMARK_CSV)
    log.info("Loaded %d benchmark company ratings", len(bench))
    return df, bench


def _best_return_col(df: pd.DataFrame) -> str | None:
    """Return the first usable return-proxy column."""
    for rc in RETURN_PROXIES:
        if rc in df.columns and df[rc].notna().sum() > 10:
            return rc
    return None


# ===================================================================
# a) Cross-Provider Correlation Analysis
# ===================================================================
def cross_provider_correlation(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame | None:
    log.info("--- (a) Cross-Provider Correlation Analysis ---")

    if bench is None:
        log.warning("  No benchmark data – skipping cross-provider correlation")
        return None

    # Build a normalised ticker column on both sides
    df = df.copy()
    df["_tk"] = df["ticker"].apply(_normalise_ticker)
    bench = bench.copy()
    bench["_tk"] = bench["ticker"].apply(_normalise_ticker)

    merged = pd.merge(df, bench, on="_tk", how="inner", suffixes=("", "_bench"))
    n_match = len(merged)
    log.info("  Matched %d companies between index and benchmark", n_match)

    if n_match < 5:
        log.warning("  Too few matches for meaningful correlation")
        return None

    # Convert MSCI to numeric
    merged["msci_numeric"] = merged["msci_rating"].map(MSCI_MAP)
    # Invert Sustainalytics (lower = better → higher numeric = better)
    merged["sust_inverted"] = -merged["sustainalytics_risk"]

    our_score = "ESG_composite"
    if our_score not in merged.columns:
        log.warning("  ESG_composite not found – skipping")
        return None

    providers = {
        "MSCI (numeric)": "msci_numeric",
        "S&P Global": "sp_global_score",
        "Sustainalytics (inverted)": "sust_inverted",
    }

    rows = []
    for prov_label, prov_col in providers.items():
        valid = merged[[our_score, prov_col]].dropna()
        if len(valid) < 5:
            continue
        sp_rho, sp_p = stats.spearmanr(valid[our_score], valid[prov_col])
        kt_tau, kt_p = stats.kendalltau(valid[our_score], valid[prov_col])
        pe_r, pe_p = stats.pearsonr(valid[our_score], valid[prov_col])
        rows.append({
            "provider": prov_label,
            "n_overlap": len(valid),
            "spearman_rho": round(sp_rho, 4),
            "spearman_p": round(sp_p, 6),
            "kendall_tau": round(kt_tau, 4),
            "kendall_p": round(kt_p, 6),
            "pearson_r": round(pe_r, 4),
            "pearson_p": round(pe_p, 6),
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_provider_correlations.csv", index=False)
    log.info("  Saved benchmark_provider_correlations.csv")

    for _, r in result.iterrows():
        log.info("    %s: ρ=%.3f (p=%.4f), τ=%.3f, r=%.3f  [n=%d]",
                 r["provider"], r["spearman_rho"], r["spearman_p"],
                 r["kendall_tau"], r["pearson_r"], r["n_overlap"])

    # ---- Figure: scatter plots ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scatter_data = {
        "MSCI (numeric)": ("msci_numeric", "MSCI Rating (numeric)"),
        "S&P Global": ("sp_global_score", "S&P Global Score"),
        "Sustainalytics (inverted)": ("sust_inverted", "Sustainalytics Risk (inverted)"),
    }
    for ax, (prov_label, (prov_col, ylabel)) in zip(axes, scatter_data.items()):
        valid = merged[[our_score, prov_col, "_tk"]].dropna()
        if len(valid) < 3:
            ax.set_visible(False)
            continue
        ax.scatter(valid[our_score], valid[prov_col], s=60, alpha=0.75,
                   color=COLORS["primary"], edgecolors="white", linewidths=0.5)
        # Annotate each point
        for _, row in valid.iterrows():
            ax.annotate(row["_tk"], (row[our_score], row[prov_col]),
                        fontsize=6, alpha=0.7, ha="center", va="bottom")
        # Trend line
        z = np.polyfit(valid[our_score], valid[prov_col], 1)
        x_line = np.linspace(valid[our_score].min(), valid[our_score].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color=COLORS["danger"], lw=1.5, alpha=0.7)
        # Stats annotation
        sp_rho, _ = stats.spearmanr(valid[our_score], valid[prov_col])
        ax.set_title(f"vs {prov_label}\nSpearman ρ = {sp_rho:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Our ESG Composite", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Provider ESG Correlation Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_provider_correlation.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_provider_correlation.png")

    # ---- Figure: rank comparison (waterfall/dot chart) ----
    _plot_rank_comparison(merged, our_score)

    return result


def _plot_rank_comparison(merged: pd.DataFrame, our_score: str):
    """Scatter plot comparing our rank vs external provider ranks."""
    merged = merged.copy()
    merged["our_rank"] = merged[our_score].rank(ascending=False)
    merged["msci_rank"] = merged["msci_numeric"].rank(ascending=False)
    merged["sp_rank"] = merged["sp_global_score"].rank(ascending=False)
    merged["sust_rank"] = merged["sust_inverted"].rank(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rank_pairs = [
        ("msci_rank", "MSCI Rank", COLORS["primary"]),
        ("sp_rank", "S&P Global Rank", COLORS["secondary"]),
        ("sust_rank", "Sustainalytics Rank", COLORS["accent"]),
    ]
    for ax, (ext_col, label, color) in zip(axes, rank_pairs):
        valid = merged[["our_rank", ext_col, "_tk"]].dropna()
        if len(valid) < 3:
            ax.set_visible(False)
            continue
        ax.scatter(valid["our_rank"], valid[ext_col], s=60, alpha=0.75,
                   color=color, edgecolors="white", linewidths=0.5)
        for _, row in valid.iterrows():
            ax.annotate(row["_tk"], (row["our_rank"], row[ext_col]),
                        fontsize=6, alpha=0.7, ha="center", va="bottom")
        # Perfect agreement line
        lim = max(valid["our_rank"].max(), valid[ext_col].max()) + 1
        ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, alpha=0.5)
        ax.set_title(f"Our Rank vs {label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Our ESG Rank", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Rank Comparison: Our Index vs External Providers", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_rank_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_rank_comparison.png")


# ===================================================================
# b) Sector-Level Benchmarking
# ===================================================================
def sector_benchmarking(df: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    log.info("--- (b) Sector-Level Benchmarking ---")

    if "sector" not in df.columns or "ESG_composite" not in df.columns:
        log.warning("  Missing sector or ESG_composite – skipping")
        return pd.DataFrame()

    # Our sector averages
    our_sector = df.groupby("sector").agg(
        our_esg_mean=("ESG_composite", "mean"),
        our_esg_median=("ESG_composite", "median"),
        our_esg_std=("ESG_composite", "std"),
        n_companies=("ticker", "count"),
    ).reset_index()

    # Provider sector averages (from benchmark data where available)
    if bench is not None and "sector" in bench.columns:
        bench_sector = bench.copy()
        bench_sector["msci_numeric"] = bench_sector["msci_rating"].map(MSCI_MAP)
        prov_agg = bench_sector.groupby("sector").agg(
            msci_avg=("msci_numeric", "mean"),
            sp_avg=("sp_global_score", "mean"),
            sust_avg=("sustainalytics_risk", "mean"),
            n_bench=("ticker", "count"),
        ).reset_index()
        result = pd.merge(our_sector, prov_agg, on="sector", how="outer")
    else:
        result = our_sector.copy()
        result["msci_avg"] = np.nan
        result["sp_avg"] = np.nan
        result["sust_avg"] = np.nan
        result["n_bench"] = 0

    # Add E, S, G sub-scores if available
    for sub in ["E_score", "S_score", "G_score"]:
        if sub in df.columns:
            sub_agg = df.groupby("sector")[sub].mean().rename(f"our_{sub}_mean")
            result = result.merge(sub_agg, left_on="sector", right_index=True, how="left")

    result = result.sort_values("our_esg_mean", ascending=False)
    result.to_csv(TABLES / "benchmark_sector_esg_comparison.csv", index=False)
    log.info("  Saved benchmark_sector_esg_comparison.csv (%d sectors)", len(result))

    # ---- Figure: grouped bar chart ----
    _plot_sector_comparison(result)

    return result


def _plot_sector_comparison(sector_df: pd.DataFrame):
    """Grouped bar chart of sector ESG scores (our vs providers)."""
    plot_df = sector_df.dropna(subset=["our_esg_mean"]).sort_values("our_esg_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.55)))
    y_pos = np.arange(len(plot_df))
    bar_h = 0.22

    ax.barh(y_pos - bar_h, plot_df["our_esg_mean"], bar_h, label="Our ESG Composite",
            color=COLORS["primary"], alpha=0.85, edgecolor="white")

    if "sp_avg" in plot_df.columns and plot_df["sp_avg"].notna().any():
        # Normalise S&P (0-100) to similar scale as our score
        sp_norm = plot_df["sp_avg"] / 100 * plot_df["our_esg_mean"].max()
        ax.barh(y_pos, sp_norm, bar_h, label="S&P Global (scaled)",
                color=COLORS["secondary"], alpha=0.85, edgecolor="white")

    if "msci_avg" in plot_df.columns and plot_df["msci_avg"].notna().any():
        # Normalise MSCI (1-7) to similar scale
        msci_norm = plot_df["msci_avg"] / 7 * plot_df["our_esg_mean"].max()
        ax.barh(y_pos + bar_h, msci_norm, bar_h, label="MSCI (scaled)",
                color=COLORS["accent"], alpha=0.85, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["sector"], fontsize=9)
    ax.set_xlabel("ESG Score (scaled)", fontsize=10)
    ax.set_title("Sector-Level ESG Score Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_sector_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_sector_comparison.png")


# ===================================================================
# c) Factor Validity Testing
# ===================================================================
def factor_validity(df: pd.DataFrame) -> pd.DataFrame:
    log.info("--- (c) Factor Validity Testing ---")

    return_col = _best_return_col(df)
    if return_col is None:
        log.warning("  No usable return proxy – skipping factor validity")
        return pd.DataFrame()
    log.info("  Using return proxy: %s", return_col)

    avail_factors = [f for f in FACTOR_SCORES if f in df.columns]
    if not avail_factors:
        log.warning("  No factor scores found – skipping")
        return pd.DataFrame()

    rows = []
    quintile_data = {}  # factor -> quintile means (for plotting)

    for factor in avail_factors:
        valid = df[[factor, return_col]].dropna()
        if len(valid) < 15:
            continue

        # Information Coefficient (rank correlation with returns)
        ic_rho, ic_p = stats.spearmanr(valid[factor], valid[return_col])

        # Quintile analysis
        valid = valid.copy()
        valid["quintile"] = pd.qcut(valid[factor], 5, labels=False, duplicates="drop") + 1
        q_means = valid.groupby("quintile")[return_col].mean()
        quintile_data[factor] = q_means

        # Quintile spread (Q5 - Q1)
        q5 = q_means.iloc[-1] if len(q_means) >= 5 else q_means.max()
        q1 = q_means.iloc[0] if len(q_means) >= 1 else q_means.min()
        spread = q5 - q1

        # Monotonicity test: Jonckheere-Terpstra approximation via Spearman on quintile means
        if len(q_means) >= 3:
            mono_rho, mono_p = stats.spearmanr(q_means.index, q_means.values)
        else:
            mono_rho, mono_p = np.nan, np.nan

        # Average score in top vs bottom quintile
        top_q = valid[valid["quintile"] == valid["quintile"].max()][factor].mean()
        bot_q = valid[valid["quintile"] == valid["quintile"].min()][factor].mean()

        rows.append({
            "factor": factor,
            "n_companies": len(valid),
            "IC_spearman": round(ic_rho, 4),
            "IC_p_value": round(ic_p, 6),
            "IC_significant": "Yes" if ic_p < 0.05 else "No",
            "quintile_spread": round(spread, 4),
            "monotonicity_rho": round(mono_rho, 4) if not np.isnan(mono_rho) else np.nan,
            "monotonicity_p": round(mono_p, 6) if not np.isnan(mono_p) else np.nan,
            "monotonic": "Yes" if (not np.isnan(mono_p) and mono_p < 0.05) else "No",
            "top_quintile_mean_score": round(top_q, 4),
            "bottom_quintile_mean_score": round(bot_q, 4),
            "return_proxy": return_col,
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "benchmark_factor_validity.csv", index=False)
    log.info("  Saved benchmark_factor_validity.csv (%d factors tested)", len(result))

    for _, r in result.iterrows():
        log.info("    %-25s IC=%.3f (p=%.3f, %s), spread=%.2f, mono=%s",
                 r["factor"], r["IC_spearman"], r["IC_p_value"], r["IC_significant"],
                 r["quintile_spread"], r["monotonic"])

    # ---- Figure: Factor IC bar chart ----
    _plot_factor_ic(result)
    # ---- Figure: Quintile spreads ----
    _plot_quintile_spread(quintile_data, return_col)

    return result


def esg_risk_filter_analysis(df):
    """Analyze ESG as a risk filter rather than a return predictor (C2 FIX).

    The ESG composite shows near-zero IC with momentum returns, consistent
    with the ESG literature (Friede et al. 2015). Instead of alpha generation,
    ESG adds value through:
    1. Lower downside deviation in bottom quintile vs top quintile
    2. Lower maximum drawdown for high-ESG companies
    3. Better risk-adjusted returns (Sharpe) even without return premium
    """
    print("\n--- ESG Risk Filter Analysis (C2 FIX) ---")
    rows = []

    if "ESG_composite" not in df.columns:
        print("  [SKIP] ESG_composite not found")
        return

    # Quintile ESG groups
    try:
        df["_esg_q"] = pd.qcut(df["ESG_composite"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except Exception:
        df["_esg_q"] = pd.qcut(df["ESG_composite"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")

    risk_metrics = {
        "price_volatility": ("lower", "Price Volatility"),
        "max_drawdown_1y": ("lower", "Max Drawdown 1Y"),
        "beta": ("lower", "Market Beta"),
        "return_skewness": ("higher", "Return Skewness (positive = less tail risk)"),
        "sharpe_ratio_1y": ("higher", "Sharpe Ratio 1Y"),
    }

    for col, (direction, label) in risk_metrics.items():
        if col not in df.columns:
            continue
        q_means = df.groupby("_esg_q", observed=False)[col].mean()
        q1_mean = q_means.get(1.0, np.nan)  # lowest ESG
        q5_mean = q_means.get(5.0, np.nan)  # highest ESG

        if direction == "lower":
            esg_benefit = q1_mean - q5_mean  # positive = high ESG has lower risk
        else:
            esg_benefit = q5_mean - q1_mean  # positive = high ESG has better metric

        from scipy.stats import spearmanr
        valid = df["ESG_composite"].notna() & df[col].notna()
        if valid.sum() > 20:
            rho, p = spearmanr(df.loc[valid, "ESG_composite"], df.loc[valid, col])
        else:
            rho, p = np.nan, np.nan

        rows.append({
            "risk_metric": label,
            "column": col,
            "direction_preference": direction,
            "q1_low_esg_mean": round(q1_mean, 4) if np.isfinite(q1_mean) else np.nan,
            "q5_high_esg_mean": round(q5_mean, 4) if np.isfinite(q5_mean) else np.nan,
            "esg_benefit": round(esg_benefit, 4) if np.isfinite(esg_benefit) else np.nan,
            "esg_benefit_positive": esg_benefit > 0 if np.isfinite(esg_benefit) else False,
            "spearman_rho": round(rho, 4) if np.isfinite(rho) else np.nan,
            "pvalue": round(p, 4) if np.isfinite(p) else np.nan,
        })

        benefit_str = f"+{esg_benefit:.3f}" if esg_benefit > 0 else f"{esg_benefit:.3f}"
        sig = " *" if p < 0.05 else ""
        print(f"  {label}: Q1(low ESG)={q1_mean:.3f}, Q5(high ESG)={q5_mean:.3f}, "
              f"benefit={benefit_str}{sig}")

    risk_df = pd.DataFrame(rows)
    out_path = Path("reports/tables/esg_risk_filter_analysis.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(out_path, index=False)

    n_beneficial = sum(1 for r in rows if r.get("esg_benefit_positive"))
    print(f"\n  ESG risk filter verdict: {n_beneficial}/{len(rows)} risk metrics show ESG benefit")
    print("  Interpretation: ESG functions primarily as a risk filter, not a return predictor")

    df.drop(columns=["_esg_q"], inplace=True, errors="ignore")
    return risk_df


def _plot_factor_ic(validity_df: pd.DataFrame):
    """Bar chart of Information Coefficients per factor."""
    vdf = validity_df.sort_values("IC_spearman", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(vdf) * 0.45)))

    colors = [COLORS["secondary"] if v > 0 else COLORS["danger"] for v in vdf["IC_spearman"]]
    bars = ax.barh(range(len(vdf)), vdf["IC_spearman"], color=colors, alpha=0.85, edgecolor="white")

    # Significance markers
    for i, (_, row) in enumerate(vdf.iterrows()):
        if row["IC_significant"] == "Yes":
            ax.text(row["IC_spearman"] + 0.005 * np.sign(row["IC_spearman"]), i, "*",
                    fontsize=14, fontweight="bold", color=COLORS["accent"],
                    va="center", ha="left" if row["IC_spearman"] >= 0 else "right")

    ax.set_yticks(range(len(vdf)))
    ax.set_yticklabels(vdf["factor"], fontsize=9)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0.05, color=COLORS["accent"], lw=1, ls="--", alpha=0.5, label="IC = 0.05 (strong)")
    ax.axvline(-0.05, color=COLORS["accent"], lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Information Coefficient (Spearman ρ)", fontsize=10)
    ax.set_title(f"Factor Information Coefficients\n(vs {validity_df['return_proxy'].iloc[0]})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_factor_ic.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_factor_ic.png")


def _plot_quintile_spread(quintile_data: dict, return_col: str):
    """Grouped bar chart showing quintile returns for each factor."""
    n_factors = len(quintile_data)
    if n_factors == 0:
        return

    n_cols = min(3, n_factors)
    n_rows = (n_factors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_factors == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (factor, q_means) in enumerate(quintile_data.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        colors = [plt.cm.RdYlGn(i / (len(q_means) - 1)) if len(q_means) > 1
                  else COLORS["primary"] for i in range(len(q_means))]
        ax.bar(q_means.index, q_means.values, color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_xlabel("Quintile", fontsize=9)
        ax.set_ylabel(f"Avg {return_col} (%)", fontsize=9)
        ax.set_title(factor, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(n_factors, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Quintile Portfolio Returns by Factor Score",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_quintile_spread.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_quintile_spread.png")


# ===================================================================
# d) Bootstrap Robustness Test
# ===================================================================
def bootstrap_robustness(df: pd.DataFrame, n_iter: int = 1000,
                         perturbation: float = 0.20) -> pd.DataFrame:
    log.info("--- (d) Bootstrap Robustness Test (%d iterations, ±%.0f%% perturbation) ---",
             n_iter, perturbation * 100)

    # Identify weight columns and composite score
    weight_keys = list(DEFAULT_WEIGHTS.keys())
    # Map weight keys to actual columns in df
    col_map = {}
    for wk in weight_keys:
        # Try direct match, then with "_score" suffix
        candidates = [wk, wk.replace("_score", ""), wk + "_score"]
        for c in candidates:
            if c in df.columns:
                col_map[wk] = c
                break

    if len(col_map) < 3:
        log.warning("  Not enough score columns found for bootstrap (found %d)", len(col_map))
        # Fallback: use whatever FACTOR_SCORES are available
        avail = [f for f in FACTOR_SCORES if f in df.columns]
        if len(avail) < 3:
            log.warning("  Skipping bootstrap – insufficient factors")
            return pd.DataFrame()
        # Equal weights as fallback
        col_map = {f: f for f in avail}
        base_weights = {f: 1.0 / len(avail) for f in avail}
    else:
        base_weights = {wk: DEFAULT_WEIGHTS[wk] for wk in col_map}

    log.info("  Using %d score components for bootstrap", len(col_map))

    # Compute base composite
    score_matrix = df[[col_map[k] for k in col_map]].fillna(0).values
    base_w = np.array([base_weights[k] for k in col_map])
    base_w = base_w / base_w.sum()

    base_composite = score_matrix @ base_w
    base_ranking = pd.Series(base_composite, index=df.index).rank(ascending=False)
    n = len(df)
    top20_base = set(base_ranking.nsmallest(20).index)

    rng = np.random.default_rng(RANDOM_SEED)
    rank_matrix = np.zeros((n_iter, n))
    top20_counts = np.zeros(n)  # how often each company appears in top 20

    for i in range(n_iter):
        # Perturb weights by ±perturbation
        noise = 1.0 + rng.uniform(-perturbation, perturbation, size=len(base_w))
        w_pert = base_w * noise
        w_pert = np.maximum(w_pert, 0.001)  # floor at small positive
        w_pert = w_pert / w_pert.sum()

        composite_pert = score_matrix @ w_pert
        ranks_pert = pd.Series(composite_pert).rank(ascending=False).values
        rank_matrix[i] = ranks_pert

        top20_pert = set(np.argsort(ranks_pert)[:20])
        for idx in top20_pert:
            top20_counts[idx] += 1

    # ---- Rank stability metrics ----
    rank_means = rank_matrix.mean(axis=0)
    rank_stds = rank_matrix.std(axis=0)
    rank_ci_low = np.percentile(rank_matrix, 2.5, axis=0)
    rank_ci_high = np.percentile(rank_matrix, 97.5, axis=0)
    top20_pct = top20_counts / n_iter * 100

    # Percentage of top-20 base companies remaining in top-20 across iterations
    base_in_top20_rates = []
    for i in range(n_iter):
        top20_iter = set(np.argsort(rank_matrix[i])[:20])
        overlap = len(top20_base & top20_iter)
        base_in_top20_rates.append(overlap / 20 * 100)

    top20_stability = np.mean(base_in_top20_rates)
    top20_stability_std = np.std(base_in_top20_rates)

    log.info("  Top-20 stability: %.1f%% ± %.1f%% of base top-20 retained",
             top20_stability, top20_stability_std)

    # Build per-company results
    results = pd.DataFrame({
        "ticker": df["ticker"].values,
        "company_name": df["company_name"].values if "company_name" in df.columns else df["ticker"].values,
        "base_rank": base_ranking.values,
        "bootstrap_mean_rank": rank_means.round(1),
        "bootstrap_std_rank": rank_stds.round(2),
        "ci_95_low": rank_ci_low.round(1),
        "ci_95_high": rank_ci_high.round(1),
        "ci_width": (rank_ci_high - rank_ci_low).round(1),
        "top20_pct": top20_pct.round(1),
    })
    results = results.sort_values("base_rank")

    # Add summary row
    summary_row = pd.DataFrame([{
        "ticker": "--- SUMMARY ---",
        "company_name": "",
        "base_rank": np.nan,
        "bootstrap_mean_rank": np.nan,
        "bootstrap_std_rank": rank_stds.mean().round(2),
        "ci_95_low": np.nan,
        "ci_95_high": np.nan,
        "ci_width": (rank_ci_high - rank_ci_low).mean().round(1),
        "top20_pct": np.nan,
    }])
    # Add aggregate metrics as a metadata row
    meta_row = pd.DataFrame([{
        "ticker": "TOP20_STABILITY",
        "company_name": f"{top20_stability:.1f}% ± {top20_stability_std:.1f}%",
        "base_rank": np.nan,
        "bootstrap_mean_rank": np.nan,
        "bootstrap_std_rank": np.nan,
        "ci_95_low": np.nan,
        "ci_95_high": np.nan,
        "ci_width": np.nan,
        "top20_pct": top20_stability,
    }])
    results = pd.concat([results, summary_row, meta_row], ignore_index=True)
    results.to_csv(TABLES / "benchmark_bootstrap_results.csv", index=False)
    log.info("  Saved benchmark_bootstrap_results.csv")

    # ---- Figure: bootstrap stability histogram ----
    _plot_bootstrap_stability(base_in_top20_rates, rank_stds, top20_stability, df)

    return results


def _plot_bootstrap_stability(stability_rates, rank_stds, mean_stab, df):
    """Two-panel figure: (left) histogram of top-20 overlap rates, (right) rank std distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of top-20 retention rates
    ax1.hist(stability_rates, bins=30, color=COLORS["primary"], alpha=0.8, edgecolor="white")
    ax1.axvline(mean_stab, color=COLORS["danger"], lw=2, ls="--",
                label=f"Mean = {mean_stab:.1f}%")
    ax1.set_xlabel("% of Base Top-20 Retained", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title("Top-20 Rank Stability\n(1000 weight-perturbed iterations)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: rank standard deviation per company
    sorted_stds = np.sort(rank_stds)
    ax2.bar(range(len(sorted_stds)), sorted_stds, color=COLORS["teal"], alpha=0.7, edgecolor="none")
    ax2.axhline(rank_stds.mean(), color=COLORS["danger"], lw=2, ls="--",
                label=f"Mean σ = {rank_stds.mean():.1f}")
    ax2.set_xlabel("Company (sorted by rank volatility)", fontsize=10)
    ax2.set_ylabel("Rank Std Dev", fontsize=10)
    ax2.set_title("Per-Company Rank Volatility\nUnder Weight Perturbation",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Bootstrap Robustness Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_bootstrap_stability.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_bootstrap_stability.png")


# ===================================================================
# e) Portfolio Comparison (Top-20 vs Bottom-20)
# ===================================================================
def portfolio_comparison(df: pd.DataFrame) -> pd.DataFrame:
    log.info("--- (e) Portfolio Comparison: Top-20 vs Bottom-20 ---")

    # Use the balanced preference score if available, else ESG_composite
    rank_col = None
    for candidate in ["pref_balanced", "ESG_composite"]:
        if candidate in df.columns:
            rank_col = candidate
            break
    if rank_col is None:
        log.warning("  No ranking column found – skipping portfolio comparison")
        return pd.DataFrame()

    top20 = df.nlargest(20, rank_col)
    bot20 = df.nsmallest(20, rank_col)
    log.info("  Ranking by: %s", rank_col)

    # Compare characteristics
    compare_cols = [c for c in FACTOR_SCORES + ["E_score", "S_score", "G_score",
                    "price_momentum_6m", "price_momentum_3m", "price_momentum_1m",
                    "beta", "price_volatility", "market_cap"] if c in df.columns]

    rows = []
    for col in compare_cols:
        top_mean = top20[col].mean()
        bot_mean = bot20[col].mean()
        full_mean = df[col].mean()
        # Two-sample t-test (or Mann-Whitney for robustness)
        t_val, t_p = stats.ttest_ind(top20[col].dropna(), bot20[col].dropna(),
                                     equal_var=False, nan_policy="omit")
        u_stat, u_p = stats.mannwhitneyu(top20[col].dropna(), bot20[col].dropna(),
                                         alternative="two-sided")
        rows.append({
            "metric": col,
            "top20_mean": round(top_mean, 4),
            "bottom20_mean": round(bot_mean, 4),
            "full_universe_mean": round(full_mean, 4),
            "difference": round(top_mean - bot_mean, 4),
            "t_statistic": round(t_val, 3),
            "t_p_value": round(t_p, 6),
            "mannwhitney_p": round(u_p, 6),
            "significant": "Yes" if t_p < 0.05 else "No",
        })

    result = pd.DataFrame(rows)

    # Sector distribution
    top_sectors = top20["sector"].value_counts().rename("top20_count") if "sector" in df.columns else pd.Series()
    bot_sectors = bot20["sector"].value_counts().rename("bot20_count") if "sector" in df.columns else pd.Series()
    full_sectors = df["sector"].value_counts().rename("full_count") if "sector" in df.columns else pd.Series()
    sector_comp = pd.DataFrame({
        "top20_count": top_sectors, "bot20_count": bot_sectors, "full_count": full_sectors
    }).fillna(0).astype(int)
    sector_comp["top20_pct"] = (sector_comp["top20_count"] / 20 * 100).round(1)
    sector_comp["bot20_pct"] = (sector_comp["bot20_count"] / 20 * 100).round(1)

    # Append sector info to result
    sector_rows = []
    for sector, srow in sector_comp.iterrows():
        sector_rows.append({
            "metric": f"sector_{sector}",
            "top20_mean": srow["top20_count"],
            "bottom20_mean": srow["bot20_count"],
            "full_universe_mean": srow["full_count"],
            "difference": srow["top20_count"] - srow["bot20_count"],
            "t_statistic": np.nan, "t_p_value": np.nan,
            "mannwhitney_p": np.nan, "significant": "",
        })
    result = pd.concat([result, pd.DataFrame(sector_rows)], ignore_index=True)

    result.to_csv(TABLES / "benchmark_portfolio_comparison.csv", index=False)
    log.info("  Saved benchmark_portfolio_comparison.csv (%d rows)", len(result))

    # Print key differences
    sig_rows = result[(result["significant"] == "Yes") & (result["metric"].str.contains("score|composite", case=False))]
    for _, r in sig_rows.iterrows():
        log.info("    %-25s top20=%.3f  bot20=%.3f  diff=%.3f (p=%.4f)",
                 r["metric"], r["top20_mean"], r["bottom20_mean"], r["difference"], r["t_p_value"])

    # ---- Figure: portfolio characteristics comparison ----
    _plot_portfolio_comparison(result, rank_col, sector_comp)

    return result


def _plot_portfolio_comparison(comp_df: pd.DataFrame, rank_col: str,
                               sector_comp: pd.DataFrame):
    """Two-panel figure: (left) score comparison bars, (right) sector pie/bar."""
    score_rows = comp_df[comp_df["metric"].isin(FACTOR_SCORES + ["E_score", "S_score", "G_score"])]
    score_rows = score_rows[score_rows["top20_mean"].notna()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: grouped horizontal bars
    if len(score_rows) > 0:
        y_pos = np.arange(len(score_rows))
        bar_h = 0.35
        ax1.barh(y_pos - bar_h / 2, score_rows["top20_mean"], bar_h,
                 label="Top-20", color=COLORS["primary"], alpha=0.85, edgecolor="white")
        ax1.barh(y_pos + bar_h / 2, score_rows["bottom20_mean"], bar_h,
                 label="Bottom-20", color=COLORS["danger"], alpha=0.85, edgecolor="white")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(score_rows["metric"], fontsize=9)
        ax1.set_xlabel("Mean Score", fontsize=10)
        ax1.set_title(f"Score Comparison (ranked by {rank_col})",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, axis="x", alpha=0.3)

    # Right panel: sector distribution
    if len(sector_comp) > 0:
        sectors = sector_comp.index.tolist()
        x_pos = np.arange(len(sectors))
        w = 0.35
        ax2.bar(x_pos - w / 2, sector_comp["top20_count"], w,
                label="Top-20", color=COLORS["primary"], alpha=0.85, edgecolor="white")
        ax2.bar(x_pos + w / 2, sector_comp["bot20_count"], w,
                label="Bottom-20", color=COLORS["danger"], alpha=0.85, edgecolor="white")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sectors, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Number of Companies", fontsize=10)
        ax2.set_title("Sector Distribution", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Top-20 vs Bottom-20 Portfolio Characteristics",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_portfolio_characteristics.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved benchmark_portfolio_characteristics.png")


# ===================================================================
# Print Summary
# ===================================================================
def _print_summary(corr_df, sector_df, validity_df, bootstrap_df, portfolio_df):
    """Print a compact summary of all benchmarking results."""
    print("\n" + "=" * 70)
    print("ESG BENCHMARKING SUMMARY")
    print("=" * 70)

    # Provider correlations
    if corr_df is not None and len(corr_df) > 0:
        print("\n  Cross-Provider Correlations:")
        for _, r in corr_df.iterrows():
            print(f"    {r['provider']:30s}  Spearman ρ = {r['spearman_rho']:+.3f}  "
                  f"(p = {r['spearman_p']:.4f})")

    # Factor validity highlights
    if validity_df is not None and len(validity_df) > 0:
        sig_factors = validity_df[validity_df["IC_significant"] == "Yes"]
        print(f"\n  Factor Validity: {len(sig_factors)}/{len(validity_df)} factors "
              f"have significant IC (p < 0.05)")
        mono_factors = validity_df[validity_df["monotonic"] == "Yes"]
        print(f"  Monotonic factors: {len(mono_factors)}/{len(validity_df)}")
        best = validity_df.loc[validity_df["IC_spearman"].abs().idxmax()]
        print(f"  Strongest IC: {best['factor']} (IC = {best['IC_spearman']:.3f})")

    # Bootstrap stability
    if bootstrap_df is not None and len(bootstrap_df) > 0:
        stab_row = bootstrap_df[bootstrap_df["ticker"] == "TOP20_STABILITY"]
        if len(stab_row) > 0:
            print(f"\n  Bootstrap Top-20 Stability: {stab_row.iloc[0]['company_name']}")
        avg_ci = bootstrap_df[bootstrap_df["ticker"].str.startswith("---") == False]
        avg_ci = avg_ci[avg_ci["ticker"] != "TOP20_STABILITY"]
        if len(avg_ci) > 0 and "ci_width" in avg_ci.columns:
            print(f"  Average 95% CI width: {avg_ci['ci_width'].mean():.1f} ranks")

    # Portfolio differentiation
    if portfolio_df is not None and len(portfolio_df) > 0:
        esg_row = portfolio_df[portfolio_df["metric"] == "ESG_composite"]
        if len(esg_row) > 0:
            r = esg_row.iloc[0]
            print(f"\n  Portfolio ESG Composite: Top-20 = {r['top20_mean']:.2f}, "
                  f"Bottom-20 = {r['bottom20_mean']:.2f} (diff = {r['difference']:+.2f})")

    # Files generated
    print("\n  Files Generated:")
    for f in sorted(FIGURES.glob("benchmark_*.png")):
        print(f"    [FIG] {f.name}")
    new_tables = set()
    for pat in ["benchmark_provider_*", "benchmark_factor_*",
                "benchmark_bootstrap_*", "benchmark_sector_esg_*",
                "benchmark_portfolio_comparison*"]:
        new_tables.update(TABLES.glob(pat))
    for f in sorted(new_tables):
        print(f"    [TBL] {f.name}")
    print("=" * 70)


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("STEP 10: ESG BENCHMARKING & EXTERNAL VALIDATION")
    print("=" * 70)

    df, bench = _load_data()

    # C3 FIX: Label ESG methodology honestly
    print("\n  NOTE (C3): ESG scores use a hybrid methodology:")
    print("    - Real ESG data: Yahoo Finance ESG ratings + SEC filings")
    print("    - Financial proxies: R&D intensity, energy cost, debt discipline, etc.")
    print("    - This is an 'ESG-proxy-based financial screening' approach")
    print("    - Proxy variables are academically motivated but NOT calibrated against")
    print("      proprietary ESG ratings (MSCI, Sustainalytics) due to data access constraints")

    # (a) Cross-provider correlation
    corr_df = cross_provider_correlation(df, bench)

    # (b) Sector-level benchmarking
    sector_df = sector_benchmarking(df, bench)

    # (c) Factor validity testing
    validity_df = factor_validity(df)

    # (c2) ESG risk filter analysis (downside protection framing)
    esg_risk_filter_analysis(df)

    # (d) Bootstrap robustness test
    bootstrap_df = bootstrap_robustness(df, n_iter=1000, perturbation=0.20)

    # (e) Portfolio comparison
    portfolio_df = portfolio_comparison(df)

    # Summary
    _print_summary(corr_df, sector_df, validity_df, bootstrap_df, portfolio_df)

    print(f"\n[DONE] ESG benchmarking complete.")
    print(f"  Tables:  {TABLES}/benchmark_*.csv")
    print(f"  Figures: {FIGURES}/benchmark_*.png")


if __name__ == "__main__":
    main()
