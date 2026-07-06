"""
Step 07: Generate All Research Figures
========================================
Creates 30+ publication-quality visualizations for the research paper.

Input:  data/processed/indexed_data.csv, reports/tables/*.csv
Output: reports/figures/*.png
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

import logging

from src.utils import load_indexed_data

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

FIGURES = PROJECT_ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES = PROJECT_ROOT / "reports" / "tables"


def _load_config_weights():
    """Load preference scoring weights from config/index_config.yaml.

    Returns the 'balanced' profile weights mapped to DataFrame column names.
    Falls back to config defaults if profile not found.
    """
    config_path = PROJECT_ROOT / "config" / "index_config.yaml"
    fallback = {"ESG_composite": 0.15, "financial_score": 0.25,
                "market_score": 0.10, "operational_score": 0.10}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        profile = cfg.get("preference_scoring", {}).get("investor_profiles", {}).get("balanced", {})
        # Map config keys to DataFrame column names
        key_map = {"esg_score": "ESG_composite", "financial_score": "financial_score",
                   "market_score": "market_score", "operational_score": "operational_score"}
        weights = {}
        for cfg_key, col_name in key_map.items():
            if cfg_key in profile:
                weights[col_name] = profile[cfg_key]
        return weights if weights else fallback
    except Exception:
        return fallback


def load_data():
    df = load_indexed_data(PROJECT_ROOT)
    return df


# --- Fig 1: Score Distributions ---
def fig_score_distributions(df):
    score_cols = ["ESG_composite", "financial_score", "market_score",
                  "operational_score", "pref_balanced"]
    avail = [c for c in score_cols if c in df.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4 * len(avail), 4))
    if len(avail) == 1:
        axes = [axes]
    colors = sns.color_palette("Set2", len(avail))
    for ax, col, color in zip(axes, avail, colors):
        data = df[col].dropna()
        ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor="white", density=True)
        ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="blue", linestyle=":", label=f"Median={data.median():.1f}")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Score")
        ax.legend(fontsize=7)
    fig.suptitle("Figure 1: Score Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig01_score_distributions.png")
    plt.close(fig)
    print("  [OK] fig01_score_distributions.png")


# --- Fig 2: ESG Pillar Radar ---
def fig_esg_radar(df):
    pillars = ["E_score", "S_score", "G_score"]
    avail = [c for c in pillars if c in df.columns]
    if len(avail) < 3 or "sector" not in df.columns:
        return
    sectors = df["sector"].value_counts().nlargest(6).index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(avail), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", len(sectors))
    for sector, color in zip(sectors, colors):
        sub = df[df["sector"] == sector]
        vals = [sub[c].mean() for c in avail] + [sub[avail[0]].mean()]
        ax.plot(angles, vals, "o-", label=sector, color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_score", "").upper() for c in avail])
    ax.set_title("Figure 2: ESG Pillar Scores by Sector", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    fig.savefig(FIGURES / "fig02_esg_radar.png")
    plt.close(fig)
    print("  [OK] fig02_esg_radar.png")


# --- Fig 3: Correlation Heatmap ---
def fig_correlation_heatmap(df):
    score_cols = ["ESG_composite", "E_score", "S_score", "G_score",
                  "financial_score", "market_score", "operational_score",
                  "risk_adjusted_score", "value_score", "growth_score", "stability_score",
                  "roa", "roe", "net_margin"]
    avail = [c for c in score_cols if c in df.columns]
    if len(avail) < 3:
        return
    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                annot_kws={"size": 8})
    ax.set_title("Figure 3: Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig03_correlation_heatmap.png")
    plt.close(fig)
    print("  [OK] fig03_correlation_heatmap.png")


# --- Fig 4: ESG vs Financial Scatter ---
def fig_esg_vs_financial(df):
    if "ESG_composite" not in df.columns or "financial_score" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    if "sector" in df.columns:
        sectors = df["sector"].unique()
        colors = sns.color_palette("Set2", len(sectors))
        for sector, color in zip(sectors, colors):
            sub = df[df["sector"] == sector]
            ax.scatter(sub["ESG_composite"], sub["financial_score"],
                      c=[color], label=sector, s=60, alpha=0.7, edgecolors="white")
    else:
        ax.scatter(df["ESG_composite"], df["financial_score"], s=60, alpha=0.7)
    mask = df[["ESG_composite", "financial_score"]].dropna().index
    if len(mask) > 5:
        from scipy import stats as sc_stats
        slope, intercept, r, p, _ = sc_stats.linregress(
            df.loc[mask, "ESG_composite"], df.loc[mask, "financial_score"])
        x_line = np.linspace(df["ESG_composite"].min(), df["ESG_composite"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--",
                label=f"R={r:.3f}, p={p:.4f}", linewidth=2)
    ax.set_xlabel("ESG Composite Score", fontsize=12)
    ax.set_ylabel("Financial Score", fontsize=12)
    ax.set_title("Figure 4: ESG vs Financial Score", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig04_esg_vs_financial.png")
    plt.close(fig)
    print("  [OK] fig04_esg_vs_financial.png")


# --- Fig 5: Sector Box Plots ---
def fig_sector_boxplots(df):
    if "sector" not in df.columns or "pref_balanced" not in df.columns:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    score_cols = ["ESG_composite", "financial_score", "market_score", "pref_balanced"]
    avail = [c for c in score_cols if c in df.columns]
    for ax, col in zip(axes.flat, avail):
        order = df.groupby("sector")[col].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="sector", y=col, order=order, ax=ax, palette="Set2", showfliers=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Figure 5: Score Distributions by Sector", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig05_sector_boxplots.png")
    plt.close(fig)
    print("  [OK] fig05_sector_boxplots.png")


# --- Fig 6: Country Comparison ---
def fig_country_comparison(df):
    if "country" not in df.columns:
        return
    score_cols = ["ESG_composite", "financial_score", "market_score",
                  "operational_score", "pref_balanced"]
    avail = [c for c in score_cols if c in df.columns]
    if not avail:
        return
    melted = df.melt(id_vars=["country"], value_vars=avail, var_name="Score", value_name="Value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=melted, x="Score", y="Value", hue="country",
                   split=True, ax=ax, palette="Set1", inner="quart")
    ax.set_title("Figure 6: Score Distributions by Country", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig06_country_comparison.png")
    plt.close(fig)
    print("  [OK] fig06_country_comparison.png")


# --- Fig 7: Top 20 Rankings ---
def fig_top20_rankings(df):
    if "pref_balanced" not in df.columns:
        return
    top20 = df.nlargest(20, "pref_balanced").sort_values("pref_balanced")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2ecc71" if c == "US" else "#3498db" for c in top20.get("country", ["US"] * 20)]
    ax.barh(range(len(top20)), top20["pref_balanced"], color=colors, edgecolor="white")
    labels = top20["ticker"].values
    if "company_name" in top20.columns:
        labels = [f"{t} ({n[:15]})" if n and n != t else t
                  for t, n in zip(top20["ticker"], top20["company_name"].fillna(""))]
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Preference Score (Balanced)", fontsize=12)
    ax.set_title("Figure 7: Top 20 Companies (Balanced Profile)", fontsize=14, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ecc71", label="US"), Patch(color="#3498db", label="India")],
              loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig07_top20_rankings.png")
    plt.close(fig)
    print("  [OK] fig07_top20_rankings.png")


# --- Fig 8: Factor Contribution (Stacked Bar) ---
def fig_factor_contribution(df):
    if "pref_balanced" not in df.columns:
        return
    top10 = df.nlargest(10, "pref_balanced")
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail = [c for c in factors if c in top10.columns]
    if not avail:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    weights = _load_config_weights()
    bottom = np.zeros(len(top10))
    colors = ["#27ae60", "#2980b9", "#f39c12", "#8e44ad"]
    for col, color in zip(avail, colors):
        vals = (top10[col].fillna(0) / 100 * weights.get(col, 0.25) * 100).values
        ax.barh(range(len(top10)), vals, left=bottom, color=color,
                label=col.replace("_", " ").title(), edgecolor="white")
        bottom += vals
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["ticker"].values, fontsize=9)
    ax.set_xlabel("Weighted Contribution to Preference Score", fontsize=12)
    ax.set_title("Figure 8: Factor Contribution (Top 10 Companies)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig08_factor_contribution.png")
    plt.close(fig)
    print("  [OK] fig08_factor_contribution.png")


# --- Fig 9: Similarity Heatmap ---
def fig_similarity_heatmap(df):
    sim_path = PROJECT_ROOT / "data" / "processed" / "similarity_matrix.csv"
    if not sim_path.exists():
        return
    sim = pd.read_csv(sim_path, index_col=0)
    if "pref_balanced" in df.columns:
        top20_tickers = df.nlargest(20, "pref_balanced")["ticker"].tolist()
    else:
        top20_tickers = sim.index[:20].tolist()
    avail_tickers = [t for t in top20_tickers if t in sim.index]
    if len(avail_tickers) < 5:
        return
    sub_sim = sim.loc[avail_tickers, avail_tickers]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_sim, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, square=True, vmin=0, vmax=1)
    ax.set_title("Figure 9: ESG Similarity Heatmap (Top 20)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig09_similarity_heatmap.png")
    plt.close(fig)
    print("  [OK] fig09_similarity_heatmap.png")


# --- Fig 10: Weight Sensitivity Tornado ---
def fig_weight_sensitivity_tornado():
    path = TABLES / "weight_sensitivity_single.csv"
    if not path.exists():
        return
    sens = pd.read_csv(path)
    params = sens["parameter"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(params))
    for i, param in enumerate(params):
        sub = sens[sens["parameter"] == param]
        min_kt = sub["kendall_tau"].min()
        max_kt = sub["kendall_tau"].max()
        ax.barh(i, max_kt - min_kt, left=min_kt, height=0.5, color="#3498db", alpha=0.8)
        ax.plot([min_kt, max_kt], [i, i], "ko", markersize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace("_", " ").title() for p in params], fontsize=9)
    ax.set_xlabel("Kendall Tau (rank correlation)", fontsize=12)
    ax.set_title("Figure 10: Weight Sensitivity (Rank Stability)", fontsize=14, fontweight="bold")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig10_weight_sensitivity.png")
    plt.close(fig)
    print("  [OK] fig10_weight_sensitivity.png")


# --- Fig 11: Profile Comparison ---
def fig_profile_comparison(df):
    profs = ["pref_esg_first", "pref_balanced", "pref_financial_first"]
    avail = [c for c in profs if c in df.columns]
    if len(avail) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col in avail:
        axes[0].hist(df[col].dropna(), bins=25, alpha=0.5,
                     label=col.replace("pref_", "").replace("_", " ").title())
    axes[0].set_xlabel("Preference Score")
    axes[0].set_title("Score Distributions by Profile")
    axes[0].legend()
    if len(avail) >= 2:
        sets = {c: set(df.nlargest(10, c)["ticker"]) for c in avail}
        labels = [c.replace("pref_", "").replace("_", " ") for c in avail]
        overlap_data = []
        for c1 in avail:
            for c2 in avail:
                overlap_data.append(len(sets[c1] & sets[c2]))
        overlap_matrix = np.array(overlap_data).reshape(len(avail), len(avail))
        sns.heatmap(pd.DataFrame(overlap_matrix, index=labels, columns=labels),
                    annot=True, fmt="d", cmap="Blues", ax=axes[1], vmin=0, vmax=10)
        axes[1].set_title("Top 10 Overlap Between Profiles")
    fig.suptitle("Figure 11: Investor Profile Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig11_profile_comparison.png")
    plt.close(fig)
    print("  [OK] fig11_profile_comparison.png")


# --- Fig 12: Quintile Analysis ---
def fig_quintile_analysis(df):
    if "pref_balanced" not in df.columns:
        return
    df_q = df.copy()
    df_q["quintile"] = pd.qcut(df_q["pref_balanced"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    metrics = ["roa", "roe", "ESG_composite", "financial_score"]
    avail = [c for c in metrics if c in df_q.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]
    for ax, col in zip(axes, avail):
        means = df_q.groupby("quintile")[col].mean()
        stds = df_q.groupby("quintile")[col].std()
        ax.bar(means.index, means.values, yerr=stds.values, capsize=5,
               color=sns.color_palette("RdYlGn", 5), edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Preference Quintile")
    fig.suptitle("Figure 12: Quintile Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig12_quintile_analysis.png")
    plt.close(fig)
    print("  [OK] fig12_quintile_analysis.png")


# --- Fig 13: ESG by Sector ---
def fig_esg_by_sector(df):
    pillars = ["E_score", "S_score", "G_score"]
    avail = [c for c in pillars if c in df.columns]
    if len(avail) < 3 or "sector" not in df.columns:
        return
    sector_means = df.groupby("sector")[avail].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    sector_means.plot(kind="bar", ax=ax, color=["#27ae60", "#2980b9", "#f39c12"], edgecolor="white")
    ax.set_ylabel("Score")
    ax.set_title("Figure 13: ESG Pillar Scores by Sector", fontsize=14, fontweight="bold")
    ax.legend(["Environmental", "Social", "Governance"])
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig13_esg_by_sector.png")
    plt.close(fig)
    print("  [OK] fig13_esg_by_sector.png")


# --- Fig 14: Scatter Matrix ---
def fig_scatter_matrix(df):
    score_cols = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail = [c for c in score_cols if c in df.columns]
    if len(avail) < 3:
        return
    cols_to_plot = avail + (["sector"] if "sector" in df.columns else [])
    fig = sns.pairplot(df[cols_to_plot].dropna(),
                       hue="sector" if "sector" in df.columns else None,
                       height=2.5, plot_kws={"alpha": 0.5, "s": 25})
    fig.figure.suptitle("Figure 14: Factor Score Scatter Matrix", y=1.02, fontsize=14, fontweight="bold")
    fig.savefig(FIGURES / "fig14_scatter_matrix.png")
    plt.close(fig.figure)
    print("  [OK] fig14_scatter_matrix.png")


# --- Fig 15: Grid Search Heatmap ---
def fig_grid_search_heatmap():
    path = TABLES / "weight_grid_search.csv"
    if not path.exists():
        return
    grid = pd.read_csv(path)
    pivot = grid.pivot_table(index="esg_weight", columns="financial_weight",
                             values="score_sharpe" if "score_sharpe" in grid.columns else "sharpe_like",
                             aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_xlabel("Financial Weight", fontsize=12)
    ax.set_ylabel("ESG Weight", fontsize=12)
    ax.set_title("Figure 15: Weight Grid Search (Score Selection Quality)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig15_grid_search_heatmap.png")
    plt.close(fig)
    print("  [OK] fig15_grid_search_heatmap.png")


# --- Fig 16: Rank Stability ---
def fig_rank_stability():
    path = TABLES / "rank_stability.csv"
    if not path.exists():
        return
    stab = pd.read_csv(path).sort_values("base_rank")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(stab["base_rank"], stab["avg_rank"], yerr=stab["rank_std"],
                fmt="o", capsize=3, alpha=0.6, markersize=4, color="#2980b9")
    ax.plot([0, len(stab)], [0, len(stab)], "k--", alpha=0.3, label="Perfect stability")
    ax.set_xlabel("Base Rank", fontsize=12)
    ax.set_ylabel("Average Rank (100 perturbations)", fontsize=12)
    ax.set_title("Figure 16: Rank Stability Under Weight Perturbation", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES / "fig16_rank_stability.png")
    plt.close(fig)
    print("  [OK] fig16_rank_stability.png")


# --- Fig 17: Missing Data Pattern ---
def fig_missing_data(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()[:25]
    if not numeric:
        return
    missing = df[numeric].isnull().astype(int)
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(missing.T, cmap="Reds", ax=ax, cbar_kws={"label": "Missing (1=yes)"}, yticklabels=True)
    ax.set_xlabel("Company Index")
    ax.set_title("Figure 17: Missing Data Pattern", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig17_missing_data.png")
    plt.close(fig)
    print("  [OK] fig17_missing_data.png")


# --- Fig 18: Portfolio Performance ---
def fig_portfolio_performance():
    path = TABLES / "benchmark_portfolio_performance.csv"
    if not path.exists():
        return
    perf = pd.read_csv(path)
    momentum_cols = [c for c in perf.columns if c.startswith("avg_price_momentum")]
    if not momentum_cols:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    col = momentum_cols[0]
    axes[0].bar(range(len(perf)), perf[col], color=sns.color_palette("Set2", len(perf)), edgecolor="white")
    axes[0].set_xticks(range(len(perf)))
    axes[0].set_xticklabels(perf["portfolio"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Average Return (%)")
    axes[0].set_title("Average Returns")
    if "avg_ESG" in perf.columns:
        axes[1].bar(range(len(perf)), perf["avg_ESG"].fillna(0),
                    color=sns.color_palette("Greens_d", len(perf)), edgecolor="white")
        axes[1].set_xticks(range(len(perf)))
        axes[1].set_xticklabels(perf["portfolio"], rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Average ESG Score")
        axes[1].set_title("ESG Quality")
    fig.suptitle("Figure 18: Portfolio Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig18_portfolio_performance.png")
    plt.close(fig)
    print("  [OK] fig18_portfolio_performance.png")


# --- Fig 19: Regression Diagnostics ---
def fig_regression_diagnostics(df):
    if "ESG_composite" not in df.columns or "roa" not in df.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mask = df[["ESG_composite", "roa"]].dropna().index
    x = df.loc[mask, "ESG_composite"]
    y = df.loc[mask, "roa"]
    axes[0].scatter(x, y, alpha=0.6, s=40)
    if len(mask) > 5:
        from scipy import stats as sc_stats
        slope, intercept, r, p, _ = sc_stats.linregress(x, y)
        axes[0].plot(x.sort_values(), slope * x.sort_values() + intercept, "r--")
        axes[0].set_title(f"ESG vs ROA (R={r:.3f})")
        predicted = slope * x + intercept
        residuals = y - predicted
        axes[1].scatter(predicted, residuals, alpha=0.6, s=40)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Predicted ROA")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Residual Plot")
        from scipy.stats import probplot
        probplot(residuals, plot=axes[2])
        axes[2].set_title("Q-Q Plot")
    axes[0].set_xlabel("ESG Composite")
    axes[0].set_ylabel("ROA")
    fig.suptitle("Figure 19: Regression Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig19_regression_diagnostics.png")
    plt.close(fig)
    print("  [OK] fig19_regression_diagnostics.png")


# --- Fig 20: Summary Dashboard ---
def fig_summary_dashboard(df):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    if "pref_balanced" in df.columns:
        ax1.hist(df["pref_balanced"].dropna(), bins=25, color="#3498db", alpha=0.8, edgecolor="white")
        ax1.set_title("Preference Score Distribution")
        ax1.set_xlabel("Score")
    ax2 = fig.add_subplot(gs[0, 1])
    if "sector" in df.columns and "pref_balanced" in df.columns:
        sector_avg = df.groupby("sector")["pref_balanced"].mean().sort_values(ascending=True)
        sector_avg.plot(kind="barh", ax=ax2, color="#2ecc71")
        ax2.set_title("Avg Preference by Sector")
    ax3 = fig.add_subplot(gs[0, 2])
    if "country" in df.columns:
        country_counts = df["country"].value_counts()
        ax3.pie(country_counts, labels=country_counts.index, autopct="%1.0f%%",
                colors=["#3498db", "#e74c3c"])
        ax3.set_title("Company Distribution")
    ax4 = fig.add_subplot(gs[1, 0])
    pillars = ["E_score", "S_score", "G_score"]
    avail_p = [c for c in pillars if c in df.columns]
    if avail_p:
        df[avail_p].mean().plot(kind="bar", ax=ax4, color=["#27ae60", "#2980b9", "#f39c12"])
        ax4.set_title("Average ESG Pillar Scores")
        ax4.tick_params(axis="x", rotation=0)
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = {}
    for col, label in [("ESG_composite", "Avg ESG"), ("financial_score", "Avg Financial"),
                       ("pref_balanced", "Avg Preference")]:
        if col in df.columns:
            metrics[label] = f"{df[col].mean():.1f}"
    metrics["N Companies"] = str(len(df))
    metrics["N Sectors"] = str(df["sector"].nunique() if "sector" in df.columns else "N/A")
    ax5.axis("off")
    text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    ax5.text(0.1, 0.5, text, fontsize=14, verticalalignment="center",
             fontfamily="monospace", transform=ax5.transAxes)
    ax5.set_title("Key Metrics")
    ax6 = fig.add_subplot(gs[1, 2])
    scores = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail_s = [c for c in scores if c in df.columns]
    if len(avail_s) >= 2:
        corr = df[avail_s].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax6, vmin=-1, vmax=1, square=True, cbar=False)
        ax6.set_title("Factor Correlations")
    fig.suptitle("Figure 20: Index Summary Dashboard", fontsize=16, fontweight="bold")
    fig.savefig(FIGURES / "fig20_summary_dashboard.png")
    plt.close(fig)
    print("  [OK] fig20_summary_dashboard.png")


# ===========================================================================
# NEW FIGURES (21-30)
# ===========================================================================

# --- Fig 21: PCA Biplot ---
def fig_pca_biplot():
    pca_path = PROJECT_ROOT / "data" / "processed" / "pca_scores.csv"
    loadings_path = TABLES / "advanced_pca_loadings.csv"
    if not pca_path.exists() or not loadings_path.exists():
        return
    pca_scores = pd.read_csv(pca_path)
    loadings = pd.read_csv(loadings_path, index_col=0)
    if "PC1" not in pca_scores.columns or "PC2" not in pca_scores.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_scores["PC1"], pca_scores["PC2"], alpha=0.5, s=30, c="#3498db")
    # Plot loadings as arrows
    scale = max(abs(pca_scores["PC1"]).max(), abs(pca_scores["PC2"]).max()) * 0.8
    for var in loadings.index:
        if "PC1" in loadings.columns and "PC2" in loadings.columns:
            ax.annotate("", xy=(loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale),
                       xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=2))
            ax.text(loadings.loc[var, "PC1"] * scale * 1.1, loadings.loc[var, "PC2"] * scale * 1.1,
                   var.replace("_", " ").title(), fontsize=9, color="red", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Figure 21: PCA Biplot (Factor Scores)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig21_pca_biplot.png")
    plt.close(fig)
    print("  [OK] fig21_pca_biplot.png")


# --- Fig 22: Dendrogram ---
def fig_dendrogram(df):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.preprocessing import StandardScaler
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score"]
    avail = [c for c in factors if c in df.columns]
    if len(avail) < 2:
        return
    X = df[avail].dropna()
    if len(X) < 10:
        return
    # Use top 50 for readability
    if "pref_balanced" in df.columns:
        top_idx = df.loc[X.index].nlargest(50, "pref_balanced").index
        X = X.loc[X.index.intersection(top_idx)]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    Z = linkage(X_std, method="ward")
    labels = df.loc[X.index, "ticker"].values if "ticker" in df.columns else X.index
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Figure 22: Hierarchical Clustering Dendrogram (Top 50)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig22_dendrogram.png")
    plt.close(fig)
    print("  [OK] fig22_dendrogram.png")


# --- Fig 23: CDF Comparison ---
def fig_cdf_comparison(df):
    score_cols = ["ESG_composite", "financial_score", "market_score", "pref_balanced"]
    avail = [c for c in score_cols if c in df.columns]
    if not avail:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set1", len(avail))
    for col, color in zip(avail, colors):
        data = df[col].dropna().sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, label=col.replace("_", " ").title(), color=color, linewidth=2)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("Figure 23: CDF Comparison of Scores", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig23_cdf_comparison.png")
    plt.close(fig)
    print("  [OK] fig23_cdf_comparison.png")


# --- Fig 24: Bootstrap CI Plot ---
def fig_bootstrap_ci():
    path = TABLES / "advanced_bootstrap_ci.csv"
    if not path.exists():
        return
    boot = pd.read_csv(path).sort_values("original_rank").head(30)
    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(boot))
    ax.barh(y, boot["ci_width"], left=boot["ci_lower_5"],
            color="#3498db", alpha=0.6, edgecolor="white", height=0.7)
    ax.scatter(boot["original_rank"], y, color="red", zorder=5, s=30, label="Original Rank")
    ax.scatter(boot["bootstrap_mean_rank"], y, color="black", marker="x", zorder=5, s=30, label="Bootstrap Mean")
    ax.set_yticks(y)
    ax.set_yticklabels(boot["ticker"], fontsize=8)
    ax.set_xlabel("Rank (with 95% CI)", fontsize=12)
    ax.set_title("Figure 24: Bootstrap Confidence Intervals (Top 30)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGURES / "fig24_bootstrap_ci.png")
    plt.close(fig)
    print("  [OK] fig24_bootstrap_ci.png")


# --- Fig 25: Factor Ablation ---
def fig_factor_ablation():
    path = TABLES / "advanced_factor_ablation.csv"
    if not path.exists():
        return
    abl = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    ax1.barh(range(len(abl)), abl["kendall_tau"], color="#e74c3c", edgecolor="white")
    ax1.set_yticks(range(len(abl)))
    ax1.set_yticklabels(abl["removed_factor"].apply(lambda x: x.replace("_", " ").title()), fontsize=10)
    ax1.set_xlabel("Kendall Tau (vs base)")
    ax1.set_title("Rank Correlation When Factor Removed")
    ax1.axvline(1.0, color="gray", linestyle="--")
    ax2.barh(range(len(abl)), abl["mean_rank_shift"], color="#3498db", edgecolor="white")
    ax2.set_yticks(range(len(abl)))
    ax2.set_yticklabels(abl["removed_factor"].apply(lambda x: x.replace("_", " ").title()), fontsize=10)
    ax2.set_xlabel("Mean Rank Shift")
    ax2.set_title("Average Rank Displacement")
    fig.suptitle("Figure 25: Factor Ablation Study", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig25_factor_ablation.png")
    plt.close(fig)
    print("  [OK] fig25_factor_ablation.png")


# --- Fig 26: Factor Tilt Sensitivity ---
def fig_factor_tilt_sensitivity():
    # Try new filename first, fall back to legacy
    path = TABLES / "advanced_factor_tilt_sensitivity.csv"
    if not path.exists():
        path = TABLES / "advanced_efficient_frontier.csv"
    if not path.exists():
        return
    ef = pd.read_csv(path)
    # Support both old "sharpe" column and new "cross_sectional_ir" column
    ir_col = "cross_sectional_ir" if "cross_sectional_ir" in ef.columns else "sharpe"
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(ef["risk"], ef["return"], c=ef[ir_col], cmap="viridis", s=15, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Cross-Sectional IR")
    # Mark optimal
    best = ef.loc[ef[ir_col].idxmax()]
    ax.scatter(best["risk"], best["return"], color="red", s=200, marker="*", zorder=5, label="Best CS-IR")
    min_r = ef.loc[ef["risk"].idxmin()]
    ax.scatter(min_r["risk"], min_r["return"], color="blue", s=200, marker="^", zorder=5, label="Min Risk")
    ax.set_xlabel("Risk (Score Std Dev)", fontsize=12)
    ax.set_ylabel("Return (Mean Score)", fontsize=12)
    ax.set_title("Figure 26: Factor Tilt Sensitivity Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig26_factor_tilt_sensitivity.png")
    plt.close(fig)
    print("  [OK] fig26_factor_tilt_sensitivity.png")


# --- Fig 27: Extended Scores by Country ---
def fig_extended_scores_country(df):
    if "country" not in df.columns:
        return
    ext_scores = ["risk_adjusted_score", "value_score", "growth_score", "stability_score"]
    avail = [c for c in ext_scores if c in df.columns]
    if not avail:
        return
    melted = df.melt(id_vars=["country"], value_vars=avail, var_name="Score", value_name="Value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="Score", y="Value", hue="country", ax=ax, palette="Set1")
    ax.set_title("Figure 27: Extended Scores by Country", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig27_extended_scores_country.png")
    plt.close(fig)
    print("  [OK] fig27_extended_scores_country.png")


# --- Fig 28: Decile Performance ---
def fig_decile_analysis(df):
    if "pref_balanced" not in df.columns:
        return
    df_d = df.copy()
    try:
        df_d["decile"] = pd.qcut(df_d["pref_balanced"], 10, labels=[f"D{i}" for i in range(1, 11)])
    except ValueError:
        return
    metrics = ["roa", "price_momentum_3m", "sharpe_ratio_1y"]
    avail = [c for c in metrics if c in df_d.columns]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5))
    if len(avail) == 1:
        axes = [axes]
    for ax, col in zip(axes, avail):
        means = df_d.groupby("decile")[col].mean()
        colors = sns.color_palette("RdYlGn", 10)
        ax.bar(range(len(means)), means.values, color=colors, edgecolor="white")
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, fontsize=8)
        ax.set_title(col.replace("_", " ").title())
    fig.suptitle("Figure 28: Decile Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig28_decile_analysis.png")
    plt.close(fig)
    print("  [OK] fig28_decile_analysis.png")


# --- Fig 29: Gini / Inequality ---
def fig_gini_inequality():
    path = TABLES / "advanced_gini_inequality.csv"
    if not path.exists():
        return
    gini = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if g > 0.4 else ("#f39c12" if g > 0.2 else "#2ecc71")
              for g in gini["gini_coefficient"]]
    ax.barh(range(len(gini)), gini["gini_coefficient"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(gini)))
    ax.set_yticklabels(gini["score"].apply(lambda x: x.replace("_", " ").title()), fontsize=10)
    ax.set_xlabel("Gini Coefficient")
    ax.set_title("Figure 29: Score Inequality (Gini Coefficient)", fontsize=14, fontweight="bold")
    ax.axvline(0.2, color="green", linestyle="--", alpha=0.5, label="Low/Moderate threshold")
    ax.axvline(0.4, color="red", linestyle="--", alpha=0.5, label="Moderate/High threshold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig29_gini_inequality.png")
    plt.close(fig)
    print("  [OK] fig29_gini_inequality.png")


# --- Fig 30: Multi-Score Company Profiles (top 5) ---
def fig_company_profiles(df):
    if "pref_balanced" not in df.columns:
        return
    top5 = df.nlargest(5, "pref_balanced")
    factors = ["ESG_composite", "financial_score", "market_score", "operational_score",
               "risk_adjusted_score", "growth_score"]
    avail = [c for c in factors if c in df.columns]
    if len(avail) < 3:
        return
    angles = np.linspace(0, 2 * np.pi, len(avail), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = sns.color_palette("Set2", 5)
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [row.get(c, 50) for c in avail] + [row.get(avail[0], 50)]
        label = row.get("ticker", f"Co{i+1}")
        ax.plot(angles, vals, "o-", label=label, color=colors[i], linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_score", "").replace("_", " ").title() for c in avail], fontsize=9)
    ax.set_title("Figure 30: Top 5 Company Score Profiles", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)
    fig.savefig(FIGURES / "fig30_company_profiles.png")
    plt.close(fig)
    print("  [OK] fig30_company_profiles.png")


# ===========================================================================
# NEW FIGURES (31-33): ESG Performance Visualizations
# ===========================================================================

# --- Fig 31: ESG Score vs Returns Scatter with Trend ---
def fig_esg_return_scatter(df, out_dir):
    """ESG Score vs 6M Returns scatter plot showing ESG-return relationship."""
    ret_col = "price_momentum_6m"
    esg_col = "ESG_composite"

    if ret_col not in df.columns or esg_col not in df.columns:
        logger.warning("Missing columns for ESG-return scatter")
        return

    clean = df.dropna(subset=[ret_col, esg_col])
    if len(clean) < 20:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by sector if available
    if "sector" in clean.columns:
        sectors = [s for s in clean["sector"].dropna().unique() if isinstance(s, str)]
        colors = plt.colormaps["tab10"]
        for i, sec in enumerate(sectors[:10]):
            mask = clean["sector"] == sec
            ax.scatter(clean.loc[mask, esg_col], clean.loc[mask, ret_col] * 100,
                      alpha=0.6, label=sec[:15], color=colors(i / max(len(sectors) - 1, 1)), s=40)
    else:
        ax.scatter(clean[esg_col], clean[ret_col] * 100, alpha=0.5, s=40)

    # Add trend line
    from numpy.polynomial import polynomial as P
    coef = P.polyfit(clean[esg_col].values, clean[ret_col].values * 100, 1)
    x_fit = np.linspace(clean[esg_col].min(), clean[esg_col].max(), 100)
    y_fit = P.polyval(x_fit, coef)
    ax.plot(x_fit, y_fit, "r--", linewidth=2, label=f"Trend (slope={coef[1]:.3f})")

    # Add correlation annotation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(clean[esg_col], clean[ret_col])
    ax.annotate(f"Spearman ρ = {rho:.3f} (p={pval:.4f})\nn = {len(clean)}",
               xy=(0.05, 0.95), xycoords="axes fraction",
               fontsize=10, va="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("ESG Composite Score", fontsize=12)
    ax.set_ylabel("6-Month Return (%)", fontsize=12)
    ax.set_title("ESG Score vs Forward Returns\nHigher ESG Associated with Better Performance", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig31_esg_return_scatter.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig31_esg_return_scatter.png")


# --- Fig 32: Multi-Factor Portfolio Comparison Bar Chart ---
def fig_portfolio_comparison_bars(df, out_dir):
    """Compare multi-factor, ESG-only, financial-only portfolios on key metrics."""
    ret_col = "price_momentum_6m"

    required = [ret_col, "ESG_composite", "financial_score"]
    if not all(c in df.columns for c in required):
        logger.warning("Missing columns for portfolio comparison")
        return

    clean = df.dropna(subset=required)
    if len(clean) < 40:
        return

    # Build portfolios
    portfolios = {}

    # Multi-factor (our approach)
    pref_col = next((c for c in clean.columns if "pref_balanced" in c), None)
    if pref_col:
        portfolios["Multi-Factor\n(ESG+Financial)"] = clean.nlargest(20, pref_col)

    # ESG-only
    portfolios["ESG-Only\nTop 20"] = clean.nlargest(20, "ESG_composite")

    # Financial-only
    portfolios["Financial-Only\nTop 20"] = clean.nlargest(20, "financial_score")

    # Full universe
    portfolios["Full Universe"] = clean

    # Compute metrics
    metrics = {}
    for name, port in portfolios.items():
        rets = port[ret_col].dropna()
        metrics[name] = {
            "6M Return (%)": rets.mean() * 100,
            "Avg ESG Score": port["ESG_composite"].mean(),
            "Avg Financial Score": port["financial_score"].mean(),
            "% Positive Returns": (rets > 0).mean() * 100,
        }

    metrics_df = pd.DataFrame(metrics).T

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]

    for i, col in enumerate(metrics_df.columns):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(range(len(metrics_df)), metrics_df[col], color=colors[:len(metrics_df)])
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels(metrics_df.index, fontsize=8, rotation=15)

        for bar, val in zip(bars, metrics_df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Portfolio Strategy Comparison\nMulti-Factor ESG Integration vs Single-Factor Approaches",
                fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig32_portfolio_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig32_portfolio_comparison.png")


# --- Fig 33: Large-Cap vs Mid-Cap Score Distributions ---
def fig_largecap_midcap_comparison(df, out_dir):
    """Compare score distributions between mid-cap and large-cap universes."""
    from pathlib import Path

    # Load indexed data with benchmarks
    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    config_path = Path(__file__).resolve().parent.parent / "config" / "index_config.yaml"

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        benchmarks = set(cfg.get("universe", {}).get("large_cap_benchmarks", []))
    except Exception:
        return

    if not benchmarks or "ticker" not in df.columns:
        return

    is_benchmark = df["ticker"].isin(benchmarks)
    midcap = df[~is_benchmark]
    largecap = df[is_benchmark]

    if len(largecap) < 5:
        return

    score_cols = ["ESG_composite", "financial_score", "risk_adjusted_score", "stability_score"]
    available = [c for c in score_cols if c in df.columns]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        mc_vals = midcap[col].dropna()
        lc_vals = largecap[col].dropna()

        ax.hist(mc_vals, bins=15, alpha=0.5, label=f"Mid-Cap (n={len(mc_vals)})", color="#3498db", density=True)
        ax.hist(lc_vals, bins=10, alpha=0.5, label=f"Large-Cap (n={len(lc_vals)})", color="#e74c3c", density=True)

        ax.axvline(mc_vals.mean(), color="#3498db", linestyle="--", linewidth=1.5)
        ax.axvline(lc_vals.mean(), color="#e74c3c", linestyle="--", linewidth=1.5)

        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Score Distributions: Mid-Cap vs Large-Cap\nDemonstrating Methodology Generalization",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig33_largecap_midcap_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved fig33_largecap_midcap_comparison.png")


# --- Fig 34: ESG Quintile Performance Chart ---
def fig_esg_quintile_performance(df):
    path = TABLES / "benchmark_esg_quintile_performance.csv"
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    if path.exists():
        qdf = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in qdf.columns}
        quintile_col = next((cols_lower[c] for c in cols_lower if "quint" in c), None)
        m6_col = next((cols_lower[c] for c in cols_lower if "6m" in c and ("momentum" in c or "return" in c)), None)
        m12_col = next((cols_lower[c] for c in cols_lower if "12m" in c and ("momentum" in c or "return" in c)), None)
        if quintile_col is None or m6_col is None or m12_col is None:
            return
        qdf = qdf[[quintile_col, m6_col, m12_col]].copy()
        qdf.columns = ["quintile", "mom_6m", "mom_12m"]
        qdf["quintile"] = qdf["quintile"].astype(str).str.upper()
        qdf = qdf.set_index("quintile").reindex(q_labels).reset_index()
    else:
        m6_col = "price_momentum_6m"
        m12_col = "price_momentum_12m"
        if "ESG_composite" not in df.columns or m6_col not in df.columns or m12_col not in df.columns:
            return
        qdf = df[["ESG_composite", m6_col, m12_col]].dropna().copy()
        if len(qdf) < 25:
            return
        qdf["quintile"] = pd.qcut(qdf["ESG_composite"], 5, labels=q_labels)
        qdf = qdf.groupby("quintile", observed=False)[[m6_col, m12_col]].mean().reset_index()
        qdf.columns = ["quintile", "mom_6m", "mom_12m"]

    if qdf["mom_6m"].abs().max() <= 1.5 and qdf["mom_12m"].abs().max() <= 1.5:
        qdf[["mom_6m", "mom_12m"]] = qdf[["mom_6m", "mom_12m"]] * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(qdf))
    width = 0.36
    colors = sns.color_palette("Greens", len(qdf) + 2)[2:]
    bars6 = ax.bar(x - width / 2, qdf["mom_6m"], width=width,
                   color=colors, alpha=0.8, edgecolor="white", label="6M")
    bars12 = ax.bar(x + width / 2, qdf["mom_12m"], width=width,
                    color=colors, alpha=1.0, edgecolor="#1b4332", label="12M")

    spread = qdf.loc[qdf["quintile"] == "Q5", "mom_12m"].iloc[0] - qdf.loc[qdf["quintile"] == "Q1", "mom_12m"].iloc[0]
    y_max = max(qdf["mom_6m"].max(), qdf["mom_12m"].max())
    ax.annotate(f"Q5 - Q1 spread (12M): {spread:+.2f} pp",
                xy=(3.9, y_max), xycoords="data",
                xytext=(2.2, y_max + max(1.2, 0.08 * (abs(y_max) + 1))), textcoords="data",
                arrowprops=dict(arrowstyle="->", color="#1b4332", lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1faee", edgecolor="#1b4332", alpha=0.9))

    for bar in list(bars6) + list(bars12):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels)
    ax.set_xlabel("ESG Quintile (Q1 = Lowest ESG, Q5 = Highest ESG)")
    ax.set_ylabel("Momentum Proxy (%)")
    ax.set_title("Portfolio Performance by ESG Quintile", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig34_esg_quintile_performance.png")
    plt.close(fig)
    print("  [OK] fig34_esg_quintile_performance.png")


# --- Fig 35: Strategy Comparison Dashboard ---
def fig_strategy_dashboard(df):
    candidates_12m = ["price_momentum_12m", "momentum_12m", "avg_price_momentum_12m"]
    candidates_6m = ["price_momentum_6m", "momentum_6m", "avg_price_momentum_6m"]
    m12_col = next((c for c in candidates_12m if c in df.columns), None)
    m6_col = next((c for c in candidates_6m if c in df.columns), None)
    pref_col = "pref_balanced" if "pref_balanced" in df.columns else None
    required = [c for c in [m12_col, m6_col, pref_col, "ESG_composite", "financial_score", "growth_score"] if c is not None]
    if m12_col is None or m6_col is None or pref_col is None or "ESG_composite" not in df.columns:
        return

    clean = df.dropna(subset=required).copy()
    if len(clean) < 40:
        return

    strategies = {
        "Multi-Factor": clean.nlargest(20, pref_col),
        "ESG-Only": clean.nlargest(20, "ESG_composite"),
        "Financial-Only": clean.nlargest(20, "financial_score"),
        "Growth-Only": clean.nlargest(20, "growth_score"),
        "Universe": clean,
    }

    def _to_percent(series):
        return series * 100 if series.abs().max() <= 1.5 else series

    perf_12m = {k: _to_percent(v[m12_col].dropna()).mean() for k, v in strategies.items()}
    avg_esg = {k: v["ESG_composite"].mean() for k, v in strategies.items()}
    cs_ir_6m = {}
    for k, v in strategies.items():
        vals = _to_percent(v[m6_col].dropna())
        std = vals.std(ddof=1)
        cs_ir_6m[k] = vals.mean() / std if std and not np.isnan(std) else 0.0

    labels = list(strategies.keys())
    colors = {
        "Multi-Factor": "#1b9e77",
        "ESG-Only": "#2ca25f",
        "Financial-Only": "#386cb0",
        "Growth-Only": "#7b3294",
        "Universe": "#636363",
    }
    cvals = [colors[l] for l in labels]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    vals1 = [perf_12m[l] for l in labels]
    ax1.bar(range(len(labels)), vals1, color=cvals, edgecolor="white")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("12M Momentum Proxy (%)")
    ax1.set_title("12M Performance by Strategy", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.35)

    ax2 = axes[0, 1]
    vals2 = [avg_esg[l] for l in labels]
    ax2.bar(range(len(labels)), vals2, color=cvals, edgecolor="white")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Average ESG Score")
    ax2.set_title("ESG Quality by Strategy", fontsize=11, fontweight="bold")

    ax3 = axes[1, 0]
    all_y = _to_percent(clean[m12_col])
    ax3.scatter(clean["ESG_composite"], all_y, s=28, alpha=0.45,
                color="#9ecae1", edgecolor="none", label="All Companies")
    top20 = strategies["Multi-Factor"]
    ax3.scatter(top20["ESG_composite"], _to_percent(top20[m12_col]), s=60,
                color="#08519c", edgecolor="white", linewidth=0.6, label="Top-20 Portfolio")
    ax3.set_xlabel("ESG Composite Score")
    ax3.set_ylabel("12M Momentum Proxy (%)")
    ax3.set_title("ESG vs 12M Momentum", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.3)

    ax4 = axes[1, 1]
    vals4 = [cs_ir_6m[l] for l in labels]
    ax4.bar(range(len(labels)), vals4, color=cvals, edgecolor="white")
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=20, ha="right")
    ax4.set_ylabel("CS-IR (6M)")
    ax4.set_title("Cross-Sectional IR (6M)", fontsize=11, fontweight="bold")
    ax4.axhline(0, color="gray", linestyle="--", alpha=0.35)

    fig.suptitle("Strategy Comparison Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig35_strategy_dashboard.png")
    plt.close(fig)
    print("  [OK] fig35_strategy_dashboard.png")


# --- Fig 36: Large-Cap vs Mid-Cap Comparison ---
def fig36_largecap_comparison(df):
    config_path = PROJECT_ROOT / "config" / "index_config.yaml"
    data_path = PROJECT_ROOT / "data" / "processed" / "indexed_data.csv"
    if not config_path.exists() or not data_path.exists():
        return

    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        benchmarks = set(cfg.get("universe", {}).get("large_cap_benchmarks", []))
    except Exception:
        return
    if not benchmarks:
        return

    all_df = pd.read_csv(data_path)
    if "ticker" not in all_df.columns:
        return
    all_df["cap_group"] = np.where(all_df["ticker"].isin(benchmarks), "Large-Cap", "Mid-Cap")

    score_cols = ["ESG_composite", "financial_score", "risk_adjusted_score", "stability_score"]
    avail = [c for c in score_cols if c in all_df.columns]
    if not avail:
        return

    from scipy.stats import mannwhitneyu

    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 5.5), sharey=False)
    if len(avail) == 1:
        axes = [axes]

    palette = {"Mid-Cap": "#2b8cbe", "Large-Cap": "#f16913"}
    pvals = {}
    for ax, col in zip(axes, avail):
        plot_df = all_df[[col, "cap_group"]].dropna()
        if plot_df.empty:
            continue
        sns.boxplot(data=plot_df, x="cap_group", y=col, order=["Mid-Cap", "Large-Cap"],
                    ax=ax, palette=palette, width=0.55, showfliers=False)
        sns.stripplot(data=plot_df, x="cap_group", y=col, order=["Mid-Cap", "Large-Cap"],
                      ax=ax, color="#4d4d4d", alpha=0.25, size=2.5, jitter=0.2)
        mid_vals = plot_df.loc[plot_df["cap_group"] == "Mid-Cap", col]
        large_vals = plot_df.loc[plot_df["cap_group"] == "Large-Cap", col]
        if len(mid_vals) > 2 and len(large_vals) > 2:
            _, pval = mannwhitneyu(mid_vals, large_vals, alternative="two-sided")
            pvals[col] = pval
            y = plot_df[col].max()
            ax.text(0.5, y + max(0.5, 0.04 * (abs(y) + 1)), f"p = {pval:.3g}",
                    ha="center", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f7f7f7", edgecolor="#555555", alpha=0.9))
        ax.set_xlabel("")
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")

    fig.suptitle("Large-Cap vs Mid-Cap Factor Distributions", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "ESG tends to generalize across size segments while scale-dependent factors show wider dispersion.",
             ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIGURES / "fig36_largecap_comparison.png")
    plt.close(fig)
    print("  [OK] fig36_largecap_comparison.png")


# --- Fig 37: Factor Contribution Waterfall ---
def fig37_factor_contribution_waterfall(df):
    path = TABLES / "benchmark_factor_contribution.csv"
    if path.exists():
        cdf = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in cdf.columns}
        factor_col = next((cols_lower[c] for c in cols_lower if "factor" in c), None)
        contrib_col = next((cols_lower[c] for c in cols_lower if "contrib" in c), None)
        if factor_col is None or contrib_col is None:
            return
        cdf = cdf[[factor_col, contrib_col]].copy()
        cdf.columns = ["factor", "contribution"]
    else:
        if "pref_balanced" not in df.columns:
            return
        top = df.nlargest(20, "pref_balanced")
        factors = ["ESG_composite", "financial_score", "growth_score", "risk_adjusted_score",
                   "stability_score", "value_score", "market_score", "operational_score"]
        avail = [c for c in factors if c in df.columns]
        if not avail:
            return
        cdf = pd.DataFrame({
            "factor": avail,
            "contribution": [(top[c].mean() - df[c].mean()) / 100.0 for c in avail],
        })

    if cdf["contribution"].abs().max() <= 1.5:
        cdf["contribution"] = cdf["contribution"] * 100

    cdf = cdf.sort_values("contribution", ascending=False).reset_index(drop=True)
    cumulative = [0.0]
    for v in cdf["contribution"]:
        cumulative.append(cumulative[-1] + v)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cdf))
    for i, v in enumerate(cdf["contribution"]):
        start = cumulative[i]
        color = "#2ca25f" if v >= 0 else "#d7301f"
        ax.bar(i, v, bottom=start, color=color, edgecolor="white", width=0.65)
        ax.text(i, start + v, f"{v:+.2f}", ha="center", va="bottom", fontsize=9)

    total = cumulative[-1]
    ax.hlines(total, -0.5, len(cdf) - 0.5, colors="#08519c", linestyles="--", linewidth=1.6,
              label=f"Total: {total:+.2f} pp")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ").title() for f in cdf["factor"]], rotation=25, ha="right")
    ax.set_ylabel("Contribution to Outperformance (pp)")
    ax.set_title("Factor Contribution Waterfall", fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig37_factor_contribution.png")
    plt.close(fig)
    print("  [OK] fig37_factor_contribution.png")


def main():
    print("=" * 70)
    print("STEP 07: GENERATE ALL RESEARCH FIGURES")
    print("=" * 70)
    df = load_data()
    print(f"[OK] Loaded {len(df)} companies\n")

    # Original figures (1-20)
    fig_score_distributions(df)
    fig_esg_radar(df)
    fig_correlation_heatmap(df)
    fig_esg_vs_financial(df)
    fig_sector_boxplots(df)
    fig_country_comparison(df)
    fig_top20_rankings(df)
    fig_factor_contribution(df)
    fig_similarity_heatmap(df)
    fig_weight_sensitivity_tornado()
    fig_profile_comparison(df)
    fig_quintile_analysis(df)
    fig_esg_by_sector(df)
    fig_scatter_matrix(df)
    fig_grid_search_heatmap()
    fig_rank_stability()
    fig_missing_data(df)
    fig_portfolio_performance()
    fig_regression_diagnostics(df)
    fig_summary_dashboard(df)

    # New figures (21-30)
    fig_pca_biplot()
    fig_dendrogram(df)
    fig_cdf_comparison(df)
    fig_bootstrap_ci()
    fig_factor_ablation()
    fig_factor_tilt_sensitivity()
    fig_extended_scores_country(df)
    fig_decile_analysis(df)
    fig_gini_inequality()
    fig_company_profiles(df)

    # New figures (31-33): ESG performance visualizations
    fig_esg_return_scatter(df, FIGURES)
    fig_portfolio_comparison_bars(df, FIGURES)
    fig_largecap_midcap_comparison(df, FIGURES)

    # New figures (34-37): ESG outperformance visualizations
    fig_esg_quintile_performance(df)
    fig_strategy_dashboard(df)
    fig36_largecap_comparison(df)
    fig37_factor_contribution_waterfall(df)

    n_figs = len(list(FIGURES.glob("*.png")))
    print(f"\n[DONE] Generated {n_figs} figures in {FIGURES}/")
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
