from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from regime.utils import DISPLAY_NAMES, FIGURES, MODEL_FEATURES, REGIME_COLORS, REGIME_ORDER, ensure_project_dirs


INPUT_PATH = ROOT / "data" / "processed" / "regime_inputs_simplified.csv"
LABELS_PATH = ROOT / "results" / "regime" / "simplified_regime_labels.csv"
TIMELINE_PATH = FIGURES / "simplified_regime_timeline.png"
DISTRIBUTIONS_PATH = FIGURES / "regime_feature_distributions.png"
PER_VARIABLE_DISTRIBUTION_PATHS = {
    feature: FIGURES / f"distribution_{feature}.png" for feature in MODEL_FEATURES
}
PERCENTILES_PATH = ROOT / "results" / "regime" / "regime_feature_percentiles.csv"
PERCENTILE_HEATMAP_PATH = FIGURES / "regime_feature_percentile_heatmap.png"
PROFILE_BARS_PATH = FIGURES / "regime_profile_percentile_bars.png"


def plot_timeline(panel: pd.DataFrame) -> None:
    y_map = {name: idx for idx, name in enumerate(REGIME_ORDER)}
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.step(panel["date"], panel["regime_order"], where="post", color="#d0d0d0", linewidth=1.2, zorder=1)
    for regime_name in REGIME_ORDER:
        regime_panel = panel.loc[panel["regime_name"] == regime_name]
        ax.scatter(
            regime_panel["date"],
            np.full(len(regime_panel), y_map[regime_name]),
            s=22,
            color=REGIME_COLORS[regime_name],
            label=regime_name,
            zorder=2,
        )
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(REGIME_ORDER)
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.set_title("Simplified Baseline Regime Timeline")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(TIMELINE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, feature in enumerate(MODEL_FEATURES):
        ax = axes[idx]
        feature_series = panel[feature].dropna()
        bins = np.histogram_bin_edges(feature_series, bins="fd")
        if len(bins) < 6:
            bins = np.histogram_bin_edges(feature_series, bins=12)
        grouped = [panel.loc[panel["regime_name"] == regime_name, feature].dropna().values for regime_name in REGIME_ORDER]
        ax.hist(
            grouped,
            bins=bins,
            stacked=True,
            color=[REGIME_COLORS[name] for name in REGIME_ORDER],
            label=REGIME_ORDER,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(feature_series, bins=bins, histtype="step", color="#333333", linewidth=1.1)
        ax.set_title(DISPLAY_NAMES[feature])
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Feature Distributions by Regime", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(DISTRIBUTIONS_PATH, dpi=180)
    plt.close(fig)


def plot_variable_distribution_by_regime(panel: pd.DataFrame, feature: str, output_path: Path) -> None:
    feature_series = panel[feature].dropna()
    bins = np.histogram_bin_edges(feature_series, bins="fd")
    if len(bins) < 6:
        bins = np.histogram_bin_edges(feature_series, bins=12)
    x_min, x_max = float(bins[0]), float(bins[-1])

    full_counts, _ = np.histogram(feature_series, bins=bins)
    y_max = int(full_counts.max()) if len(full_counts) else 0
    for regime_name in REGIME_ORDER:
        regime_values = panel.loc[panel["regime_name"] == regime_name, feature].dropna()
        regime_counts, _ = np.histogram(regime_values, bins=bins)
        y_max = max(y_max, int(regime_counts.max()) if len(regime_counts) else 0)
    y_max = max(y_max, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    subplot_titles = ["Full Sample"] + REGIME_ORDER

    for ax, title in zip(axes[:5], subplot_titles):
        ax.hist(
            feature_series,
            bins=bins,
            color="#bdbdbd",
            alpha=0.55,
            edgecolor="white",
            linewidth=0.6,
        )
        if title != "Full Sample":
            regime_values = panel.loc[panel["regime_name"] == title, feature].dropna()
            ax.hist(
                regime_values,
                bins=bins,
                color=REGIME_COLORS[title],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max * 1.08)
        ax.grid(axis="y", alpha=0.18)

    axes[5].axis("off")
    for idx, ax in enumerate(axes[:5]):
        if idx in (0, 3):
            ax.set_ylabel("Count")
        if idx >= 3:
            ax.set_xlabel(DISPLAY_NAMES[feature])

    fig.suptitle(f"Distribution of {DISPLAY_NAMES[feature]} by Regime", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all_variable_distributions(panel: pd.DataFrame) -> list[Path]:
    saved_paths: list[Path] = []
    for feature in MODEL_FEATURES:
        output_path = PER_VARIABLE_DISTRIBUTION_PATHS[feature]
        plot_variable_distribution_by_regime(panel, feature, output_path)
        saved_paths.append(output_path)
    return saved_paths


def plot_regime_feature_percentile_heatmap(percentiles: pd.DataFrame) -> None:
    heatmap_df = (
        percentiles.pivot(index="regime_name", columns="variable", values="full_sample_mean_percentile")
        .reindex(index=REGIME_ORDER, columns=MODEL_FEATURES)
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(heatmap_df.to_numpy(), cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(MODEL_FEATURES)))
    ax.set_xticklabels([DISPLAY_NAMES[col] for col in MODEL_FEATURES], rotation=20, ha="right")
    ax.set_yticks(range(len(REGIME_ORDER)))
    ax.set_yticklabels(REGIME_ORDER)
    for i in range(len(REGIME_ORDER)):
        for j in range(len(MODEL_FEATURES)):
            value = heatmap_df.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9, color="#111111")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Percentile Rank of Regime Mean")
    ax.set_title("Regime Feature Percentile Heatmap")
    fig.tight_layout()
    fig.savefig(PERCENTILE_HEATMAP_PATH, dpi=180)
    plt.close(fig)


def plot_regime_profile_percentile_bars(percentiles: pd.DataFrame) -> None:
    profile_df = (
        percentiles.pivot(index="regime_name", columns="variable", values="full_sample_mean_percentile")
        .reindex(index=REGIME_ORDER, columns=MODEL_FEATURES)
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(MODEL_FEATURES))
    labels = [DISPLAY_NAMES[col] for col in MODEL_FEATURES]
    for ax, regime_name in zip(axes, REGIME_ORDER):
        values = profile_df.loc[regime_name].to_numpy(dtype=float)
        ax.bar(x, values, color=REGIME_COLORS[regime_name], alpha=0.9, width=0.72)
        ax.axhline(50, color="#666666", linewidth=0.9, linestyle="--")
        ax.set_title(regime_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentile")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Regime Profile Percentile Bars", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PROFILE_BARS_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_project_dirs()
    inputs = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    labels = pd.read_csv(LABELS_PATH, parse_dates=["date"])
    percentiles = pd.read_csv(PERCENTILES_PATH)
    panel = inputs.merge(labels[["date", "state_raw", "regime_name", "regime_order"]], on="date", how="inner")
    plot_timeline(panel)
    saved_distribution_paths = plot_all_variable_distributions(panel)
    plot_regime_feature_percentile_heatmap(percentiles)
    plot_regime_profile_percentile_bars(percentiles)
    print(f"Saved timeline figure to {TIMELINE_PATH}")
    for path in saved_distribution_paths:
        print(f"Saved distribution figure to {path}")
    print(f"Saved percentile heatmap to {PERCENTILE_HEATMAP_PATH}")
    print(f"Saved regime profile bars to {PROFILE_BARS_PATH}")


if __name__ == "__main__":
    main()
