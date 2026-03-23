"""
Aggregate evaluation results across experiment baselines.
Auto-discovers results/*/metrics.json and produces comparison tables/charts.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = "results"
HUMAN_REF_EXPERIMENT = "human_reference"


def discover_experiments(results_root: str = RESULTS_ROOT) -> list:
    """Discover experiment directories that contain metrics.json."""
    if not os.path.isdir(results_root):
        return []

    experiments = []
    for name in sorted(os.listdir(results_root)):
        path = os.path.join(results_root, name)
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.isdir(path) and os.path.isfile(metrics_path):
            experiments.append(name)
    return experiments


def load_metrics(results_root: str = RESULTS_ROOT) -> dict:
    """Load metrics.json from each experiment directory.

    Args:
        results_root: Root directory containing experiment subdirectories.

    Returns:
        Dict mapping experiment name -> metrics dict.
    """
    all_metrics = {}
    for exp in discover_experiments(results_root):
        metrics_path = os.path.join(results_root, exp, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                all_metrics[exp] = json.load(f)
    return all_metrics


def build_comparison_dataframe(all_metrics: dict) -> pd.DataFrame:
    """Flatten metrics into a comparison DataFrame.

    Args:
        all_metrics: Dict mapping experiment name -> metrics dict.

    Returns:
        DataFrame with one row per experiment and columns for each metric.
    """
    rows = []
    for exp_name, metrics in all_metrics.items():
        row = {"experiment": exp_name}

        # Lexical coverage
        lex = metrics.get("lexical_coverage", {})
        row["lex_coverage_mean"] = lex.get("mean")
        row["lex_coverage_min"] = lex.get("min")
        row["lex_coverage_max"] = lex.get("max")

        # Perplexity
        ppl = metrics.get("perplexity", {})
        row["ppl_mean"] = ppl.get("mean_ppl")
        row["ppl_min"] = ppl.get("min_ppl")
        row["ppl_max"] = ppl.get("max_ppl")

        # chrF++
        chrf = metrics.get("chrf", {})
        row["chrf_corpus"] = chrf.get("corpus_chrf")
        row["chrf_mean_sentence"] = chrf.get("mean_sentence_chrf")

        # LLM Judge
        llm = metrics.get("llm_judge", {})
        row["llm_semantic"] = llm.get("mean_semantic_completeness")
        row["llm_fluency"] = llm.get("mean_fluency")

        row["num_examples"] = metrics.get("num_examples")

        rows.append(row)

    return pd.DataFrame(rows)


def save_comparison(df: pd.DataFrame, results_root: str = RESULTS_ROOT):
    """Save comparison DataFrame to JSON and CSV.

    Args:
        df: Comparison DataFrame.
        results_root: Root results directory.
    """
    json_path = os.path.join(results_root, "comparison.json")
    csv_path = os.path.join(results_root, "comparison.csv")

    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    df.to_csv(csv_path, index=False)

    print(f"  Saved {json_path}")
    print(f"  Saved {csv_path}")


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return np.nan
    if denominator == 0:
        return np.nan
    return numerator / denominator


def build_dual_anchor_dataframe(
    df: pd.DataFrame,
    human_ref_experiment: str = HUMAN_REF_EXPERIMENT,
) -> pd.DataFrame:
    """Build dual-anchor + relative-normalized comparison table.

    Anchor A: model-vs-reference absolute metrics (existing metrics columns).
    Anchor B: human_reference-vs-reference baseline metrics.

    Adds normalized columns relative to human_reference:
      - Higher-is-better metrics: model / human_reference
      - Lower-is-better PPL: human_reference / model
    """
    dual = df.copy()
    dual["anchor_primary"] = "model_vs_reference"
    dual["anchor_secondary"] = "human_reference_vs_reference"

    human_row = dual.loc[dual["experiment"] == human_ref_experiment]
    if human_row.empty:
        print(
            f"  [WARNING] {human_ref_experiment} not found; "
            "skip dual-anchor relative columns."
        )
        return dual

    human = human_row.iloc[0]
    human_lex = human.get("lex_coverage_mean")
    human_ppl = human.get("ppl_mean")
    human_chrf = human.get("chrf_corpus")
    human_sem = human.get("llm_semantic")
    human_flu = human.get("llm_fluency")

    # Anchor B values explicitly attached for side-by-side reporting.
    dual["anchor_human_lex_coverage_mean"] = human_lex
    dual["anchor_human_ppl_mean"] = human_ppl
    dual["anchor_human_chrf_corpus"] = human_chrf
    dual["anchor_human_llm_semantic"] = human_sem
    dual["anchor_human_llm_fluency"] = human_flu

    # Relative normalization (higher means closer/better than human baseline).
    dual["relative_lex"] = dual["lex_coverage_mean"].apply(
        lambda x: _safe_ratio(x, human_lex)
    )
    dual["relative_chrf"] = dual["chrf_corpus"].apply(
        lambda x: _safe_ratio(x, human_chrf)
    )
    dual["relative_llm_semantic"] = dual["llm_semantic"].apply(
        lambda x: _safe_ratio(x, human_sem)
    )
    dual["relative_llm_fluency"] = dual["llm_fluency"].apply(
        lambda x: _safe_ratio(x, human_flu)
    )
    dual["relative_ppl"] = dual["ppl_mean"].apply(
        lambda x: _safe_ratio(human_ppl, x)
    )

    return dual


def save_dual_anchor_comparison(df: pd.DataFrame, results_root: str = RESULTS_ROOT):
    """Save dual-anchor + relative-normalized comparison table."""
    dual_df = build_dual_anchor_dataframe(df)

    json_path = os.path.join(results_root, "comparison_dual_anchor.json")
    csv_path = os.path.join(results_root, "comparison_dual_anchor.csv")

    dual_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    dual_df.to_csv(csv_path, index=False)

    print(f"  Saved {json_path}")
    print(f"  Saved {csv_path}")


def create_bar_chart(df: pd.DataFrame, results_root: str = RESULTS_ROOT):
    """Create a grouped bar chart with unified normalization across metrics.

    Saves the chart to results/comparison.png.

    Args:
        df: Comparison DataFrame.
        results_root: Root results directory.
    """
    required_cols = [
        "lex_coverage_mean",
        "ppl_mean",
        "chrf_corpus",
        "llm_semantic",
        "llm_fluency",
    ]
    missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
    if missing:
        print(f"  [WARNING] Missing/empty columns for normalized chart: {missing}")
        return

    def _clip_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
        return series.astype(float).clip(lower=lo, upper=hi)

    # Unified [0,1] normalization so no single metric dominates chart scale.
    normalized = pd.DataFrame(
        {
            "Lex": _clip_series(df["lex_coverage_mean"], 0.0, 1.0),
            "PPLQ": (
                6.0 - np.log10(_clip_series(df["ppl_mean"], 1.0, 1_000_000.0))
            )
            / 6.0,
            "chrF": _clip_series(df["chrf_corpus"], 0.0, 100.0) / 100.0,
            "Sem": (_clip_series(df["llm_semantic"], 1.0, 5.0) - 1.0) / 4.0,
            "Flu": (_clip_series(df["llm_fluency"], 1.0, 5.0) - 1.0) / 4.0,
        }
    ).fillna(0.0)

    experiments = df["experiment"].astype(str).tolist()
    # Wrap long experiment names to avoid overlap in bottom text.
    wrapped_labels = [
        exp.replace("_", "_\n") if len(exp) > 10 else exp for exp in experiments
    ]

    n_experiments = len(experiments)
    metric_labels = list(normalized.columns)
    n_metrics = len(metric_labels)

    fig, ax = plt.subplots(figsize=(15.2, 7.8))

    # Wider gaps between experiment groups reduce cross-group visual interference
    # when some bars are near zero.
    group_gap = 1.34
    group_span = 0.68
    bar_width = group_span / n_metrics
    x = np.arange(n_experiments) * group_gap
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for i, label in enumerate(metric_labels):
        offsets = x + (i - (n_metrics - 1) / 2.0) * bar_width
        values = normalized[label].tolist()
        ax.bar(offsets, values, bar_width, label=label, color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Normalized Score (0-1, higher is better)")
    ax.set_title("Tangut-NLP Evaluation: Unified Normalized Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02)
    if n_experiments > 1:
        split_points = (x[:-1] + x[1:]) / 2.0
        for s in split_points:
            ax.axvline(s, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)

    ax.set_xlim(x[0] - group_gap * 0.6, x[-1] + group_gap * 0.6)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax.grid(axis="y", alpha=0.28)

    # Extra bottom margin to keep long labels readable.
    plt.subplots_adjust(bottom=0.24, top=0.83)
    chart_path = os.path.join(results_root, "comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"  Saved {chart_path}")


def create_radar_chart(df: pd.DataFrame, results_root: str = RESULTS_ROOT):
    """Create redesigned radar charts with fixed, interpretable scaling.

    Saves:
        - results/comparison_radar_new.png (all experiments overlay)
        - results/comparison_radar_each.png (one panel per experiment)

    Metric scaling (all mapped to [0, 1], higher is better):
        - Lexical Coverage: direct, clipped to [0, 1]
        - PPL Quality: log-scale inverse from PPL in [1, 1e6]
        - chrF++: clipped to [0, 30], then /30
        - Semantic: from judge score [1, 5] to [0, 1]
        - Fluency: from judge score [1, 5] to [0, 1]
    """

    required_cols = [
        "lex_coverage_mean",
        "ppl_mean",
        "chrf_corpus",
        "llm_semantic",
        "llm_fluency",
    ]
    for col in required_cols:
        if col not in df.columns or df[col].isna().all():
            print(f"  [WARNING] Missing/empty column for radar chart: {col}")
            return

    def _clip_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
        return series.astype(float).clip(lower=lo, upper=hi)

    normalized = pd.DataFrame(
        {
            "Lexical Coverage": _clip_series(df["lex_coverage_mean"], 0.0, 1.0),
            "PPL Quality": (
                6.0 - np.log10(_clip_series(df["ppl_mean"], 1.0, 1_000_000.0))
            )
            / 6.0,
            "chrF++": _clip_series(df["chrf_corpus"], 0.0, 30.0) / 30.0,
            "Semantic": (_clip_series(df["llm_semantic"], 1.0, 5.0) - 1.0) / 4.0,
            "Fluency": (_clip_series(df["llm_fluency"], 1.0, 5.0) - 1.0) / 4.0,
        }
    ).fillna(0.0)

    axis_labels = [
        "Lexical\n(0-1)",
        "PPL Quality\n(log 1..1e6)",
        "chrF++\n(0-30)",
        "Semantic\n(1-5)",
        "Fluency\n(1-5)",
    ]

    labels = list(normalized.columns)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Global overlay chart.
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    colors = plt.cm.get_cmap("tab10", len(normalized))

    for idx, row in normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        exp_name = df.loc[idx, "experiment"]
        color = colors(idx % 10)
        ax.plot(angles, values, linewidth=1.8, marker="o", markersize=3, label=exp_name, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title("Tangut-NLP Evaluation: Radar Comparison (Rescaled)", pad=28)
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=8)

    plt.tight_layout()
    radar_path = os.path.join(results_root, "comparison_radar.png")
    plt.savefig(radar_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"  Saved {radar_path}")

    # Per-experiment panels to ensure every baseline is clearly visible.
    exp_count = len(normalized)
    n_cols = 3
    n_rows = int(np.ceil(exp_count / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.8 * n_rows),
        subplot_kw={"polar": True},
    )
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(normalized.iterrows()):
        ax_i = axes[i]
        values = row.tolist()
        values += values[:1]
        exp_name = df.loc[row.name, "experiment"]

        color = colors(i % 10)
        ax_i.plot(angles, values, linewidth=2.0, marker="o", markersize=3, color=color)
        ax_i.fill(angles, values, alpha=0.12, color=color)
        ax_i.set_xticks(angles[:-1])
        ax_i.set_xticklabels(axis_labels, fontsize=8)
        ax_i.set_ylim(0, 1)
        ax_i.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_i.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax_i.set_title(exp_name, fontsize=11, pad=14)
        ax_i.grid(alpha=0.35)

        raw_text = (
            f"lex={df.loc[row.name, 'lex_coverage_mean']:.3f}\n"
            f"ppl={df.loc[row.name, 'ppl_mean']:.1f}\n"
            f"chrf={df.loc[row.name, 'chrf_corpus']:.2f}\n"
            f"sem={df.loc[row.name, 'llm_semantic']:.2f}\n"
            f"flu={df.loc[row.name, 'llm_fluency']:.2f}"
        )
        ax_i.text(
            np.deg2rad(6),
            1.14,
            raw_text,
            fontsize=7,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2.5},
        )

    for j in range(exp_count, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Tangut-NLP Evaluation: Radar by Experiment (Rescaled)", y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    radar_each_path = os.path.join(results_root, "comparison_radar_each.png")
    plt.savefig(radar_each_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"  Saved {radar_each_path}")


def create_relative_radar_chart(df: pd.DataFrame, results_root: str = RESULTS_ROOT):
    """Create radar chart using metrics relative to human_reference baseline.

    Relative columns come from build_dual_anchor_dataframe:
      - relative_lex
      - relative_chrf
      - relative_llm_semantic
      - relative_llm_fluency
      - relative_ppl

    1.0 indicates parity with human_reference on that metric.
    """
    dual_df = build_dual_anchor_dataframe(df)

    required_cols = [
        "relative_lex",
        "relative_chrf",
        "relative_llm_semantic",
        "relative_llm_fluency",
        "relative_ppl",
    ]
    for col in required_cols:
        if col not in dual_df.columns or dual_df[col].isna().all():
            print(f"  [WARNING] Missing/empty column for relative radar chart: {col}")
            return

    plot_df = dual_df.copy()

    # General ratio clip for non-PPL relative metrics.
    ratio_clip_max = 2.0
    for col in [
        "relative_lex",
        "relative_chrf",
        "relative_llm_semantic",
        "relative_llm_fluency",
    ]:
        plot_df[col] = (
            plot_df[col]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=ratio_clip_max)
        )

    # For PPL, use adaptive log scaling around parity=1 to avoid collapse.
    # Step 1: signed log ratio d = log10(relative_ppl), where d=0 means parity.
    # Step 2: normalize by p90(|d|) so most points are spread in [0, 2].
    # plot value = 1 + clip(d / scale, -1, 1)
    ppl_ratio = (
        plot_df["relative_ppl"]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=1e-6, upper=1e6)
    )
    ppl_delta = np.log10(ppl_ratio)
    ppl_scale = float(np.nanpercentile(np.abs(ppl_delta), 90))
    if not np.isfinite(ppl_scale) or ppl_scale < 1e-6:
        ppl_scale = 1.0
    plot_df["relative_ppl_plot"] = 1.0 + np.clip(ppl_delta / ppl_scale, -1.0, 1.0)

    metric_labels = [
        "Rel Lex",
        "Rel chrF++",
        "Rel Semantic",
        "Rel Fluency",
        "Rel PPL\n(adaptive log, centered@1)",
    ]

    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    colors = plt.cm.get_cmap("tab10", len(plot_df))

    for idx, row in plot_df.iterrows():
        values = [
            row["relative_lex"],
            row["relative_chrf"],
            row["relative_llm_semantic"],
            row["relative_llm_fluency"],
            row["relative_ppl_plot"],
        ]
        values += values[:1]
        exp_name = plot_df.loc[idx, "experiment"]
        color = colors(idx % 10)
        ax.plot(angles, values, linewidth=1.8, marker="o", markersize=3, label=exp_name, color=color)

    # Human baseline parity ring.
    ax.plot(angles, [1.0] * (n + 1), linestyle="--", linewidth=1.2, color="gray", alpha=0.7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5", "1.0", "1.5", "2.0"])
    ax.set_title("Tangut-NLP Evaluation: Relative Radar (Human Reference = 1.0)", pad=28)
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=8)

    plt.tight_layout()
    path = os.path.join(results_root, "comparison_radar_relative.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"  Saved {path}")


def main():
    print("Aggregating evaluation results ...")
    print()

    all_metrics = load_metrics()
    if not all_metrics:
        print("No metrics found. Run eval/run_all_metrics.py first.")
        return

    print(f"Found results for: {list(all_metrics.keys())}")
    print()

    df = build_comparison_dataframe(all_metrics)

    print("Comparison table:")
    print(df.to_string(index=False))
    print()

    save_comparison(df)
    save_dual_anchor_comparison(df)
    create_bar_chart(df)
    create_radar_chart(df)
    create_relative_radar_chart(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
