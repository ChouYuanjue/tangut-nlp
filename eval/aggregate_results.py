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


def create_bar_chart(df: pd.DataFrame, results_root: str = RESULTS_ROOT):
    """Create a grouped bar chart comparing key metrics across experiments.

    Saves the chart to results/comparison.png.

    Args:
        df: Comparison DataFrame.
        results_root: Root results directory.
    """
    # Select key metrics for visualization
    metric_cols = [
        ("lex_coverage_mean", "Lex Coverage"),
        ("chrf_corpus", "chrF++ (corpus)"),
        ("llm_semantic", "LLM Semantic"),
        ("llm_fluency", "LLM Fluency"),
    ]

    # Filter to columns that exist and have data
    valid_cols = []
    for col, label in metric_cols:
        if col in df.columns and df[col].notna().any():
            valid_cols.append((col, label))

    if not valid_cols:
        print("  [WARNING] No valid metric columns found for chart.")
        return

    experiments = df["experiment"].tolist()
    n_experiments = len(experiments)
    n_metrics = len(valid_cols)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.8 / n_metrics
    x = range(n_experiments)

    for i, (col, label) in enumerate(valid_cols):
        offsets = [xi + i * bar_width for xi in x]
        values = df[col].fillna(0).tolist()
        ax.bar(offsets, values, bar_width, label=label)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("Tangut-NLP Evaluation: Metric Comparison Across Experiments")
    ax.set_xticks([xi + bar_width * (n_metrics - 1) / 2 for xi in x])
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
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
    create_bar_chart(df)
    create_radar_chart(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
