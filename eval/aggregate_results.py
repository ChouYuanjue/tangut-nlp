"""
Aggregate evaluation results across experiment baselines.
Auto-discovers results/*/metrics.json and produces comparison tables/charts.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
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

    print("\nDone.")


if __name__ == "__main__":
    main()
