"""
Unified evaluation entry point.
Runs all 4 metrics (Lexical Coverage, Perplexity, chrF++, LLM Judge) on
a set of predictions and saves combined results to a JSON file.
"""

import argparse
import json
import sys

from eval.lexical_coverage import LexicalCoverageScorer
from eval.perplexity import PerplexityScorer
from eval.chrf_scorer import ChrFScorer
from eval.llm_judge import LLMJudgeScorer


def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluation metrics for Tangut-NLP translations."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help='Path to predictions JSONL (must have "input" and "prediction" fields).',
    )
    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        help='Path to test set JSONL (must have "output" field for references).',
    )
    parser.add_argument(
        "--reward-dict",
        type=str,
        required=True,
        help="Path to reward_dict.json.",
    )
    parser.add_argument(
        "--ppl-model",
        type=str,
        default="models/qwen2.5-0.5b",
        help="Path to the perplexity LM (default: models/qwen2.5-0.5b).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output metrics JSON.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading predictions from {args.predictions} ...")
    predictions = load_jsonl(args.predictions)

    print(f"Loading test set from {args.test_set} ...")
    test_set = load_jsonl(args.test_set)

    inputs = [p["input"] for p in predictions]
    candidates = [p["prediction"] for p in predictions]
    references = [t["output"] for t in test_set]

    print(f"  {len(predictions)} predictions, {len(test_set)} references loaded.")

    # ------------------------------------------------------------------
    # Metric 1: Lexical Coverage
    # ------------------------------------------------------------------
    print("\n[1/4] Computing Lexical Coverage ...")
    lex_scorer = LexicalCoverageScorer(args.reward_dict)
    lex_pairs = list(zip(inputs, candidates))
    lex_results = lex_scorer.score_batch(lex_pairs)
    print(
        f"  Lexical Coverage  mean={lex_results['mean']:.4f}  "
        f"min={lex_results['min']:.4f}  max={lex_results['max']:.4f}"
    )

    # ------------------------------------------------------------------
    # Metric 2: Perplexity
    # ------------------------------------------------------------------
    print("\n[2/4] Computing Perplexity ...")
    try:
        ppl_scorer = PerplexityScorer(model_path=args.ppl_model)
        ppl_results = ppl_scorer.score_batch(candidates)
        print(
            f"  Perplexity  mean={ppl_results['mean_ppl']:.2f}  "
            f"min={ppl_results['min_ppl']:.2f}  max={ppl_results['max_ppl']:.2f}"
        )
    except Exception as e:
        print(f"  [WARNING] Perplexity scoring failed: {e}")
        ppl_results = {
            "mean_ppl": None,
            "min_ppl": None,
            "max_ppl": None,
            "scores": [],
            "error": str(e),
        }

    # ------------------------------------------------------------------
    # Metric 3: chrF++
    # ------------------------------------------------------------------
    print("\n[3/4] Computing chrF++ ...")
    chrf_scorer = ChrFScorer()
    chrf_results = chrf_scorer.score(candidates, references)
    print(
        f"  chrF++ corpus={chrf_results['corpus_chrf']:.2f}  "
        f"mean_sentence={chrf_results['mean_sentence_chrf']:.2f}"
    )

    # ------------------------------------------------------------------
    # Metric 4: LLM Judge (mock)
    # ------------------------------------------------------------------
    print("\n[4/4] Computing LLM Judge scores (mock) ...")
    llm_scorer = LLMJudgeScorer(mock=True)
    # Build dummy glosses from the lexical scorer lookup
    glosses_list = []
    for inp in inputs:
        glosses = []
        for key, meanings in lex_scorer.lookup.items():
            if key in inp:
                glosses.extend(meanings)
        glosses_list.append(glosses)

    llm_results = llm_scorer.score_batch(inputs, candidates, glosses_list)
    print(
        f"  LLM Judge  mean_semantic={llm_results['mean_semantic_completeness']:.2f}  "
        f"mean_fluency={llm_results['mean_fluency']:.2f}"
    )

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    combined = {
        "lexical_coverage": {
            "mean": lex_results["mean"],
            "min": lex_results["min"],
            "max": lex_results["max"],
        },
        "perplexity": {
            "mean_ppl": ppl_results["mean_ppl"],
            "min_ppl": ppl_results["min_ppl"],
            "max_ppl": ppl_results["max_ppl"],
        },
        "chrf": {
            "corpus_chrf": chrf_results["corpus_chrf"],
            "mean_sentence_chrf": chrf_results["mean_sentence_chrf"],
        },
        "llm_judge": {
            "mean_semantic_completeness": llm_results["mean_semantic_completeness"],
            "mean_fluency": llm_results["mean_fluency"],
        },
        "num_examples": len(predictions),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output}")
    print("\n===== Summary =====")
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
