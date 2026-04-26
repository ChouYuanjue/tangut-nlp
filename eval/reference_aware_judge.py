"""Reference-aware LLM judge for Tangut short-text translation.

Unlike the legacy judge in ``eval/llm_judge.py``, this script gives the judge
access to the source Tangut text, the gold reference, and a candidate output.
It is intended as a reviewer-facing supplementary evaluation rather than a
training-time reward.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.azure_openai_config import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    resolve_azure_openai_config,
)


DEFAULT_API_VERSION = DEFAULT_AZURE_OPENAI_API_VERSION


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ReferenceAwareJudge:
    """Judge candidate translations against both the source and the reference."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        deployment: Optional[str] = None,
        mock: bool = False,
        timeout: int = 30,
    ) -> None:
        self.mock = mock
        self.timeout = timeout

        if self.mock:
            self.deployment = deployment or ""
            self.client = None
            return

        config = resolve_azure_openai_config(
            api_key=api_key,
            endpoint=endpoint,
            deployment=deployment,
        )
        self.deployment = config["deployment"]
        self.client = AzureOpenAI(
            azure_endpoint=config["endpoint"],
            api_key=config["api_key"],
            api_version=api_version,
            timeout=timeout,
        )

    def score(
        self,
        tangut_input: str,
        reference: str,
        candidate: str,
        max_retries: int = 2,
    ) -> dict:
        if self.mock:
            return self._mock_score()

        prompt = (
            "你是一位严格的西夏文短文本翻译评审员。任务是判断候选翻译是否既忠于原文，"
            "又与参考译文保持一致，尤其要注意是否把简短标题错误扩写成解释性句子。\n\n"
            f"【西夏文原文】\n{tangut_input}\n\n"
            f"【参考译文】\n{reference}\n\n"
            f"【候选译文】\n{candidate}\n\n"
            "请按 1-5 的整数对以下维度打分，并严格输出 JSON：\n"
            "1. reference_agreement: 与参考译文的一致程度。\n"
            "2. source_faithfulness: 对原文关键信息是否忠实。\n"
            "3. title_style_fitness: 是否保持了短标题/书名体的紧凑形式，而不是扩写成解释性句子。\n"
            "4. overall: 综合评分。\n"
            '格式示例：{"reference_agreement": 4, "source_faithfulness": 4, "title_style_fitness": 5, "overall": 4, "reasoning": "简要理由"}'
        )

        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model=self.deployment,
                    instructions=(
                        "Return strict JSON only. Score conservatively. "
                        "A candidate that paraphrases a title into a sentence should lose points."
                    ),
                    input=prompt,
                    max_output_tokens=320,
                )
                content = (response.output_text or "{}").strip()
                if content.startswith("```json"):
                    content = content.replace("```json\n", "", 1)
                elif content.startswith("```"):
                    content = content.replace("```\n", "", 1)
                if content.endswith("```"):
                    content = content.rpartition("```")[0]
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx != 0:
                    content = content[start_idx:end_idx]
                parsed = json.loads(content)
                return {
                    "reference_agreement": int(parsed.get("reference_agreement", 3)),
                    "source_faithfulness": int(parsed.get("source_faithfulness", 3)),
                    "title_style_fitness": int(parsed.get("title_style_fitness", 3)),
                    "overall": int(parsed.get("overall", 3)),
                    "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
                }
            except (TimeoutError, ConnectionError) as exc:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {
                    "reference_agreement": 3,
                    "source_faithfulness": 3,
                    "title_style_fitness": 3,
                    "overall": 3,
                    "reasoning": f"Timeout: {exc}",
                }
            except Exception as exc:
                return {
                    "reference_agreement": 3,
                    "source_faithfulness": 3,
                    "title_style_fitness": 3,
                    "overall": 3,
                    "reasoning": f"Error: {str(exc)[:80]}",
                }

    def _mock_score(self) -> dict:
        return {
            "reference_agreement": random.randint(1, 5),
            "source_faithfulness": random.randint(1, 5),
            "title_style_fitness": random.randint(1, 5),
            "overall": random.randint(1, 5),
            "reasoning": "[MOCK] Placeholder score.",
        }

    def score_batch(self, items: List[dict]) -> dict:
        scores = []
        total = len(items)
        for idx, item in enumerate(items, start=1):
            sys.stdout.write(f"\r  Processing {idx}/{total}")
            sys.stdout.flush()
            score = self.score(
                tangut_input=item["input"],
                reference=item["reference"],
                candidate=item["prediction"],
            )
            merged = dict(item)
            merged.update(score)
            scores.append(merged)
        if total:
            sys.stdout.write(f"\r  Processing {total}/{total} ✓\n")
            sys.stdout.flush()

        if not scores:
            return {
                "mean_reference_agreement": 0.0,
                "mean_source_faithfulness": 0.0,
                "mean_title_style_fitness": 0.0,
                "mean_overall": 0.0,
                "scores": [],
            }

        return {
            "mean_reference_agreement": sum(x["reference_agreement"] for x in scores) / len(scores),
            "mean_source_faithfulness": sum(x["source_faithfulness"] for x in scores) / len(scores),
            "mean_title_style_fitness": sum(x["title_style_fitness"] for x in scores) / len(scores),
            "mean_overall": sum(x["overall"] for x in scores) / len(scores),
            "scores": scores,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reference-aware LLM judge on predictions.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl.")
    parser.add_argument(
        "--test-set",
        default="data/eval/test_set.jsonl",
        help="Path to the reference test set JSONL.",
    )
    parser.add_argument("--output", required=True, help="Path to save JSON results.")
    parser.add_argument("--mock", action="store_true", help="Use random mock scores.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--api-key", default=None, help="Optional explicit Azure API key override.")
    parser.add_argument("--endpoint", default=None, help="Optional explicit Azure endpoint override.")
    parser.add_argument("--deployment", default=None, help="Optional explicit Azure deployment override.")
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION, help="Azure API version.")
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions)
    test_set = load_jsonl(args.test_set)
    if len(predictions) != len(test_set):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(predictions)} vs {len(test_set)}"
        )

    items = []
    for pred, ref in zip(predictions, test_set):
        items.append(
            {
                "input": pred["input"],
                "reference": ref["output"],
                "prediction": pred["prediction"],
                "method": pred.get("method", "unknown"),
            }
        )

    judge = ReferenceAwareJudge(
        api_key=args.api_key,
        endpoint=args.endpoint,
        api_version=args.api_version,
        deployment=args.deployment,
        mock=args.mock,
        timeout=args.timeout,
    )
    result = judge.score_batch(items)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "mean_reference_agreement": result["mean_reference_agreement"],
            "mean_source_faithfulness": result["mean_source_faithfulness"],
            "mean_title_style_fitness": result["mean_title_style_fitness"],
            "mean_overall": result["mean_overall"],
            "num_examples": len(result["scores"]),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
