"""
Metric 4: LLM-as-a-Judge (mock for now)
Uses an LLM to rate translation quality on semantic completeness and fluency.
Currently provides a mock implementation with random scores.
"""

import random
from typing import List, Optional


class LLMJudgeScorer:
    """LLM-based judge for translation quality assessment.

    Currently operates in mock mode, returning random scores.
    Designed to be extended with real API calls (e.g. to GPT-4 / Claude).
    """

    def __init__(self, api_key: Optional[str] = None, mock: bool = True):
        """Initialize the LLM judge.

        Args:
            api_key: API key for the LLM service (unused in mock mode).
            mock: If True, return random scores instead of calling an LLM.
        """
        self.api_key = api_key
        self.mock = mock

    def score(
        self,
        tangut_input: str,
        candidate: str,
        dictionary_glosses: list,
    ) -> dict:
        """Score a single translation using the LLM judge.

        Args:
            tangut_input: Source Tangut string.
            candidate: Candidate Chinese translation.
            dictionary_glosses: List of expected dictionary glosses for
                reference context.

        Returns:
            Dict with keys:
                - "semantic_completeness": int 1-5
                - "fluency": int 1-5
                - "reasoning": str explanation
        """
        if self.mock:
            return self._mock_score()

        # TODO: Implement real LLM API call
        raise NotImplementedError(
            "Real LLM judge not yet implemented. Set mock=True."
        )

    def _mock_score(self) -> dict:
        """Generate random mock scores for testing.

        Returns:
            Dict with random semantic_completeness (1-5), fluency (1-5),
            and a placeholder reasoning string.
        """
        semantic = random.randint(1, 5)
        fluency = random.randint(1, 5)
        return {
            "semantic_completeness": semantic,
            "fluency": fluency,
            "reasoning": (
                f"[MOCK] Semantic completeness: {semantic}/5, "
                f"Fluency: {fluency}/5. "
                "This is a placeholder score for testing purposes."
            ),
        }

    def score_batch(
        self,
        inputs: List[str],
        candidates: List[str],
        glosses_list: List[list],
    ) -> dict:
        """Score a batch of translations.

        Args:
            inputs: List of source Tangut strings.
            candidates: List of candidate Chinese translations.
            glosses_list: List of gloss lists (one per example).

        Returns:
            Dict with keys:
                - "mean_semantic_completeness": float
                - "mean_fluency": float
                - "scores": list of individual score dicts
        """
        scores = [
            self.score(inp, cand, glosses)
            for inp, cand, glosses in zip(inputs, candidates, glosses_list)
        ]

        if not scores:
            return {
                "mean_semantic_completeness": 0.0,
                "mean_fluency": 0.0,
                "scores": [],
            }

        mean_semantic = sum(s["semantic_completeness"] for s in scores) / len(scores)
        mean_fluency = sum(s["fluency"] for s in scores) / len(scores)

        return {
            "mean_semantic_completeness": mean_semantic,
            "mean_fluency": mean_fluency,
            "scores": scores,
        }
