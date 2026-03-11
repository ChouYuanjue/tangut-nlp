"""
Metric 3: chrF++ via sacrebleu
Character n-gram F-score with word n-grams (chrF++).
Particularly suited for character-rich languages and Chinese output.
"""

from typing import List

import sacrebleu


class ChrFScorer:
    """Computes chrF++ scores using sacrebleu."""

    def score(
        self, hypotheses: List[str], references: List[str]
    ) -> dict:
        """Compute corpus-level and sentence-level chrF++ scores.

        Args:
            hypotheses: List of candidate translation strings.
            references: List of reference translation strings (same length).

        Returns:
            Dict with keys:
                - "corpus_chrf": corpus-level chrF++ score (float)
                - "per_sentence": list of per-sentence chrF++ scores
                - "mean_sentence_chrf": mean of per-sentence scores (float)
        """
        # Corpus-level chrF++
        corpus_result = sacrebleu.corpus_chrf(
            hypotheses,
            [references],
            char_order=6,
            word_order=2,
            beta=2,
        )
        corpus_chrf = corpus_result.score

        # Per-sentence chrF++
        per_sentence = []
        for hyp, ref in zip(hypotheses, references):
            sent_result = sacrebleu.corpus_chrf(
                [hyp],
                [[ref]],
                char_order=6,
                word_order=2,
                beta=2,
            )
            per_sentence.append(sent_result.score)

        mean_sentence_chrf = (
            sum(per_sentence) / len(per_sentence) if per_sentence else 0.0
        )

        return {
            "corpus_chrf": corpus_chrf,
            "per_sentence": per_sentence,
            "mean_sentence_chrf": mean_sentence_chrf,
        }
