"""
Metric 1: Lexical Coverage (Dictionary Yield)
For each Tangut character/phrase in source, check if any dictionary meanings appear in output.
Uses reward_dict.json (keyed by phrase length) for lookup.
"""

import json
import jieba


class LexicalCoverageScorer:
    """Scores how well a Chinese translation covers the expected dictionary
    meanings for each Tangut character/phrase in the source input."""

    def __init__(self, reward_dict_path: str):
        """Load reward_dict.json and flatten to tangut_key -> set(cn_tokens).

        Args:
            reward_dict_path: Path to reward_dict.json, which is keyed by
                phrase length (e.g. "2", "3") with Tangut phrases mapping
                to lists of Chinese translations.
        """
        with open(reward_dict_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Flatten: {tangut_phrase: set(chinese_meanings)}
        self.lookup = {}
        for _length, entries in raw.items():
            for tangut_key, cn_list in entries.items():
                if tangut_key not in self.lookup:
                    self.lookup[tangut_key] = set()
                self.lookup[tangut_key].update(cn_list)

    def score(self, tangut_input: str, chinese_output: str) -> float:
        """Compute lexical coverage score for a single example.

        Uses jieba to segment the Chinese output, then performs maximum forward
        matching (up to 5 chars) over the Tangut input to find dictionary
        entries and checks whether any expected meanings appear in the output.

        Args:
            tangut_input: Source Tangut string.
            chinese_output: Candidate Chinese translation.

        Returns:
            Float in [0, 1] representing the fraction of matched Tangut spans
            whose expected Chinese meanings appear in the output.
        """
        # Build output token set from jieba segmentation + individual chars
        output_tokens = set(jieba.lcut(chinese_output))
        for ch in chinese_output:
            if ch.strip():
                output_tokens.add(ch)

        # Maximum forward matching over tangut_input
        matched = 0
        total = 0
        i = 0
        max_len = 5

        while i < len(tangut_input):
            best_phrase = None
            best_meanings = None

            # Try longest match first
            for length in range(min(max_len, len(tangut_input) - i), 0, -1):
                phrase = tangut_input[i : i + length]
                if phrase in self.lookup:
                    best_phrase = phrase
                    best_meanings = self.lookup[phrase]
                    break

            if best_phrase is not None:
                total += 1
                # Check if any expected meaning appears in output tokens
                if best_meanings & output_tokens:
                    matched += 1
                i += len(best_phrase)
            else:
                # Single character not in dictionary -- skip
                i += 1

        if total == 0:
            return 0.0

        return matched / total

    def score_batch(self, pairs: list) -> dict:
        """Score a batch of (tangut_input, chinese_output) pairs.

        Args:
            pairs: List of (tangut_input, chinese_output) tuples.

        Returns:
            Dict with keys "mean", "min", "max", and "scores".
        """
        scores = [self.score(tangut, chinese) for tangut, chinese in pairs]

        if not scores:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "scores": []}

        return {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "scores": scores,
        }
