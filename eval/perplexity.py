"""
Metric 2: Perplexity via a small Chinese LM (Qwen2.5-0.5B)
Measures how fluent / natural the generated Chinese text is according to a
pretrained language model.  Lower perplexity indicates more natural text.
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PerplexityScorer:
    """Computes token-level perplexity of Chinese text using a causal LM."""

    def __init__(
        self,
        model_path: str = "models/qwen2.5-0.5b",
        device: str | None = None,
        max_length: int = 512,
    ):
        """Load model and tokenizer.

        Args:
            model_path: Path (or HF hub name) for Qwen2.5-0.5B.
            device: Torch device string. Defaults to the first visible CUDA
                device, or CPU when CUDA is unavailable.
            max_length: Maximum token length for input truncation.
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def score(self, text: str) -> float:
        """Compute perplexity for a single text string.

        Args:
            text: Chinese text to evaluate.

        Returns:
            Perplexity (float).  Lower is better.
        """
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encodings.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        ppl = math.exp(loss.item())
        return ppl

    def score_batch(self, texts: list) -> dict:
        """Compute perplexity for a batch of texts.

        Args:
            texts: List of Chinese text strings.

        Returns:
            Dict with keys "mean_ppl", "min_ppl", "max_ppl", and "scores".
        """
        scores = [self.score(t) for t in texts]

        if not scores:
            return {"mean_ppl": 0.0, "min_ppl": 0.0, "max_ppl": 0.0, "scores": []}

        return {
            "mean_ppl": sum(scores) / len(scores),
            "min_ppl": min(scores),
            "max_ppl": max(scores),
            "scores": scores,
        }
