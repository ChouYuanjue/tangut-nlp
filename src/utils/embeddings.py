"""Qwen-based embedding service for semantic projection in data synthesis."""

import torch
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


class QwenEmbeddingService:
    """
    Embedding service using Qwen2.5-0.5B's last hidden state.
    Produces 1024-dimensional normalized embeddings.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the embedding service.

        Args:
            model_path: Path to the Qwen model (e.g., "models/qwen2.5-0.5b")
            device: Device to load model on ("cuda" or "cpu")
        """
        self.device = device
        self.model_path = model_path
        model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=model_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        self.model.eval()

        # Verify hidden size
        self.hidden_size = self.model.config.hidden_size
        print(f"✓ Qwen embedding service initialized: {self.hidden_size}-dim embeddings")

    def embed(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """
        Compute embeddings for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape [len(texts), hidden_size] with normalized embeddings
        """
        embeddings = []
        iterator = tqdm(
            range(0, len(texts), batch_size),
            desc="Computing embeddings",
            disable=not show_progress,
        )

        with torch.no_grad():
            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                for key in encoded:
                    encoded[key] = encoded[key].to(self.device)

                # Forward pass
                output = self.model(**encoded, output_hidden_states=True)
                # Get last hidden state: [batch_size, seq_len, hidden_size]
                last_hidden = output.hidden_states[-1]

                # Mean pooling over sequence dimension
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                sequence_lengths = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(-1)
                mean_pooled = sum_hidden / sequence_lengths  # [batch_size, hidden_size]

                # L2 normalize
                norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True)
                normalized = mean_pooled / (norm + 1e-8)

                # NumPy does not reliably support direct bfloat16 conversion in this env.
                embeddings.append(normalized.float().cpu().numpy().astype(np.float32))

        return np.vstack(embeddings) if embeddings else np.array([])

    def embed_single(self, text: str) -> np.ndarray:
        """
        Compute embedding for a single text.

        Args:
            text: Input text

        Returns:
            numpy array of shape [1, hidden_size] with normalized embedding
        """
        return self.embed([text], batch_size=1, show_progress=False)

    @classmethod
    def from_pretrained(
        cls, model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cuda"
    ):
        """
        Load embedding service directly from HuggingFace model name.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load on

        Returns:
            QwenEmbeddingService instance
        """
        from transformers import AutoModel

        # This would require downloading from HuggingFace
        return cls(model_name, device=device)
