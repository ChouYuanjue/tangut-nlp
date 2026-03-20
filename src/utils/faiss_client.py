"""FAISS-based semantic search client for Tangut character projection."""

import json
import numpy as np
from typing import List
from collections import Counter

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS not installed. Install with: pip install faiss-gpu or faiss-cpu"
    )


class FAISSSemanticClient:
    """
    FAISS-based semantic search client for retrieving candidate Tangut characters
    based on semantic similarity of Chinese words.
    """

    def __init__(self, index_path: str, id2char_path: str):
        """
        Initialize FAISS client with pre-built index.

        Args:
            index_path: Path to FAISS index file (e.g., "data/indices/tangut_semantic_index.index")
            id2char_path: Path to ID→Tangut character mapping JSON
        """
        self.index_path = index_path
        self.id2char_path = id2char_path

        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")

        # Load ID→character mapping
        with open(id2char_path, "r", encoding="utf-8") as f:
            self.id2char = json.load(f)
        print(f"✓ Loaded {len(self.id2char)} ID→Tangut mappings")

    def search_topk(
        self, query_embedding: np.ndarray, k: int = 3
    ) -> List[str]:
        """
        Search for top-k most similar Tangut characters.

        Args:
            query_embedding: Query embedding vector [1, 1024] or [1024]
            k: Number of candidates to retrieve

        Returns:
            List of top-k Tangut character strings
        """
        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32 for FAISS
        query_embedding = query_embedding.astype(np.float32)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Convert indices to Tangut characters
        candidates = []
        for idx in indices[0]:
            idx_str = str(int(idx))
            if idx_str in self.id2char:
                candidates.append(self.id2char[idx_str])
            else:
                print(f"⚠️  Warning: ID {idx_str} not found in mapping")

        return candidates

    def vote(self, candidates: List[str]) -> str:
        """
        Select final Tangut character using majority voting (Counter).

        Args:
            candidates: List of candidate Tangut characters (typically 3)

        Returns:
            Most frequent candidate, or random if all different
        """
        if not candidates:
            raise ValueError("Empty candidate list")

        if len(candidates) == 1:
            return candidates[0]

        # Count occurrences
        counter = Counter(candidates)
        most_common = counter.most_common(1)[0][0]

        return most_common

    def search_and_vote(
        self, query_embedding: np.ndarray, k: int = 3
    ) -> str:
        """
        Convenience method: search for top-k and immediately vote.

        Args:
            query_embedding: Query embedding vector
            k: Number of candidates

        Returns:
            Single Tangut character selected by voting
        """
        candidates = self.search_topk(query_embedding, k)
        return self.vote(candidates)
