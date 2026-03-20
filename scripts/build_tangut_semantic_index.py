"""Build FAISS semantic index from Tangut dictionary."""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    import faiss
except Exception as e:
    raise ImportError(
        "FAISS import failed. This is often caused by NumPy/FAISS ABI mismatch. "
        "Please install compatible versions, e.g. numpy<2 and faiss-cpu (or faiss-gpu). "
        f"Original error: {e}"
    )

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dictionary_utils import BilingualDictionary
from src.utils.embeddings import QwenEmbeddingService


def clean_explanation(text: str) -> str:
    """Clean Chinese explanation: remove brackets, punctuation, etc."""
    import re

    # Remove content in brackets
    text = re.sub(r'[【】\(\)\[\]（）].*?[【】\(\)\[\]（）]', '', text)
    # Remove punctuation
    text = re.sub(r'[，。！？；：、]', ' ', text)
    # Multiple spaces to single
    text = ' '.join(text.split())
    return text.strip()


def build_tangut_semantic_index(
    dict_path: str,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
):
    """
    Build FAISS index from Tangut dictionary.

    Args:
        dict_path: Path to dictionary.json
        model_path: Path to Qwen model
        output_dir: Output directory for index and mappings
        device: Device for embedding computation
        batch_size: Batch size for embedding
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🧠 Building Tangut Semantic Index")
    print("=" * 60)

    # Step 1: Load dictionary
    print("\n📚 Loading dictionary...")
    dictionary = BilingualDictionary(dict_path)
    stats = dictionary.get_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Characters: {stats['characters']}")
    print(f"  Words: {stats['words']}")

    # Step 2: Collect character explanations
    print("\n📝 Extracting character explanations...")
    char_explanations = []
    id2char = {}
    id2meta = {}

    with open(dict_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    idx = 0
    for entry in entries:
        # Get character or word
        tangut_char = entry.get("character", "") or entry.get("word", "")
        if not tangut_char:
            continue

        # Only use single characters for now (can extend to words)
        if len(tangut_char) > 1:
            continue

        cn_explanation = entry.get("explanationCN", "").strip()
        if not cn_explanation:
            continue

        # Clean explanation
        cleaned = clean_explanation(cn_explanation)
        if not cleaned:
            continue

        char_explanations.append(cleaned)
        id2char[str(idx)] = tangut_char
        id2meta[str(idx)] = {"character": tangut_char, "explanation": cleaned}
        idx += 1

    print(f"  Total character explanations: {len(char_explanations)}")

    # Step 3: Compute embeddings
    print("\n🔮 Computing embeddings with Qwen2.5-0.5B...")
    embedding_service = QwenEmbeddingService(model_path, device=device)
    embeddings = embedding_service.embed(
        char_explanations, batch_size=batch_size, show_progress=True
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Step 4: Build FAISS index
    print("\n🔧 Building FAISS index...")
    # Ensure embeddings are C-contiguous and float32
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    # Create index: L2 distance
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add vectors
    index.add(embeddings)
    print(f"  Index type: IndexFlatL2")
    print(f"  Dimension: {dimension}")
    print(f"  Total vectors: {index.ntotal}")

    # Step 5: Save index and mappings
    print("\n💾 Saving artifacts...")
    index_file = output_dir / "tangut_semantic_index.index"
    id2char_file = output_dir / "tangut_id2char.json"
    id2meta_file = output_dir / "tangut_id2meta.json"

    faiss.write_index(index, str(index_file))
    print(f"  ✓ Index saved: {index_file}")

    with open(id2char_file, "w", encoding="utf-8") as f:
        json.dump(id2char, f, ensure_ascii=False, indent=2)
    print(f"  ✓ ID→Char mapping saved: {id2char_file}")

    with open(id2meta_file, "w", encoding="utf-8") as f:
        json.dump(id2meta, f, ensure_ascii=False, indent=2)
    print(f"  ✓ ID→Meta mapping saved: {id2meta_file}")

    # Save embeddings for debugging
    embeddings_file = output_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"  ✓ Embeddings saved: {embeddings_file}")

    print("\n" + "=" * 60)
    print("✨ Index building complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build Tangut semantic FAISS index")
    parser.add_argument(
        "--dictionary",
        default="data/dictionary/dictionary.json",
        help="Path to dictionary.json",
    )
    parser.add_argument(
        "--model", default="models/qwen2.5-0.5b", help="Path to Qwen model"
    )
    parser.add_argument(
        "--output", default="data/indices", help="Output directory for index"
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")

    args = parser.parse_args()

    build_tangut_semantic_index(
        dict_path=args.dictionary,
        model_path=args.model,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
