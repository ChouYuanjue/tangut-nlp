"""Generate synthetic Tangut-Chinese parallel corpus from ancient Chinese texts.

Supports three synthesis modes:
- mixed: 65% Tangut + 35% Chinese (original Baseline 3)
- unk: 100% pure with [UNK] placeholders (Baseline 3.1)
- semantic: 100% pure with semantic projection (Baseline 3.3)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Allow `python src/data_synthesis.py` to resolve `from src...` imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _save_checkpoint(path: Path, processed: int, written: int, skipped: int) -> None:
    state = {
        "processed": processed,
        "written": written,
        "skipped": skipped,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_checkpoint(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cn_to_tangut_map(dictionary_path):
    """Build Chinese character → Tangut character(s) mapping from dictionary."""
    with open(dictionary_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    cn_to_tangut = {}
    for entry in entries:
        tangut_char = entry.get("character", "")
        if len(tangut_char) != 1:
            continue
        cn_explanation = entry.get("explanationCN", "").strip()
        if not cn_explanation:
            continue
        for c in cn_explanation:
            if "\u4e00" <= c <= "\u9fff":
                if c not in cn_to_tangut:
                    cn_to_tangut[c] = []
                if tangut_char not in cn_to_tangut[c]:
                    cn_to_tangut[c].append(tangut_char)
    return cn_to_tangut


def synthesize_mixed_pair(ancient_text, modern_text, cn_to_tangut, replacement_ratio):
    """Original Baseline 3: Mixed 65% Tangut + 35% Chinese."""
    chars = list(ancient_text)
    replaceable_indices = [i for i, c in enumerate(chars) if c in cn_to_tangut]
    if len(replaceable_indices) < 2:
        return None
    num_to_replace = max(1, int(len(replaceable_indices) * replacement_ratio))
    indices_to_replace = random.sample(
        replaceable_indices, min(num_to_replace, len(replaceable_indices))
    )
    for idx in indices_to_replace:
        chars[idx] = random.choice(cn_to_tangut[chars[idx]])
    return "".join(chars), modern_text


def synthesize_pure_tangut_unk(ancient_text, modern_text, cn_to_tangut):
    """Baseline 3.1: 100% pure Tangut with [UNK] placeholders.

    - Dictionary hit → replace with Tangut
    - Dictionary miss (Chinese char) → replace with [UNK]
    - Non-Chinese → keep as is
    """
    chars = list(ancient_text)
    replaceable_count = 0

    for i, c in enumerate(chars):
        if c in cn_to_tangut:
            chars[i] = random.choice(cn_to_tangut[c])
            replaceable_count += 1
        elif "\u4e00" <= c <= "\u9fff":  # Chinese character range
            chars[i] = "[UNK]"
            replaceable_count += 1

    # Only return if we actually replaced something
    if replaceable_count < 1:
        return None
    return "".join(chars), modern_text


def synthesize_semantic_projection(
    ancient_text,
    modern_text,
    cn_to_tangut,
    embedding_service,
    faiss_client,
    embedding_cache: Optional[dict] = None,
):
    """Baseline 3.3: Vector-space semantic projection.

    - Dictionary hit → replace with Tangut
    - Dictionary miss → compute embedding + FAISS top-3 → vote → replace
    - Failure fallback → [UNK]
    """
    if embedding_cache is None:
        embedding_cache = {}

    chars = list(ancient_text)
    replaceable_count = 0

    for i, c in enumerate(chars):
        if c in cn_to_tangut:
            # Dictionary hit
            chars[i] = random.choice(cn_to_tangut[c])
            replaceable_count += 1
        elif "\u4e00" <= c <= "\u9fff":  # Chinese character
            try:
                # Cache to avoid recomputing the same character
                if c not in embedding_cache:
                    emb = embedding_service.embed_single(c)
                    embedding_cache[c] = emb[0]  # Shape: [1024]

                embedding = embedding_cache[c]

                # FAISS search + vote
                candidates = faiss_client.search_topk(embedding, k=3)
                if candidates:
                    chosen = faiss_client.vote(candidates)
                    chars[i] = chosen
                    replaceable_count += 1
                else:
                    # No candidates → fallback to [UNK]
                    chars[i] = "[UNK]"
                    replaceable_count += 1
            except Exception as e:
                # Error → fallback to [UNK]
                print(f"⚠️  Warning: Embedding failed for '{c}': {e}, falling back to [UNK]")
                chars[i] = "[UNK]"
                replaceable_count += 1

    # Only return if we actually replaced something
    if replaceable_count < 1:
        return None
    return "".join(chars), modern_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Tangut-Chinese SFT data (3 modes)"
    )
    parser.add_argument("--dictionary-path", default="data/dictionary/dictionary.json")
    parser.add_argument("--ancient-chinese-path", default="data/raw/ancient_chinese_hf")
    parser.add_argument("--output", default="data/sft/synthetic_sft.jsonl")
    parser.add_argument(
        "--mode",
        choices=["mixed", "unk", "semantic"],
        default="mixed",
        help="Synthesis mode: mixed (original), unk (100%% UNK), semantic (vector projection)",
    )
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--replacement-min", type=float, default=0.3)
    parser.add_argument("--replacement-max", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)

    # Semantic projection specific args
    parser.add_argument(
        "--embedding-model-path",
        default="models/qwen2.5-0.5b",
        help="Path to embedding model (for semantic mode)",
    )
    parser.add_argument(
        "--faiss-index-path",
        default="data/indices/tangut_semantic_index.index",
        help="Path to FAISS index (for semantic mode)",
    )
    parser.add_argument(
        "--faiss-mapping-path",
        default="data/indices/tangut_id2char.json",
        help="Path to FAISS ID→char mapping (for semantic mode)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for semantic mode (cuda or cpu)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help="Save checkpoint every N processed rows",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    cn_to_tangut = build_cn_to_tangut_map(args.dictionary_path)
    print(f"✓ CN→Tangut map: {len(cn_to_tangut)} Chinese chars covered")
    print(f"✓ Synthesis mode: {args.mode.upper()}")

    from datasets import load_from_disk

    ds = load_from_disk(args.ancient_chinese_path)

    # Initialize semantic projection services if needed
    embedding_service = None
    faiss_client = None
    embedding_cache = {}

    if args.mode == "semantic":
        print("🔮 Loading semantic projection services...")
        from src.utils.embeddings import QwenEmbeddingService
        from src.utils.faiss_client import FAISSSemanticClient

        embedding_service = QwenEmbeddingService(args.embedding_model_path, device=args.device)
        faiss_client = FAISSSemanticClient(
            args.faiss_index_path, args.faiss_mapping_path
        )
        print("✓ Semantic services loaded")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(f"{args.output}.ckpt.json")

    processed = 0
    written = 0
    skipped = 0
    start_processed = 0

    if args.resume:
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None and output_path.exists():
            processed = int(ckpt.get("processed", 0))
            written = int(ckpt.get("written", 0))
            skipped = int(ckpt.get("skipped", 0))
            start_processed = processed
            print(
                f"↻ Resume enabled: processed={processed}, written={written}, skipped={skipped}"
            )

    write_mode = "a" if args.resume and output_path.exists() else "w"
    out_f = open(output_path, write_mode, encoding="utf-8")

    print(f"\n🔄 Processing {args.max_samples} samples...")
    for item in tqdm(ds, desc="Synthesizing"):
        if processed < start_processed:
            processed += 1
            continue

        if written >= args.max_samples:
            break

        ancient = (
            item.get("classical", "")
            or item.get("ancient", "")
            or item.get("source", "")
        )
        modern = item.get("modern", "") or item.get("target", "")
        if not ancient or not modern:
            processed += 1
            skipped += 1
            continue
        if len(ancient) < args.min_length or len(ancient) > args.max_length:
            processed += 1
            skipped += 1
            continue

        # Select synthesis function based on mode
        if args.mode == "mixed":
            ratio = random.uniform(args.replacement_min, args.replacement_max)
            result = synthesize_mixed_pair(ancient, modern, cn_to_tangut, ratio)
            metadata = {"synthetic": True, "mode": "mixed", "replacement_ratio": round(ratio, 3)}
        elif args.mode == "unk":
            result = synthesize_pure_tangut_unk(ancient, modern, cn_to_tangut)
            metadata = {"synthetic": True, "mode": "unk"}
        elif args.mode == "semantic":
            result = synthesize_semantic_projection(
                ancient, modern, cn_to_tangut, embedding_service, faiss_client, embedding_cache
            )
            metadata = {"synthetic": True, "mode": "semantic"}
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        if result is None:
            processed += 1
            skipped += 1
            continue

        input_text, target = result
        sample = {
            "instruction": "请将以下西夏文翻译为现代中文：",
            "input": input_text,
            "output": target,
            "metadata": metadata,
        }
        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        written += 1
        processed += 1

        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            out_f.flush()
            _save_checkpoint(ckpt_path, processed, written, skipped)

    out_f.flush()
    out_f.close()
    _save_checkpoint(ckpt_path, processed, written, skipped)

    print(f"\n✅ Generated {written} synthetic SFT samples")
    print(f"⊘  Skipped {skipped} invalid samples")
    print(f"📁 Output: {args.output}")
    print(f"🧷 Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
