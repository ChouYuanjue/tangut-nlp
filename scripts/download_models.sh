#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Downloading Qwen2.5-7B-Instruct ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/qwen2.5-7b-instruct')
print('Qwen2.5-7B-Instruct downloaded')
"

echo "=== Downloading Qwen2.5-0.5B (for PPL evaluation) ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='models/qwen2.5-0.5b')
print('Qwen2.5-0.5B downloaded')
"

echo "=== Downloading shibing624/ancient-chinese dataset ==="
python3 -c "
from datasets import load_dataset
ds = load_dataset('shibing624/ancient-chinese', split='train')
ds.save_to_disk('data/raw/ancient_chinese_hf')
print(f'Downloaded {len(ds)} ancient Chinese pairs')
"

echo "=== All downloads complete ==="
