#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Upgrade pip ==="
python3 -m pip install --upgrade pip setuptools wheel

echo "=== Step 2: Install base scientific stack (pin NumPy<2 for FAISS ABI) ==="
python3 -m pip install "numpy<2" pandas pyyaml tqdm matplotlib seaborn sacrebleu jieba beautifulsoup4

echo "=== Step 3: Install HF/Training stack ==="
python3 -m pip install transformers datasets accelerate peft trl

echo "=== Step 4: Install DeepSpeed ==="
python3 -m pip install deepspeed

echo "=== Step 5: Install vLLM (resolver-optimized) ==="
# vLLM dependency graph is large; use legacy resolver to avoid long backtracking.
python3 -m pip install --use-deprecated=legacy-resolver "vllm>=0.6.0"

echo "=== Step 6: Sync remaining requirements (non-fatal if already satisfied) ==="
python3 -m pip install --use-deprecated=legacy-resolver -r requirements.txt || true

echo "=== Setup complete ==="
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "import trl; print(f'trl {trl.__version__}')"
python3 -c "import deepspeed; print(f'deepspeed {deepspeed.__version__}')"
python3 -c "import vllm; print(f'vllm {vllm.__version__}')"
