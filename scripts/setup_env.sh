#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Upgrade pip ==="
python3 -m pip install --upgrade pip

echo "=== Step 2: Install core dependencies ==="
pip install transformers datasets accelerate peft trl

echo "=== Step 3: Install vLLM ==="
pip install vllm

echo "=== Step 4: Install DeepSpeed ==="
pip install deepspeed

echo "=== Step 5: Install evaluation and utility packages ==="
pip install sacrebleu jieba pyyaml tqdm pandas matplotlib seaborn

echo "=== Setup complete ==="
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "import trl; print(f'trl {trl.__version__}')"
python3 -c "import deepspeed; print(f'deepspeed {deepspeed.__version__}')"
