#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Upgrade pip ==="
python3 -m pip install --upgrade pip

echo "=== Step 2: Install project dependencies from requirements.txt ==="
python3 -m pip install -r requirements.txt

echo "=== Setup complete ==="
python3 -c "import transformers; print(f'transformers {transformers.__version__}')"
python3 -c "import trl; print(f'trl {trl.__version__}')"
python3 -c "import deepspeed; print(f'deepspeed {deepspeed.__version__}')"
