#!/bin/bash
# =============================================================================
# run.sh — Tangut-NLP 一键端到端流水线（支持断点续跑）
#
# 两张 A100 资源调度策略：
#   - vLLM 推理/候选生成: tensor_parallel_size=2（两卡并行）
#   - SFT/DPO 训练: accelerate + DeepSpeed ZeRO-2（两卡数据并行）
#   - PPL 打分（0.5B 小模型）: 单卡 cuda:0，在候选生成后 del llm / 清显存后运行
#   - 每阶段结束后打印 nvidia-smi 确认显存已释放
#
# 断点续跑机制：
#   每个阶段成功结束时在 state/ 目录下写入 <阶段名>.done 文件。
#   重新运行脚本时，已完成的阶段会被跳过。
#   SFT/DPO 训练阶段额外检测 checkpoint 目录，自动追加 --resume 参数。
#
# 用法：
#   chmod +x run.sh
#   ./run.sh            # 首次运行或从中断点继续
#   ./run.sh --reset    # 清除所有状态，强制从头重跑（慎用）
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
#  配置区 —— 如需修改模型路径、超参等，只改这里                                    #
# --------------------------------------------------------------------------- #
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型路径
MAIN_MODEL_PATH="models/qwen2.5-7b-instruct"
PPL_MODEL_PATH="models/qwen2.5-0.5b"

# SFT 训练超参
SFT_EPOCHS=3
SFT_BATCH=4
SFT_GRAD_ACCUM=4
SFT_LR="2e-4"
SFT_LORA_RANK=64

# DPO 训练超参
DPO_EPOCHS=2
DPO_BATCH=1
DPO_GRAD_ACCUM=8
DPO_LR="5e-5"
DPO_BETA=0.1
DPO_MAX_INPUTS=5000   # 用于生成候选的训练样本上限

# 合成数据量
SYNTHETIC_MAX=50000
UPSAMPLE_REAL=10

# --------------------------------------------------------------------------- #
#  内部变量                                                                      #
# --------------------------------------------------------------------------- #
STATE_DIR="$PROJECT_DIR/state"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

# 所有输出同时写日志
exec > >(tee -a "$LOG_FILE") 2>&1

# 始终从项目根目录运行
cd "$PROJECT_DIR"

# --------------------------------------------------------------------------- #
#  工具函数                                                                      #
# --------------------------------------------------------------------------- #
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }

done_flag() { echo "$STATE_DIR/$1.done"; }

is_done() {
    [[ -f "$(done_flag "$1")" ]]
}

mark_done() {
    echo "$(date -Iseconds)" > "$(done_flag "$1")"
    log "✓ 阶段 [$1] 完成"
}

# 打印 GPU 状态（不阻塞，出错不中断）
gpu_status() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader 2>/dev/null \
        | awk '{printf "  GPU%s\n", $0}' || true
}

# 检测 SFT/DPO 检查点目录中最新的 checkpoint-* 子目录
latest_checkpoint() {
    local dir="$1"
    local latest
    latest=$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    echo "$latest"
}

# --------------------------------------------------------------------------- #
#  激活 Conda 环境                                                               #
# --------------------------------------------------------------------------- #
CONDA_BASE="$(conda info --base 2>/dev/null || echo '/home/runnel/miniconda3')"
# conda.sh 内部可能引用未声明变量，临时关闭 -u
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate tangut-nlp
set -u
log "Python: $(which python3)  ($(python3 --version 2>&1))"

# --------------------------------------------------------------------------- #
#  --reset 选项                                                                  #
# --------------------------------------------------------------------------- #
if [[ "${1:-}" == "--reset" ]]; then
    warn "正在清除所有 state/*.done 文件，将从头重跑整个流水线。"
    read -rp "确认？(yes/no): " confirm
    [[ "$confirm" == "yes" ]] || { log "已取消。"; exit 0; }
    rm -f "$STATE_DIR"/*.done
    log "状态已清除。"
fi

echo ""
echo "============================================================"
echo "  Tangut-NLP Pipeline  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  日志: $LOG_FILE"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------- #
#  阶段 00: 环境依赖                                                             #
# --------------------------------------------------------------------------- #
if ! is_done "00_env"; then
    log "=== 阶段 00: 安装 Python 依赖 ==="
    bash scripts/setup_env.sh
    mark_done "00_env"
else
    log "→ 跳过 [00_env]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 01: 下载模型与数据集                                                       #
# --------------------------------------------------------------------------- #
if ! is_done "01_models"; then
    log "=== 阶段 01: 下载模型与数据集 ==="

    # Qwen2.5-7B-Instruct
    if [[ ! -f "$MAIN_MODEL_PATH/config.json" ]]; then
        log "  下载 Qwen2.5-7B-Instruct ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct',
    local_dir='$MAIN_MODEL_PATH',
    ignore_patterns=['*.gguf', '*.pt'],
    resume_download=True)
print('Qwen2.5-7B-Instruct OK')
"
    else
        log "  Qwen2.5-7B-Instruct 已存在，跳过"
    fi

    # Qwen2.5-0.5B（PPL 打分）
    if [[ ! -f "$PPL_MODEL_PATH/config.json" ]]; then
        log "  下载 Qwen2.5-0.5B ..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B',
    local_dir='$PPL_MODEL_PATH',
    ignore_patterns=['*.gguf', '*.pt'],
    resume_download=True)
print('Qwen2.5-0.5B OK')
"
    else
        log "  Qwen2.5-0.5B 已存在，跳过"
    fi

    # xmj2002/Chinese_modern_classical（文言文↔白话文，97万对）
    if [[ ! -d "data/raw/ancient_chinese_hf" ]]; then
        log "  下载 xmj2002/Chinese_modern_classical (~972K 对) ..."
        python3 -c "
from datasets import load_dataset
ds = load_dataset('xmj2002/Chinese_modern_classical', split='train')
ds.save_to_disk('data/raw/ancient_chinese_hf')
print(f'Chinese_modern_classical: {len(ds)} 条')
"
    else
        log "  ancient_chinese_hf 已存在，跳过"
    fi

    mark_done "01_models"
else
    log "→ 跳过 [01_models]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 02: 划分数据集（train / dev / test）                                       #
# --------------------------------------------------------------------------- #
if ! is_done "02_data_splits"; then
    log "=== 阶段 02: 准备数据切分 ==="
    python3 src/prepare_splits.py \
        --input       data/raw/tangut_output.jsonl \
        --test-out    data/eval/test_set.jsonl \
        --dev-out     data/eval/dev_set.jsonl \
        --train-out   data/sft/babelstone_sft.jsonl \
        --test-size   50 \
        --dev-size    41 \
        --seed        42
    mark_done "02_data_splits"
else
    log "→ 跳过 [02_data_splits]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 03: Baseline 1 —— 零样本推理                                              #
# --------------------------------------------------------------------------- #
if ! is_done "03_baseline1"; then
    log "=== 阶段 03: Baseline 1 — Zero-shot 推理 ==="
    gpu_status

    python3 experiments/baseline1_zeroshot.py \
        --test-set   data/eval/test_set.jsonl \
        --model-path "$MAIN_MODEL_PATH" \
        --output     results/baseline1/predictions.jsonl \
        --tensor-parallel 2

    log "  计算 Baseline 1 评测指标 ..."
    python3 -m eval.run_all_metrics \
        --predictions results/baseline1/predictions.jsonl \
        --test-set    data/eval/test_set.jsonl \
        --reward-dict data/dictionary/reward_dict.json \
        --ppl-model   "$PPL_MODEL_PATH" \
        --output      results/baseline1/metrics.json

    gpu_status
    mark_done "03_baseline1"
else
    log "→ 跳过 [03_baseline1]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 04: Baseline 2 —— 字典 RAG 推理                                           #
# --------------------------------------------------------------------------- #
if ! is_done "04_baseline2"; then
    log "=== 阶段 04: Baseline 2 — Dictionary-RAG 推理 ==="
    gpu_status

    python3 experiments/baseline2_dict_rag.py \
        --test-set   data/eval/test_set.jsonl \
        --dict-path  data/dictionary/dictionary.json \
        --model-path "$MAIN_MODEL_PATH" \
        --output     results/baseline2/predictions.jsonl \
        --tensor-parallel 2

    log "  计算 Baseline 2 评测指标 ..."
    python3 -m eval.run_all_metrics \
        --predictions results/baseline2/predictions.jsonl \
        --test-set    data/eval/test_set.jsonl \
        --reward-dict data/dictionary/reward_dict.json \
        --ppl-model   "$PPL_MODEL_PATH" \
        --output      results/baseline2/metrics.json

    gpu_status
    mark_done "04_baseline2"
else
    log "→ 跳过 [04_baseline2]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 05a: 生成合成 SFT 数据                                                    #
# --------------------------------------------------------------------------- #
if ! is_done "05a_synthetic"; then
    log "=== 阶段 05a: 生成合成伪平行语料 (max=$SYNTHETIC_MAX) ==="
    python3 src/data_synthesis.py \
        --dictionary-path      data/dictionary/dictionary.json \
        --ancient-chinese-path data/raw/ancient_chinese_hf \
        --output               data/sft/synthetic_sft.jsonl \
        --max-samples          "$SYNTHETIC_MAX" \
        --seed                 42
    mark_done "05a_synthetic"
else
    log "→ 跳过 [05a_synthetic]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 05b: 合并真实 + 合成数据                                                   #
# --------------------------------------------------------------------------- #
if ! is_done "05b_combine"; then
    log "=== 阶段 05b: 合并数据（upsample_real=$UPSAMPLE_REAL）==="
    python3 src/combine_data.py \
        --real          data/sft/babelstone_sft.jsonl \
        --synthetic     data/sft/synthetic_sft.jsonl \
        --output        data/sft/combined_sft.jsonl \
        --upsample-real "$UPSAMPLE_REAL" \
        --seed          42
    mark_done "05b_combine"
else
    log "→ 跳过 [05b_combine]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 05c: SFT 训练                                                             #
# --------------------------------------------------------------------------- #
if ! is_done "05c_sft_train"; then
    log "=== 阶段 05c: SFT 训练（LoRA, 2xA100, DeepSpeed ZeRO-2）==="
    gpu_status

    SFT_RESUME_ARG=""
    SFT_CKPT=$(latest_checkpoint "checkpoints/sft")
    if [[ -n "$SFT_CKPT" ]]; then
        warn "  检测到 SFT 中断检查点: $SFT_CKPT，自动续训"
        SFT_RESUME_ARG="--resume $SFT_CKPT"
    fi

    accelerate launch \
        --num_processes 2 \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        experiments/baseline3_synthetic_sft.py \
            --train-data   data/sft/combined_sft.jsonl \
            --model-path   "$MAIN_MODEL_PATH" \
            --output-dir   checkpoints/sft \
            --epochs       "$SFT_EPOCHS" \
            --batch-size   "$SFT_BATCH" \
            --grad-accum   "$SFT_GRAD_ACCUM" \
            --lr           "$SFT_LR" \
            --lora-rank    "$SFT_LORA_RANK" \
            $SFT_RESUME_ARG

    gpu_status
    mark_done "05c_sft_train"
else
    log "→ 跳过 [05c_sft_train]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 05d: 合并 LoRA + Baseline 3 推理评测                                      #
# --------------------------------------------------------------------------- #
if ! is_done "05d_sft_inference"; then
    log "=== 阶段 05d: 合并 LoRA → Baseline 3 推理评测 ==="
    gpu_status

    python3 experiments/inference.py \
        --model       checkpoints/sft/final \
        --test-set    data/eval/test_set.jsonl \
        --output      results/baseline3/predictions.jsonl \
        --method-name baseline3_sft \
        --tensor-parallel 2

    log "  计算 Baseline 3 评测指标 ..."
    python3 -m eval.run_all_metrics \
        --predictions results/baseline3/predictions.jsonl \
        --test-set    data/eval/test_set.jsonl \
        --reward-dict data/dictionary/reward_dict.json \
        --ppl-model   "$PPL_MODEL_PATH" \
        --output      results/baseline3/metrics.json

    gpu_status
    mark_done "05d_sft_inference"
else
    log "→ 跳过 [05d_sft_inference]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 06a: 生成 DPO 候选对                                                      #
# --------------------------------------------------------------------------- #
if ! is_done "06a_dpo_candidates"; then
    log "=== 阶段 06a: 生成 DPO 候选对（N=5, vLLM + reward scoring）==="
    gpu_status

    # 需要有 merged 权重（inference.py 在 05d 已自动 merge）
    MERGED_PATH="checkpoints/sft/merged"
    if [[ ! -d "$MERGED_PATH" ]]; then
        err "  找不到合并后的模型 $MERGED_PATH，请确认 05d 阶段已完成。"
        exit 1
    fi

    python3 experiments/generate_candidates.py \
        --sft-model      "$MERGED_PATH" \
        --train-data     data/sft/combined_sft.jsonl \
        --reward-dict    data/dictionary/reward_dict.json \
        --ppl-model      "$PPL_MODEL_PATH" \
        --output         data/dpo/dpo_pairs.jsonl \
        --max-inputs     "$DPO_MAX_INPUTS" \
        --tensor-parallel 2

    log "  DPO 对数: $(wc -l < data/dpo/dpo_pairs.jsonl)"
    gpu_status
    mark_done "06a_dpo_candidates"
else
    log "→ 跳过 [06a_dpo_candidates]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 06b: DPO 训练                                                             #
# --------------------------------------------------------------------------- #
if ! is_done "06b_dpo_train"; then
    log "=== 阶段 06b: DPO 训练（单个 A100）==="
    gpu_status

    DPO_RESUME_ARG=""
    DPO_CKPT=$(latest_checkpoint "checkpoints/dpo")
    if [[ -n "$DPO_CKPT" ]]; then
        warn "  检测到 DPO 中断检查点: $DPO_CKPT，自动续训"
        DPO_RESUME_ARG="--resume $DPO_CKPT"
    fi

    CUDA_VISIBLE_DEVICES=0 python3 experiments/final_dpo.py \
            --dpo-data   data/dpo/dpo_pairs.jsonl \
            --sft-model  checkpoints/sft/merged \
            --output-dir checkpoints/dpo \
            --epochs     "$DPO_EPOCHS" \
            --batch-size "$DPO_BATCH" \
            --grad-accum "$DPO_GRAD_ACCUM" \
            --lr         "$DPO_LR" \
            --beta       "$DPO_BETA" \
            $DPO_RESUME_ARG

    gpu_status
    mark_done "06b_dpo_train"
else
    log "→ 跳过 [06b_dpo_train]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 06c: DPO 推理评测（最终方案）                                               #
# --------------------------------------------------------------------------- #
if ! is_done "06c_dpo_inference"; then
    log "=== 阶段 06c: DPO 模型推理 + 最终评测 ==="
    gpu_status

    python3 experiments/inference.py \
        --model       checkpoints/dpo/final \
        --test-set    data/eval/test_set.jsonl \
        --output      results/final/predictions.jsonl \
        --method-name final_dpo \
        --tensor-parallel 2

    log "  计算最终方案评测指标 ..."
    python3 -m eval.run_all_metrics \
        --predictions results/final/predictions.jsonl \
        --test-set    data/eval/test_set.jsonl \
        --reward-dict data/dictionary/reward_dict.json \
        --ppl-model   "$PPL_MODEL_PATH" \
        --output      results/final/metrics.json

    gpu_status
    mark_done "06c_dpo_inference"
else
    log "→ 跳过 [06c_dpo_inference]（已完成）"
fi

# --------------------------------------------------------------------------- #
#  阶段 07: 汇总对比结果                                                           #
# --------------------------------------------------------------------------- #
log "=== 阶段 07: 汇总对比结果 ==="
python3 -m eval.aggregate_results
log "  已生成 results/comparison.json、results/comparison.csv、results/comparison.png"

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  完整日志: $LOG_FILE"
echo "============================================================"
echo ""
cat results/comparison.json 2>/dev/null || true
