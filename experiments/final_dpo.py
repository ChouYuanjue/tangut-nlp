"""Final method: DPO training on preference pairs from SFT model."""

import sys
import os
import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import SYSTEM_SFT

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def enable_resume_compat_if_needed(resume_path):
    if not resume_path:
        return

    # PyTorch 2.6 changed torch.load default to weights_only=True, which can
    # break loading legacy Trainer RNG state when resuming.
    version_core = torch.__version__.split("+")[0]
    parts = version_core.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if (major, minor) >= (2, 6):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        print(
            "[resume] Enabled TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for trusted checkpoint compatibility."
        )


def sync_special_token_ids(model, tokenizer):
    # Keep model config and generation config explicitly aligned with tokenizer.
    token_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for key, value in token_ids.items():
        setattr(model.config, key, value)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            setattr(model.generation_config, key, value)


def format_dpo_sample(example):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_SFT}<|im_end|>\n"
        f"<|im_start|>user\n{example['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "prompt": prompt,
        "chosen": example["chosen"] + "<|im_end|>",
        "rejected": example["rejected"] + "<|im_end|>",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-data", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument("--sft-model", default="checkpoints/sft/merged")
    parser.add_argument("--output-dir", default="checkpoints/dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--loss-type",
        nargs="+",
        default=["sigmoid"],
        help="TRL DPO loss type(s), e.g. sigmoid, robust, ipo.",
    )
    parser.add_argument(
        "--loss-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional weights for multi-loss training. Must match --loss-type length.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Noise rate / label smoothing used by robust DPO variants.",
    )
    parser.add_argument(
        "--use-weighting",
        action="store_true",
        help="Enable WPO-style weighting for off-policy preference data.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    enable_resume_compat_if_needed(args.resume)

    with open(args.dpo_data, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    formatted = [format_dpo_sample(d) for d in raw]
    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} DPO pairs")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    sync_special_token_ids(model, tokenizer)

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    sync_special_token_ids(ref_model, tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        max_length=args.max_length,
        gradient_checkpointing=True,
        report_to="none",
        loss_type=args.loss_type,
        loss_weights=args.loss_weights,
        label_smoothing=args.label_smoothing,
        use_weighting=args.use_weighting,
        seed=args.seed,
        run_name=args.run_name,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"DPO training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
