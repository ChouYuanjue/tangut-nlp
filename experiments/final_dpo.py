"""Final method: DPO training on preference pairs from SFT model."""

import sys
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.dpo_data, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    formatted = [format_dpo_sample(d) for d in raw]
    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} DPO pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    ref_model.config.pad_token_id = tokenizer.pad_token_id

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
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        gradient_checkpointing=True,
        deepspeed="configs/deepspeed_zero2.json",
        report_to="none",
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
