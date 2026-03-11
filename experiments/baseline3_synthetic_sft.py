"""Baseline 3: Supervised Fine-Tuning on synthetic Tangut-Chinese pairs with LoRA."""

import sys
import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import SYSTEM_SFT, build_sft_sample


def format_sample(example):
    user = f"{example['instruction']}\n{example['input']}"
    return {"text": build_sft_sample(SYSTEM_SFT, user, example["output"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/sft/combined_sft.jsonl")
    parser.add_argument("--model-path", default="models/qwen2.5-7b-instruct")
    parser.add_argument("--output-dir", default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.train_data, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    formatted = [format_sample(d) for d in raw_data]
    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} training samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        deepspeed="configs/deepspeed_zero2.json",
        dataloader_num_workers=4,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"SFT training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
