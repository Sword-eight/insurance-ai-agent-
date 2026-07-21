"""
Insurance AI Agent — LoRA Training Script
基于 peft + transformers + datasets 的独立训练脚本。
不依赖 LLaMA-Factory CLI。

用法:
  python finetune/scripts/train.py
  python finetune/scripts/train.py --epochs 5 --lr 1e-4
"""

import sys, os, json, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

from config import FINETUNE_CONFIG
from utils.logger import get_logger

logger = get_logger("finetune.train")

# ────────────────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────────────────

DATA_DIR = Path(FINETUNE_CONFIG["dataset_dir"])
OUTPUT_DIR = Path(FINETUNE_CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "insurance_train.json"
TEST_FILE = DATA_DIR / "insurance_test.json"


def load_dataset(file_path: Path) -> Dataset:
    """加载 Alpaca 格式数据集并转换为 HuggingFace Dataset。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = []
    for item in data:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")

        # Alpaca 格式 → 对话文本
        if inp:
            text = f"<|im_start|>user\n{instruction}\n{inp}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize 文本。"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def train(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 1,
    grad_accum: int = 16,
    max_length: int = 1024,
    fp16: bool = True,
) -> PeftModel:
    """
    执行 LoRA 微调。

    Args:
        model_name: 基础模型名称
        lora_rank: LoRA 秩
        lora_alpha: LoRA 缩放因子
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 每设备批次大小
        grad_accum: 梯度累积步数
        max_length: 最大序列长度
        fp16: 是否使用 fp16

    Returns:
        微调后的 PeftModel
    """
    logger.info(f"Loading model: {model_name}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False  # 梯度检查点需要

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    logger.info(f"Loading training data from {TRAIN_FILE}")
    train_ds = load_dataset(TRAIN_FILE)
    tokenized_train = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    eval_ds = None
    if TEST_FILE.exists():
        logger.info(f"Loading eval data from {TEST_FILE}")
        eval_ds = load_dataset(TEST_FILE)
        tokenized_eval = eval_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_length),
            batched=True,
            remove_columns=eval_ds.column_names,
        )
    else:
        tokenized_eval = None

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if eval_ds else "loss",
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=100 if eval_ds else None,
        fp16=fp16,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # 输出
        logging_dir=str(OUTPUT_DIR / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
        ),
    )

    # 开始训练
    logger.info(f"Starting LoRA training: {epochs} epochs, {len(tokenized_train)} samples")
    trainer.train()

    # 保存最终 adapter
    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"LoRA adapter saved to {adapter_path}")

    return model


def print_summary(adapter_path: str) -> None:
    """打印训练总结。"""
    print("\n" + "=" * 60)
    print("  ✅ LoRA 训练完成")
    print("=" * 60)
    print(f"  Adapter 路径: {adapter_path}")
    print(f"  使用方式:")
    print(f"    from peft import PeftModel")
    print(f"    from transformers import AutoModelForCausalLM")
    print(f"    model = AutoModelForCausalLM.from_pretrained(...)")
    print(f"    model = PeftModel.from_pretrained(model, '{adapter_path}')")
    print("=" * 60)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoRA 微调保险模型")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_true", default=False)
    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    model = train(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        fp16=args.fp16,
    )

    print_summary(str(OUTPUT_DIR / "adapter"))
