"""LoRA Training for Insurance AI Agent — Qwen2.5-0.5B-Instruct"""
import json
from pathlib import Path
from datasets import Dataset   # 必须在 torch 之前导入（避免 DLL 冲突）
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT = Path(__file__).resolve().parent.parent.parent  # finetune/scripts/ → 项目根目录
MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\models\Qwen--Qwen2.5-0.5B-Instruct\snapshots\master"
DATA_DIR = ROOT / "finetune" / "data"
OUTPUT_DIR = ROOT / "finetune" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Data ──
def load_alpaca(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for item in data:
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("output", "")
        if inp:
            text = f"<|im_start|>user\n{inst}\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        else:
            text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        texts.append({"text": text})
    return Dataset.from_list(texts)

print("[1/5] Loading data...")
train_ds = load_alpaca(DATA_DIR / "insurance_train.json")
test_ds = load_alpaca(DATA_DIR / "insurance_test.json")
print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

# ── Load Model ──
print("[2/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
model.enable_input_require_grads()
model.config.use_cache = False
print(f"  Base params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

# ── LoRA ──
print("[3/5] Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Tokenize ──
def tokenize(examples):
    r = tokenizer(examples["text"], truncation=True, max_length=512, padding=False)
    r["labels"] = r["input_ids"].copy()
    return r

tok_train = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
tok_test = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)

# ── Train ──
print("[4/5] Training...")
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=50,
    fp16=True,
    gradient_checkpointing=True,
    report_to=[],
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tok_train, eval_dataset=tok_test,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# ── Save ──
print("[5/5] Saving adapter...")
adapter_path = OUTPUT_DIR / "adapter"
model.save_pretrained(str(adapter_path))
tokenizer.save_pretrained(str(adapter_path))
print(f"  Done: {adapter_path}")
print("\n=== Training Complete! ===")
