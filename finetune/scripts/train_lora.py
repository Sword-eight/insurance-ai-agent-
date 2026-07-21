"""
Insurance AI Agent - LoRA 训练配置生成模块
为 LLaMA-Factory 生成 dataset_info.json、训练脚本和 LoRA 配置。

不实际调用训练，仅生成配置文件供 LLaMA-Factory 使用。

使用:
  python -m finetune.scripts.train_lora            # 生成配置
  python -m finetune.scripts.train_lora --dry-run  # 预览不保存
"""

import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FINETUNE_CONFIG, PROJECT_ROOT as ROOT
from utils.logger import get_logger

logger = get_logger("finetune.train_lora")


def generate_dataset_info() -> Dict[str, Any]:
    """生成 LLaMA-Factory 的 dataset_info.json 内容。"""
    data_dir = Path(FINETUNE_CONFIG["dataset_dir"]).resolve()

    return {
        "insurance_train": {
            "file_name": str(data_dir / "insurance_train.json"),
            "formatting": "sharegpt",
            "columns": {
                "messages": ["instruction", "input", "output"],
            },
        },
        "insurance_test": {
            "file_name": str(data_dir / "insurance_test.json"),
            "formatting": "sharegpt",
            "columns": {
                "messages": ["instruction", "input", "output"],
            },
        },
    }


def generate_train_config() -> Dict[str, Any]:
    """生成 LLaMA-Factory 训练 YAML 配置。"""
    return {
        # 模型
        "model_name_or_path": FINETUNE_CONFIG["base_model"],
        # 数据集
        "dataset": "insurance_train",
        "eval_dataset": "insurance_test",
        "dataset_dir": str(Path(FINETUNE_CONFIG["dataset_dir"]).resolve()),
        "template": "qwen",
        # LoRA
        "finetuning_type": "lora",
        "lora_rank": FINETUNE_CONFIG["lora_rank"],
        "lora_alpha": FINETUNE_CONFIG["lora_alpha"],
        "lora_dropout": FINETUNE_CONFIG["lora_dropout"],
        "lora_target": FINETUNE_CONFIG["lora_target"],
        # 训练超参
        "num_train_epochs": FINETUNE_CONFIG["epochs"],
        "learning_rate": FINETUNE_CONFIG["learning_rate"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "per_device_train_batch_size": FINETUNE_CONFIG["batch_size"],
        "per_device_eval_batch_size": FINETUNE_CONFIG["batch_size"],
        "gradient_accumulation_steps": FINETUNE_CONFIG["gradient_accumulation"],
        "max_length": FINETUNE_CONFIG["max_length"],
        # 评估
        "do_eval": True,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        # 精度
        "bf16": True,
        "fp16": False,
        # 日志
        "logging_steps": 10,
        "report_to": ["tensorboard"],
        "output_dir": str(Path(FINETUNE_CONFIG["output_dir"]).resolve()),
        "overwrite_output_dir": True,
    }


def generate_train_script() -> str:
    """生成 LLaMA-Factory 训练命令行脚本。"""
    return r"""#!/bin/bash
# ============================================================
# Insurance AI Agent - LoRA 训练脚本
# 使用 LLaMA-Factory 微调 Qwen2.5-1.5B-Instruct
#
# 前置条件:
#   1. 已安装 LLaMA-Factory:
#      git clone https://github.com/hiyouga/LLaMA-Factory.git
#      cd LLaMA-Factory
#      pip install -e .
#
#   2. 已将 insurance_train.json / insurance_test.json
#      放入 LLaMA-Factory/data/ 目录
#
#   3. 已将 finetune/scripts/llamafactory_dataset_info.json
#      的内容合并到 LLaMA-Factory/data/dataset_info.json
#
# 使用:
#   bash finetune/scripts/train_lora.sh
# ============================================================

# 进入 LLaMA-Factory 目录（根据实际路径修改）
LLAMA_FACTORY_DIR="./LLaMA-Factory"

cd $LLAMA_FACTORY_DIR

# LoRA 微调
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset insurance_train \
    --eval_dataset insurance_test \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --bf16 True \
    --logging_steps 10 \
    --report_to tensorboard \
    --output_dir ./output/insurance_lora \
    --overwrite_output_dir

echo "✅ LoRA 训练完成"
echo "   Adapter 路径: ./output/insurance_lora"

# 导出 LoRA adapter（可选，用于推理）
# llamafactory-cli export \
#     --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
#     --adapter_name_or_path ./output/insurance_lora \
#     --template qwen \
#     --finetuning_type lora \
#     --export_dir ./output/insurance_lora_merged \
#     --export_size 2 \
#     --export_legacy_format False
"""


def generate_all(dry_run: bool = False) -> Dict[str, Any]:
    """生成所有配置文件。"""
    configs = {
        "dataset_info": generate_dataset_info(),
        "train_config": generate_train_config(),
        "train_script": generate_train_script(),
    }

    if not dry_run:
        import json
        from utils.helpers import save_json

        output_dir = Path(FINETUNE_CONFIG["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 dataset_info.json（供 LLaMA-Factory 使用）
        dataset_info_path = Path(__file__).resolve().parent / "llamafactory_dataset_info.json"
        save_json(str(dataset_info_path), configs["dataset_info"])
        logger.info(f"  ✅ dataset_info → {dataset_info_path}")

        # 保存训练配置参考
        config_path = output_dir / "lora_train_config.json"
        save_json(str(config_path), configs["train_config"])
        logger.info(f"  ✅ train_config → {config_path}")

        # 保存训练脚本
        script_path = Path(__file__).resolve().parent / "train_lora.sh"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(configs["train_script"])
        logger.info(f"  ✅ train_script → {script_path}")

        # 打印摘要
        print_summary(configs)

    return configs


def print_summary(configs: Dict[str, Any]) -> None:
    """打印训练配置摘要。"""
    cfg = configs["train_config"]
    print("\n" + "=" * 60)
    print("  🏋️  LoRA 训练配置摘要")
    print("=" * 60)
    print(f"  基础模型:     {cfg['model_name_or_path']}")
    print(f"  微调方式:     {cfg['finetuning_type']}")
    print(f"  LoRA Rank:    {cfg['lora_rank']}")
    print(f"  LoRA Alpha:   {cfg['lora_alpha']}")
    print(f"  Epochs:       {cfg['num_train_epochs']}")
    print(f"  Learning Rate:{cfg['learning_rate']}")
    print(f"  Batch Size:   {cfg['per_device_train_batch_size']}")
    print(f"  Grad Accum:   {cfg['gradient_accumulation_steps']}")
    print(f"  Max Length:   {cfg['max_length']}")
    print(f"  输出目录:     {cfg['output_dir']}")
    print("=" * 60)
    print(f"\n  下一步:")
    print(f"  1. 将 dataset_info.json 合并到 LLaMA-Factory/data/dataset_info.json")
    print(f"  2. 将 insurance_train.json / insurance_test.json 复制到 LLaMA-Factory/data/")
    print(f"  3. bash finetune/scripts/train_lora.sh")
    print()


# ────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成 LLaMA-Factory 训练配置")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不保存文件")
    args = parser.parse_args()

    generate_all(dry_run=args.dry_run)
