#!/bin/bash
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
