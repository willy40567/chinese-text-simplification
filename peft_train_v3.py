import gc
import json
import os
import torch
import argparse
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ============================================================
# 0. 基本設定 (路徑從 config.py 統一管理)
# ============================================================
from config import BASE_MODEL, TRAINING_DATA as DATA_PATH, LORA_WEIGHTS_DIR as OUTPUT_DIR
from model_utils import get_bnb_config

LOG_DIR = "peft_logs/"
MAX_LENGTH = 512
SEED = 42

# 4-bit 量化設定（RTX 4070 12GB 必要）
BNB_CONFIG = get_bnb_config()

# LoRA 超參數（小樣本優化配置）
LORA_CONFIG = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 8,
    lora_alpha = 32,
    lora_dropout = 0.15,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
    bias = "none",
)

# 訓練超參數 (全量訓練設定)
TRAIN_ARGS = dict(
    num_train_epochs = 3,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 2e-4,
    warmup_steps = 5, # 【修改】從 3 改為 5，約佔總 steps 的 16%
    max_grad_norm = 1.0,
    fp16 = True,
    gradient_checkpointing = True,
    logging_steps = 5,
    save_strategy = "epoch",
    eval_strategy = "no", # 無驗證集，不進行 eval
    report_to = "none",
    seed = SEED,
)

# ============================================================
# 1. Prompt 格式化
# ============================================================
def format_prompt(sample, tokenizer, max_length):
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{sample['instruction']}\n\n{sample['input']}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{sample['output']}"
        f"<|eot_id|>"
    )
    tokenized = tokenizer(
        prompt,
        max_length = max_length,
        truncation = True,
        padding = "max_length",
        return_tensors = "pt"
    )
    input_ids = tokenized["input_ids"][0]
    labels = input_ids.clone()

    # 遮蔽 user 段落，只對 assistant 輸出計算 loss
    user_end_token = tokenizer.encode(
        "<|eot_id|>", add_special_tokens=False
    )[-1]
    assistant_start = (input_ids == user_end_token).nonzero(as_tuple=True)[0]
    if len(assistant_start) > 0:
        labels[:assistant_start[0]+1] = -100

    return {
        "input_ids" : input_ids,
        "attention_mask" : tokenized["attention_mask"][0],
        "labels" : labels
    }

# ============================================================
# 2. 訓練核心模組
# ============================================================
def train_model(run_name, train_samples, tokenizer):
    print(f"\n{'='*60}")
    print(f"[{run_name}] 全量訓練:{len(train_samples)} 筆 (無驗證集)")
    print(f"{'='*60}")

    # 建立 Dataset
    train_data = [format_prompt(s, tokenizer, MAX_LENGTH) for s in train_samples]
    train_ds = Dataset.from_list(train_data)

    # 載入基礎模型並套用 LoRA
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = BNB_CONFIG,
        device_map = "auto",
        max_memory = {0: "11GiB", "cpu": "30GiB"},
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    run_output = os.path.join(OUTPUT_DIR, run_name)
    run_log = os.path.join(LOG_DIR, run_name)

    args = TrainingArguments(
        output_dir = run_output,
        logging_dir = run_log,
        **TRAIN_ARGS
    )

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_ds,
        data_collator = default_data_collator,
    )

    trainer.train()

    # 儲存 LoRA 權重
    # 【注意】後續推論/評估腳本請指向此路徑：lora_weights/full_train/final_lora/
    best_path = os.path.join(run_output, "final_lora")
    model.save_pretrained(best_path)
    print(f"[{run_name}] 模型已儲存 → {best_path}")

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 3. 主程式：全量訓練
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PEFT Llama-3 訓練腳本")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="訓練資料路徑")
    args = parser.parse_args()

    data_path = args.data

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # 載入 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # 全量訓練：不做 train/val split
    print(f"\n{'='*60}")
    print(f"啟動全量訓練 (Full Train) 模式")
    print(f"載入資料: {data_path}")
    print(f"{'='*60}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line.strip()) for line in f]
        
    # 【修改】移除冗餘變數，直接將 all_samples 傳入
    train_model("full_train", all_samples, tokenizer)
    
    print("\n全量訓練完成！")

if __name__ == "__main__":
    main()