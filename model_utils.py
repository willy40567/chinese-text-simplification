#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具模組 (model_utils.py)
統一管理 BitsAndBytes 量化設定、模型載入/卸載、CUDA 記憶體監控。
"""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import BASE_MODEL, LORA_PATH


# ==========================================
# 量化設定
# ==========================================
def get_bnb_config():
    """RTX 4070 12GB 適用的 4-bit NF4 量化設定"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ==========================================
# 模型載入
# ==========================================
def load_base_model(base_model=BASE_MODEL, max_memory=None):
    """載入 base model (不含 LoRA)，回傳 (model, tokenizer)"""
    if max_memory is None:
        max_memory = {0: "11GiB", "cpu": "30GiB"}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=get_bnb_config(),
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(base_model=BASE_MODEL, lora_path=LORA_PATH, max_memory=None):
    """載入 base model + LoRA 權重，回傳 (model, tokenizer)"""
    from peft import PeftModel

    model, tokenizer = load_base_model(base_model, max_memory)
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


# ==========================================
# 模型卸載
# ==========================================
def unload_model(model):
    """安全釋放模型佔用的 GPU / CPU 記憶體"""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# CUDA 記憶體監控
# ==========================================
def log_cuda_memory(prefix=""):
    """印出當前 CUDA 記憶體使用量"""
    if not torch.cuda.is_available():
        print(f"{prefix} CUDA not available")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{prefix} CUDA allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
