#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
專題統一設定檔 (config.py)
所有路徑、常數、模型設定集中管理，避免跨檔案硬編碼。
"""

import os

# ==========================================
# 專案根目錄 (自動偵測，以本檔案所在位置為基準)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 路徑設定
# ==========================================
# HSK 詞庫
HSK_FOLDER       = os.path.join(PROJECT_ROOT, "New HSK (2025)", "HSK Words")
HSK_FOLDER_JSON  = os.path.join(HSK_FOLDER, "new")          # JSON 版 (1.json~7.json)
HSK_VOCAB_TXT    = os.path.join(PROJECT_ROOT, "hsk_vocab.txt")

# TBCL 詞庫
TBCL_PATH        = os.path.join(PROJECT_ROOT, "hsk詞彙", "14452詞語表202504.xlsx - 14452詞語表202504.csv")

# 訓練資料
TRAINING_DATA    = os.path.join(PROJECT_ROOT, "完成標註", "training_data_v5.jsonl")

# LoRA 權重
LORA_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "lora_weights")
LORA_PATH        = os.path.join(LORA_WEIGHTS_DIR, "full_train", "final_lora")

# 課本來源
TEXTBOOK_TXT_DIR = os.path.join(PROJECT_ROOT, "課本txt")
ANNOTATION_DIR   = os.path.join(PROJECT_ROOT, "標註模板")

# 評估輸出
EVAL_REPORT      = os.path.join(PROJECT_ROOT, "eval_report_v2_fixed.json")
EVAL_CHART       = os.path.join(PROJECT_ROOT, "eval_comparison_fixed.png")
ABLATION_RESULTS = os.path.join(PROJECT_ROOT, "ablation_results.json")

# ==========================================
# 模型設定
# ==========================================
BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ==========================================
# HSK 等級映射 (TXT 檔案名 → 等級)
# ==========================================
LEVEL_MAPPING = {
    "HSK_Level_1_words.txt": 1,
    "HSK_Level_2_words.txt": 2,
    "HSK_Level_3_words.txt": 3,
    "HSK_Level_4_words.txt": 4,
    "HSK_Level_5_words.txt": 5,
    "HSK_Level_6_words.txt": 6,
    "HSK_Level_7-9_words.txt": 7,
}

# ==========================================
# 簡化等級設定
# ==========================================
LEVEL_LIMITS = {1: 20, 2: 15, 3: 12}

LEVEL_INSTRUCTION = {
    1: (
        "請將以下課文簡化，每句不超過20個字。"
        "規則：①只保留原文有的資訊 ②不可加入原文沒有的句子 "
        "③遇到逗號必須拆成獨立句子。"
    ),
    2: (
        "請將以下文本進一步簡化，每句不超過15個字。"
        "規則：①不可新增原文沒有的內容 ②刪除次要細節但不改變語意。"
    ),
    3: (
        "請將以下文本簡化為最精簡版本，每句不超過12個字。"
        "規則：①只保留主詞＋動作＋結果 ②禁止任何新增內容。"
    ),
}

# convert_to_training_v3 用的稍有不同的 L1 instruction
LEVEL_INSTRUCTION_TRAINING = {
    1: (
        "請將以下課文簡化為適合閱讀障礙學生閱讀的版本，"
        "每句不超過20個字，遇到逗號或分號時必須拆成獨立句子，"
        "詞彙使用國小4年級程度。"
    ),
    2: "請將以下文本進一步簡化，每句不超過15個字，使用更基礎的詞彙。",
    3: "請將以下文本簡化為最精簡版本，每句不超過12個字，使用最基礎的詞彙。",
}
