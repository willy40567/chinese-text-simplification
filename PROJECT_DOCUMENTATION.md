# 中文文本簡化系統 — 專題完整技術文件

> 針對閱讀障礙學生的台灣國小課文三級漸進式簡化系統  
> 基於 Llama-3.1-8B-Instruct + LoRA 微調

---

## 目錄

1. [專案概述](#1-專案概述)
2. [研究目標與簡化等級定義](#2-研究目標與簡化等級定義)
3. [技術架構總覽](#3-技術架構總覽)
4. [完整工作流程](#4-完整工作流程)
5. [核心檔案詳細說明](#5-核心檔案詳細說明)
6. [測試與輔助檔案說明](#6-測試與輔助檔案說明)
7. [資料格式規格](#7-資料格式規格)
8. [模型訓練配置詳解](#8-模型訓練配置詳解)
9. [評估指標說明](#9-評估指標說明)
10. [技術堆疊與依賴套件](#10-技術堆疊與依賴套件)
11. [目錄結構](#11-目錄結構)

---

## 1. 專案概述

本專案是一個**中文文本簡化（Chinese Text Simplification）**研究系統，核心目標為：

- 將**台灣國小四年級課文**自動簡化為適合**閱讀障礙學生**閱讀的版本
- 採用**三級漸進式簡化**策略（Level 1 → Level 2 → Level 3）
- 結合**人工標註**與 **AI 模型推論（LoRA 微調）** 雙軌流程
- 透過 SARI、BERTScore、句長合規率等指標進行**全面品質評估**

系統整合了從資料標註、模型訓練、推論生成、到評估分析的完整 pipeline，並提供 GUI 圖形化操作介面。

---

## 2. 研究目標與簡化等級定義

### 三級簡化標準

| 等級 | 最大字數/句 | 詞彙限制 | 說明 |
|:----:|:----------:|:--------:|:-----|
| **Level 1** | ≤ 20 字 | 國小四年級程度（HSK 1-4） | 閱讀障礙友善基礎版 |
| **Level 2** | ≤ 15 字 | HSK 1-3 等級詞彙 | 進階簡化版 |
| **Level 3** | ≤ 12 字 | HSK 1-2 等級詞彙（最基礎） | 極簡版 |

### 簡化規則

- **逐級遞進**：原文 → L1 → L2 → L3，每一級以上一級輸出為輸入
- **逗號/分號拆句**：遇到逗號或分號時必須拆成獨立句子（L1 規則）
- **資訊保留原則**：只保留原文有的資訊，不可加入原文沒有的句子
- **語意維持**：刪除次要細節但不改變原始語意
- **L3 極簡規則**：只保留「主詞 + 動作 + 結果」的核心結構

### 詞彙替換策略

系統內建同義詞替換表（`SYNONYM_TABLE`），將高級詞彙降級為低級同義詞：

| 原詞（高級） | 替換詞（低級） | 目標 HSK 等級 |
|:------------|:-------------|:------------:|
| 制造 | 做 | 1 |
| 觀察 | 看 | 1 |
| 發現 | 找到 | 2 |
| 建造 | 建 | 2 |
| 保護 | 愛護 | 3 |
| 創造 | 做出 | 1 |

同時維護補充詞典（`SUPPLEMENT_DICT`）處理 HSK 詞庫中未收錄但常見的詞彙，以及專有名詞白名單（`PROPER_NOUNS_SIMP`）避免誤判人名地名。

---

## 3. 技術架構總覽

```
┌──────────────────────────────────────────────────────────────┐
│                     系統架構總覽                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ 課文 TXT 檔 │───>│ batch_       │───>│ 標註模板 JSON │   │
│  │ (國小課本)   │    │ processor.py │    │               │   │
│  └─────────────┘    └──────────────┘    └───────┬───────┘   │
│                                                  │           │
│                                                  v           │
│                     ┌──────────────────────────────────┐     │
│                     │ gui_annotator.py /               │     │
│                     │ simplification_gui.py            │     │
│                     │ (人工標註 + AI 輔助)              │     │
│                     └──────────────┬───────────────────┘     │
│                                    │                         │
│                                    v                         │
│                     ┌──────────────────────────────────┐     │
│                     │ convert_to_training_v3.py        │     │
│                     │ (JSON → JSONL 訓練資料)          │     │
│                     └──────────────┬───────────────────┘     │
│                                    │                         │
│                                    v                         │
│                     ┌──────────────────────────────────┐     │
│                     │ peft_train_v3.py                 │     │
│                     │ (LoRA 微調 Llama-3.1-8B)        │     │
│                     └──────────────┬───────────────────┘     │
│                                    │                         │
│                          ┌─────────┼─────────┐               │
│                          v         v         v               │
│                    ┌─────────┐ ┌────────┐ ┌─────────┐       │
│                    │ LoRA    │ │ Base   │ │ Few-    │       │
│                    │ 微調版  │ │ Model  │ │ Shot    │       │
│                    └────┬────┘ └───┬────┘ └────┬────┘       │
│                         └──────────┼───────────┘             │
│                                    v                         │
│                     ┌──────────────────────────────────┐     │
│                     │ evaluate_v2.py                   │     │
│                     │ (SARI + BERTScore + 合規率)      │     │
│                     └──────────────┬───────────────────┘     │
│                                    │                         │
│                                    v                         │
│              ┌─────────────────────┼─────────────────────┐   │
│              v                     v                     v   │
│    ┌──────────────┐    ┌────────────────┐    ┌────────────┐ │
│    │hsk_coverage  │    │tbcl_analyzer   │    │readability │ │
│    │.py           │    │.py             │    │_tool.py    │ │
│    │(HSK覆蓋分析) │    │(TBCL黃金比例)  │    │(難度對比)  │ │
│    └──────────────┘    └────────────────┘    └────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 完整工作流程

### Phase 1：資料收集與前處理

**輸入**：台灣國小課本 TXT 檔案

**處理工具**：`batch_processor.py`

- 自動遍歷目錄內所有 `.txt` 檔案
- 使用正則表達式識別課文標題（如「一、逛夜市」「第三課」等）
- 按課文邊界分割內容，擷取段落
- 生成標註模板 JSON 檔案

```
課本 TXT → batch_processor.py → 標註模板 JSON（放入 完成標註/ 目錄）
```

### Phase 2：人工標註

**工具**：`gui_annotator.py`（主要）/ `simplification_gui.py`（進階版）

- 載入 TXT 課本檔案，自動解析課文結構
- 左欄顯示原文分析（句長、字數統計）
- 中欄提供三級簡化編輯區
- 右欄即時顯示合規檢查結果（句長是否達標）
- 支援 AI 輔助：一鍵呼叫 LoRA 模型產生初稿
- 批次儲存所有標註至 JSON 檔

### Phase 3：訓練資料轉換

**工具**：`convert_to_training_v3.py`

- 讀取 `完成標註/` 目錄下所有 `*_annotation.json`
- 為每篇課文生成 3 組訓練樣本：
  - **L1 樣本**：原文 → Level 1 簡化版
  - **L2 樣本**：Level 1 → Level 2 簡化版
  - **L3 樣本**：Level 2 → Level 3 簡化版
- 合規驗證：自動檢查句長是否達標，不合規者可選擇跳過
- 過濾章節標題行（如「十一、粒粒皆辛苦——牡蠣養殖」）和書目標籤行（如「書名：大象艾瑪」）
- 輸出 JSONL 訓練資料

```
*_annotation.json → convert_to_training_v3.py → training_data_v5.jsonl
```

### Phase 4：LoRA 微調訓練

**工具**：`peft_train_v3.py`

- 基礎模型：`meta-llama/Llama-3.1-8B-Instruct`
- 4-bit NF4 量化（適配 RTX 4070 12GB 顯存）
- LoRA 配置：r=8, alpha=32, dropout=0.15
- 訓練超參數：3 epochs, batch=1, gradient accumulation=8
- 僅對 assistant 輸出計算 loss（遮蔽 user prompt 部分）
- 輸出 LoRA 權重至 `lora_weights/full_train/final_lora/`

### Phase 5：推論（三種方法）

| 方法 | 檔案 | 說明 |
|:-----|:-----|:-----|
| **LoRA 微調推論** | `simplification_gui.py` | 載入 Base Model + LoRA 權重，按指令生成簡化文本 |
| **Few-Shot 推論** | `few_shot_inference.py` | 3-shot 範例提示，不需 LoRA 權重，作為基線對照 |
| **Base Model 推論** | `evaluate_v2.py` 內建 | 純 Base Model 直接生成，作為另一基線 |

三種方法均使用 chat template 格式，包含 system prompt、instruction 和 input。

### Phase 6：評估比較

**工具**：`evaluate_v2.py`

- 載入訓練資料中的測試樣本
- 分別用三種推論方法生成簡化文本
- 計算三項評估指標（SARI、BERTScore、合規率）
- 輸出評估報告 JSON 和對比圖表 PNG

### Phase 7：品質分析

| 工具 | 功能 |
|:-----|:-----|
| `hsk_coverage.py` | 分析簡化前後的 HSK 詞彙覆蓋率變化，生成對比圖表 |
| `tbcl_analyzer.py` | 使用 TBCL（台灣基礎華語詞彙表）檢測詞彙黃金比例 |
| `readability_tool.py` | 計算原文 vs 簡化版的句長、長詞比例等可讀性指標 |

---

## 5. 核心檔案詳細說明

### 5.1 資料處理模組

#### `core_analyzer.py` — 中文文本句長分析核心引擎

- **功能**：句子切分、字數計算、合規驗證
- **核心函式**：
  - `split_sentences(text)` — 依標點符號（。！？）切分句子
  - `count_chars(sentence)` — 計算有效字符數（排除標點、空格、引號）
  - `analyze_text(text)` — 完整分析文本，回傳句數、平均句長、最大句長及各級合規狀態
  - `validate_simplified(original, simplified, level)` — 驗證簡化版是否符合指定等級標準
  - `create_annotation_template(lesson_name, text)` — 建立標註模板
- **等級標準定義**：
  ```python
  LEVEL_STANDARDS = {
      "level1": {"max_chars": 20, "hsk_range": "1-4", "desc": "閱讀障礙友善基礎版"},
      "level2": {"max_chars": 15, "hsk_range": "1-3", "desc": "進階簡化版"},
      "level3": {"max_chars": 12, "hsk_range": "1-2", "desc": "極簡版"},
  }
  ```

#### `batch_processor.py` — 批次自動化處理模組

- **功能**：自動遍歷目錄內所有 TXT 檔案，解析課文結構並生成標註模板
- **核心函式**：
  - `extract_lessons_from_txt(filepath)` — 從課本 TXT 提取課文段落，支援 UTF-8 / Big5 編碼
  - `process_directory(input_dir, output_dir)` — 批次處理目錄
- **課文識別規則**：使用正則 `r'((?:第?[一二三四五六七八九十百]+[課、]|[一二三四五六七八九十]+[、.．])[^\n]+)'` 辨識課文標題

#### `convert_to_training_v3.py` — 標註資料轉訓練格式

- **功能**：將 JSON 標註檔轉換為 JSONL 訓練資料
- **核心函式**：
  - `check_compliance(text, level)` — 合規驗證，自動過濾章節標題行與書目標籤行
  - `build_training_samples(json_path, include_levels, skip_non_compliant)` — 轉換單一 JSON 為訓練樣本
- **過濾規則**：
  - 標題行：`^[零一二三四五六七八九十百千]+[、．]` （如「十一、粒粒皆辛苦」）
  - 書目標籤行：`^(書名|作者|譯者|出版社|大意|心得感想)：`
- **分級指令模板**：
  - L1：「請將以下課文簡化為適合閱讀障礙學生閱讀的版本，每句不超過20個字，遇到逗號或分號時必須拆成獨立句子，詞彙使用國小4年級程度。」
  - L2：「請將以下文本進一步簡化，每句不超過15個字，使用更基礎的詞彙。」
  - L3：「請將以下文本簡化為最精簡版本，每句不超過12個字，使用最基礎的詞彙。」

### 5.2 GUI 工具

#### `simplification_gui.py` — 主要生產 GUI 應用程式

- **規模**：約 33KB，功能最完整的圖形化介面
- **功能**：
  - 文本簡化介面（含三級編輯區）
  - HSK 詞彙等級即時查詢與標註
  - 整合 LoRA 模型推論（一鍵生成簡化初稿）
  - 繁簡轉換（OpenCC）
  - 語義相似度計算（Sentence-BERT: `paraphrase-multilingual-MiniLM-L12-v2`）
  - 同義詞替換建議
  - 即時合規性檢查
- **模型配置**：
  - Base Model：`meta-llama/Llama-3.1-8B-Instruct`
  - LoRA 權重：`lora_weights/full_train/final_lora`
  - SBERT 模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **HSK 詞庫路徑**：`New HSK (2025)/HSK Words/`（1-7 級）

#### `gui_annotator.py` — 人工標註專用介面

- **規模**：約 21KB
- **UI 框架**：ttkbootstrap（現代化設計風格）
- **功能**：
  - 載入 TXT 課本檔案自動解析
  - 三欄設計：左欄（原文分析）、中欄（三級簡化編輯）、右欄（合規檢查）
  - AI 輔助標註（呼叫 LoRA 模型生成初稿）
  - 即時字數統計與合規驗證
  - 標註結果批次存檔至 JSON
- **依賴**：`core_analyzer.py`（同目錄）

### 5.3 模型訓練

#### `peft_train_v3.py` — LoRA 微調訓練主程式

- **基礎模型**：`meta-llama/Llama-3.1-8B-Instruct`
- **訓練資料**：`完成標註/training_data_v5.jsonl`
- **量化設定（4-bit NF4，適配 RTX 4070 12GB）**：
  ```python
  BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
  )
  ```
- **LoRA 超參數**：
  | 參數 | 值 |
  |:-----|:---|
  | task_type | CAUSAL_LM |
  | r | 8 |
  | lora_alpha | 32 |
  | lora_dropout | 0.15 |
  | target_modules | q_proj, v_proj, k_proj, o_proj |
  | bias | none |
- **訓練超參數**：
  | 參數 | 值 |
  |:-----|:---|
  | epochs | 3 |
  | batch_size | 1 |
  | gradient_accumulation_steps | 8 |
  | learning_rate | 2e-4 |
  | warmup_steps | 5（約佔總 steps 16%）|
  | max_grad_norm | 1.0 |
  | fp16 | True |
  | gradient_checkpointing | True |
  | max_sequence_length | 512 tokens |
- **Prompt 格式**：使用 Llama 3.1 chat template
  ```
  <|begin_of_text|>
  <|start_header_id|>user<|end_header_id|>
  {instruction}\n\n{input}
  <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  {output}
  <|eot_id|>
  ```
- **Loss 遮蔽策略**：只對 assistant 輸出段落計算 loss，user prompt 部分 labels 設為 -100
- **記憶體管理**：GPU 限制 11GiB，CPU offload 30GiB

#### `train_peft.py` — 替代訓練腳本

- 較簡化的 LoRA 訓練實作，功能類似 `peft_train_v3.py` 但配置較少

### 5.4 推論模組

#### `few_shot_inference.py` — Few-Shot 推論基線

- **方法**：3-shot 範例提示（每個等級各 3 組範例）
- **System Prompt**：「你是一個專業的中文語言學家，負責將複雜的中文句子改寫成簡單的初級中文。」
- **各級範例**：
  - L1：原文課文 → 20字以內簡化
  - L2：L1 簡化版 → 15字以內進一步簡化
  - L3：L2 簡化版 → 12字以內極簡化
- **輸出清理**：去除噪音詞（「簡化」「答案」「原句」「規則」等）、去除重複句號、僅保留中文內容
- **各級 max_new_tokens**：L1=300, L2=200, L3=150

### 5.5 評估模組

#### `evaluate_v2.py` — 主要評估框架

- **評估三種方法**：LoRA 微調 vs Base Model vs Few-Shot
- **評估指標**：
  1. **SARI Score**（中文字元級）：衡量新增、保留、刪除操作的 F1
  2. **BERTScore**：語義相似度（使用 `bert-base-chinese` 模型）
  3. **合規率**：符合句長限制的句子比例
- **生成配置**：
  - `repetition_penalty=1.3` — 防止重複生成
  - `no_repeat_ngram_size=4` — 避免 n-gram 重複
  - 各級不同 `max_new_tokens`（L1:300, L2:200, L3:150）
- **輸出**：
  - `eval_report_v2_fixed.json` — 詳細評估報告
  - `eval_comparison_fixed.png` — 對比圖表

### 5.6 品質分析模組

#### `hsk_coverage.py` — HSK 詞彙覆蓋率分析

- **功能**：計算簡化前後的 HSK 詞彙覆蓋率變化
- **HSK 詞庫**：新版 HSK（2025）JSON 格式，1-7 級，支援繁簡體對照
- **分析指標**：
  - HSK 覆蓋率：文本中被 HSK 詞庫收錄的詞比例
  - 低級詞比例：HSK 1-3 級詞彙佔比
- **分詞工具**：jieba
- **輸出**：`hsk_coverage_report_final.json` + 對比圖表

#### `tbcl_analyzer.py` — TBCL 詞彙黃金比例檢測器

- **功能**：使用台灣基礎華語詞彙表（TBCL, 14452 詞）檢測閱讀障礙友善程度
- **GUI**：Tkinter 圖形介面，雙欄對比（原始文本 vs 簡化版本）
- **詞庫來源**：`hsk詞彙/14452詞語表202504.xlsx`
- **對標程度**：國小四年級閱讀障礙學童

#### `readability_tool.py` — 文本難易度對比分析

- **功能**：比較原文與簡化版的可讀性指標
- **分析維度**：
  - 句數與平均句長
  - 有效中文字數
  - 長詞比例（詞長 ≥ 3 字的 jieba 分詞佔比）
- **GUI**：雙欄對比介面

---

## 6. 測試與輔助檔案說明

### 評估與分析（測試性質）

| 檔案 | 說明 |
|:-----|:-----|
| `evaluate_model.py` | 早期版本的模型評估腳本 |
| `evaluate_v1.py` | v1 版評估框架 |
| `evaluate_model v2.py` | v2 版模型評估 |
| `evaluate_few_shot.py` | 專用 Few-Shot 評估腳本 |
| `Ablation Study.py` | 消融實驗：比較不同簡化策略的效果差異 |
| `sari test.py` | SARI 指標測試（中文字元級實作驗證） |
| `LoRA model test.py` | LoRA 權重載入與推論功能驗證 |
| `HSK Vocabulary Coverage.py` | HSK 覆蓋率報告生成 |
| `hsk test.py` | HSK 詞典載入與查詢測試 |

### 品質檢查

| 檔案 | 說明 |
|:-----|:-----|
| `annotation_quality_check.py` | 驗證標註合規率，使用 BERTScore 計算語義保持度 |
| `annotation_stats.py` | 標註統計資訊收集 |
| `quality_checker.py` | 句長違規掃描工具 |

### 資料處理與修復

| 檔案 | 說明 |
|:-----|:-----|
| `convert_annotations_to_training.py` | 多課文 JSON 分割器，智慧辨識課文邊界 |
| `fix_json_clean.py` | JSON 資料清理工具 |
| `fix_vocab.py` | 詞彙表修復工具 |
| `fixjason.py` | JSON 格式修復 |
| `cleaned.py` | 資料清理工具 |
| `verify.py` | 資料驗證工具 |
| `check_words.py` | 詞彙檢查工具 |

### 測試與驗證

| 檔案 | 說明 |
|:-----|:-----|
| `test_fewshot.py` | Few-Shot prompt 策略實驗 |
| `test_peft_config.py` | LoRA 配置驗證 |
| `inference_test.py` | 推論 pipeline 測試 |
| `structure_probe.py` | 模型結構分析（探查 Llama 模型的層數與參數） |

### 其他

| 檔案 | 說明 |
|:-----|:-----|
| `MCTS.py` | 載入 MCTS（Mandarin Chinese Text Simplification）公開資料集進行探索 |
| `base_model_path.py` | 模型路徑配置 |
| `opencc test.py` | OpenCC 繁簡轉換測試 |
| `tempCodeRunnerFile.py` | VS Code 暫存執行檔（可刪除） |
| `simplification_gui copy.py` | GUI 備份檔（可刪除） |

---

## 7. 資料格式規格

### 標註檔格式（JSON）

位於 `完成標註/` 目錄，檔名格式：`{課文名}-{編號}_annotation.json`

```json
{
    "lesson_id": "一、逛夜市-001",
    "source_file": "四上課本.txt",
    "original_text": "夜市裡有各種各樣的小吃攤...",
    "simplified_versions": {
        "level1": {
            "text": "夜市有很多小吃攤。...",
            "compliance": true
        },
        "level2": {
            "text": "夜市有小吃攤。...",
            "compliance": true
        },
        "level3": {
            "text": "夜市有吃的。...",
            "compliance": true
        }
    }
}
```

### 訓練資料格式（JSONL）

檔案：`完成標註/training_data_v5.jsonl`，每行一筆 JSON：

```json
{
    "instruction": "請將以下課文簡化為適合閱讀障礙學生閱讀的版本，每句不超過20個字...",
    "input": "原文內容...",
    "output": "簡化後的版本...",
    "metadata": {
        "lesson_id": "一、逛夜市-001",
        "level": 1,
        "limit": 20,
        "source_file": "四上課本.txt"
    }
}
```

### 評估報告格式（JSON）

檔案：`eval_report_v2_fixed.json`

```json
{
    "model": "LoRA-finetuned",
    "results": [
        {
            "sample_id": 0,
            "level": 1,
            "sari": 0.45,
            "bertscore_f1": 0.87,
            "compliance_rate": 0.95,
            "input": "...",
            "reference": "...",
            "prediction": "..."
        }
    ]
}
```

### HSK 詞庫格式

**新版 JSON 格式**（位於 `New HSK (2025)/HSK Words/new/`）：

```json
[
    {
        "simplified": "你好",
        "forms": [
            {"traditional": "你好"}
        ]
    }
]
```

**舊版 TXT 格式**（位於 `New HSK (2025)/HSK Words/`）：

```
你好    1
謝謝    1
```

---

## 8. 模型訓練配置詳解

### 硬體需求

| 項目 | 規格 |
|:-----|:-----|
| GPU | NVIDIA RTX 4070（12GB VRAM）|
| GPU 記憶體分配 | 11 GiB |
| CPU Offload | 30 GiB |
| 量化方式 | 4-bit NF4 + Double Quantization |

### LoRA 訓練策略

```
原始模型參數量：~8B
LoRA 可訓練參數：
  - target_modules: q_proj, v_proj, k_proj, o_proj (Attention 層)
  - r = 8 (低秩分解維度)
  - lora_alpha = 32 (縮放因子)
  - lora_dropout = 0.15

有效 batch size = per_device_batch_size(1) × gradient_accumulation(8) = 8
```

### 訓練流程

1. 載入 4-bit 量化的 Llama-3.1-8B-Instruct
2. 透過 `prepare_model_for_kbit_training()` 準備量化模型
3. 套用 LoRA 配置（`get_peft_model()`）
4. 格式化所有訓練樣本為 chat template
5. 遮蔽 user prompt 的 labels（設為 -100）
6. 使用 Hugging Face Trainer 進行 3 個 epoch 訓練
7. 每個 epoch 自動儲存 checkpoint
8. 訓練結束儲存最終 LoRA 權重

---

## 9. 評估指標說明

### SARI Score（System output Against References and against the Input sentence）

- **用途**：衡量文本簡化品質的標準指標
- **本專案實作**：中文字元級（非詞級）
- **三個子分數**：
  - **F_add**：模型正確新增的 n-gram（參考答案有、原文沒有的）
  - **F_keep**：模型正確保留的 n-gram（參考答案有、原文也有的）
  - **P_del**：模型正確刪除的 n-gram（參考答案沒有、原文有的）
- **計算方式**：`SARI = (F_add + F_keep + P_del) / 3`
- **n-gram 範圍**：1-gram 到 4-gram

### BERTScore

- **用途**：衡量生成文本與參考答案的語義相似度
- **模型**：`bert-base-chinese`
- **設備**：強制使用 GPU（`SCORE_DEVICE = "cuda"`）
- **batch size**：16

### 句長合規率（Compliance Rate）

- **用途**：衡量簡化文本是否符合各級句長限制
- **計算方式**：`符合限制的句數 / 總句數`
- **句長計算**：排除標點符號後的純中文字元數

---

## 10. 技術堆疊與依賴套件

### 核心框架

| 套件 | 用途 |
|:-----|:-----|
| `transformers` | Hugging Face 模型載入與推論 |
| `peft` | LoRA / Parameter-Efficient Fine-Tuning |
| `bitsandbytes` | 4-bit 量化支援 |
| `datasets` | 訓練資料集管理 |
| `torch` | PyTorch 深度學習框架 |

### 評估工具

| 套件 | 用途 |
|:-----|:-----|
| `bert-score` | BERTScore 語義相似度 |
| `sentence-transformers` | Sentence-BERT 語義向量 |
| `numpy` | 數值計算 |
| `matplotlib` | 圖表生成 |

### 中文處理

| 套件 | 用途 |
|:-----|:-----|
| `jieba` | 中文分詞 |
| `opencc` | 繁簡體轉換（tw2sp / s2twp）|

### GUI

| 套件 | 用途 |
|:-----|:-----|
| `tkinter` | 標準 GUI 框架 |
| `ttkbootstrap` | 現代化 UI 主題（gui_annotator 使用）|

### 資料處理

| 套件 | 用途 |
|:-----|:-----|
| `pandas` | TBCL 詞彙表讀取 |
| `json` | JSON 資料處理 |
| `re` | 正則表達式 |

### 安裝指令

```bash
pip install transformers peft bitsandbytes datasets torch
pip install bert-score sentence-transformers
pip install jieba opencc-python-reimplemented
pip install ttkbootstrap matplotlib numpy pandas
```

---

## 11. 目錄結構

```
F:\download\專題\
│
├── 📂 核心程式
│   ├── core_analyzer.py              # 句長分析引擎
│   ├── batch_processor.py            # 批次處理模組
│   ├── convert_to_training_v3.py     # 訓練資料轉換
│   ├── simplification_gui.py         # 主要 GUI（含推論）
│   ├── gui_annotator.py              # 標註介面
│   ├── peft_train_v3.py              # LoRA 微調訓練
│   ├── train_peft.py                 # 替代訓練腳本
│   ├── few_shot_inference.py         # Few-Shot 推論
│   ├── evaluate_v2.py                # 主要評估框架
│   ├── hsk_coverage.py               # HSK 覆蓋率分析
│   ├── tbcl_analyzer.py              # TBCL 黃金比例分析
│   └── readability_tool.py           # 可讀性分析
│
├── 📂 測試與輔助
│   ├── Ablation Study.py             # 消融實驗
│   ├── evaluate_model.py             # 早期評估版本
│   ├── evaluate_v1.py                # v1 評估
│   ├── evaluate_model v2.py          # v2 模型評估
│   ├── evaluate_few_shot.py          # Few-Shot 評估
│   ├── sari test.py                  # SARI 測試
│   ├── LoRA model test.py            # LoRA 測試
│   ├── test_fewshot.py               # Few-Shot 策略測試
│   ├── test_peft_config.py           # LoRA 配置測試
│   ├── inference_test.py             # 推論測試
│   ├── structure_probe.py            # 模型結構探查
│   ├── annotation_quality_check.py   # 標註品質檢查
│   ├── annotation_stats.py           # 標註統計
│   ├── quality_checker.py            # 品質掃描
│   ├── HSK Vocabulary Coverage.py    # HSK 覆蓋率報告
│   ├── hsk test.py                   # HSK 測試
│   ├── opencc test.py                # 繁簡轉換測試
│   ├── MCTS.py                       # MCTS 資料集探索
│   ├── base_model_path.py            # 模型路徑配置
│   ├── check_words.py                # 詞彙檢查
│   ├── verify.py                     # 資料驗證
│   ├── fix_json_clean.py             # JSON 清理
│   ├── fix_vocab.py                  # 詞彙修復
│   ├── fixjason.py                   # JSON 修復
│   └── cleaned.py                    # 資料清理
│
├── 📂 資料目錄
│   ├── 完成標註/                      # 標註完成的 JSON 檔案
│   │   ├── *_annotation.json         # 各課文標註
│   │   └── training_data_v5.jsonl    # 最終訓練資料
│   ├── New HSK (2025)/               # HSK 詞庫
│   │   └── HSK Words/
│   │       ├── HSK_Level_1_words.txt # 舊版 TXT 格式
│   │       └── new/                  # 新版 JSON 格式
│   │           ├── 1.json ~ 7.json
│   ├── hsk詞彙/                      # TBCL 詞彙表
│   ├── txt_output/                   # 課本 TXT 原始檔
│   └── 測試文檔/                      # 測試用文件
│
├── 📂 模型權重
│   ├── lora_weights/                 # LoRA 訓練輸出
│   │   └── full_train/
│   │       └── final_lora/           # 最終 LoRA 權重
│   ├── peft_checkpoints/             # 訓練 checkpoint
│   ├── peft_results/                 # 訓練結果
│   └── peft_logs/                    # 訓練日誌
│
├── 📂 評估輸出
│   ├── eval_report_v2_fixed.json     # 評估報告
│   ├── eval_comparison_fixed.png     # 對比圖表
│   ├── hsk_coverage_report_final.json # HSK 覆蓋率報告
│   ├── hsk_before_after_chart.png    # HSK 前後對比圖
│   ├── ablation_results.json         # 消融實驗結果
│   └── few_shot_results.jsonl        # Few-Shot 結果
│
├── 📂 參考資料
│   ├── mcts-main/                    # MCTS 公開資料集
│   │   ├── dataset/                  # 測試集
│   │   ├── pseudo_data/              # 69 萬筆偽資料
│   │   └── scripts/                  # MCTS 原始腳本
│   └── 論文圖片/                      # 論文用圖片素材
│
├── 完整版新聞HSK三級簡化與詞彙標註研究報告.pdf  # 研究報告
└── PROJECT_DOCUMENTATION.md          # 本文件
```

---

> 本文件基於專案原始碼自動分析產生，涵蓋所有 Python 檔案的功能說明、資料格式規格、模型訓練配置及評估方法。
