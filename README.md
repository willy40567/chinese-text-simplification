# 中文閱讀障礙課文三級簡化系統

> 針對台灣國小四年級課文的自動簡化系統，採用 **Llama-3.1-8B-Instruct + LoRA** 微調，以三級漸進式策略產生適合**閱讀障礙學童**的課文版本。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![Model](https://img.shields.io/badge/base--model-Llama--3.1--8B-green.svg)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange.svg)](https://github.com/huggingface/peft)

---

## 研究背景

閱讀障礙學童在標準課文閱讀時遭遇句長過長、詞彙抽象的困難。本專題以 LoRA 微調大型語言模型，將原文逐級簡化為三個難度版本，並透過 SARI、BERTScore、句長合規率三項指標進行品質驗證。

| 等級 | 句長上限 | 詞彙範圍 | 用途 |
|:---:|:---:|:---:|:---|
| **L1** | ≤ 20 字 | HSK 1-4 | 閱讀障礙友善基礎版 |
| **L2** | ≤ 15 字 | HSK 1-3 | 進階簡化版 |
| **L3** | ≤ 12 字 | HSK 1-2 | 極簡版（主詞 + 動作 + 結果） |

逐級遞進：原文 → L1 → L2 → L3，每一級以上一級輸出為輸入。

---

## 最終評估結果

訓練資料：80 筆（L1: 26 / L2: 27 / L3: 27），來源為國小四年級課本與 PRIORI 補救教材。

| 等級 | SARI | F_add | F_keep | P_del | BERTScore | 句長合規率 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L1** | **0.4299** | 0.1143 | 0.5018 | 0.6737 | 0.8096 | 81.96% |
| **L2** | 0.3549 | 0.0634 | 0.5768 | 0.4243 | **0.8645** | **96.03%** |
| **L3** | 0.4360 | 0.1339 | 0.6324 | 0.5417 | **0.8768** | 95.06% |

聚合報告：[eval_report_v5_summary.json](eval_report_v5_summary.json)

**消融實驗**（[ablation_results.json](ablation_results.json)）

| 組別 | 設計 | 結果 |
|:---|:---|:---|
| A | 純 LoRA | 句長合規率優異 |
| B | 純 HSK 字典規則替換 | 詞彙等級下降但破壞語法 |
| C | LoRA + HSK 後處理 | 兩者結合的完整系統 |

---

## 系統架構

```
TXT 課本
   │
   ▼  batch_processor.py
標註模板 JSON
   │
   ▼  gui_annotator.py（人工 + AI 輔助）
完成標註/*_annotation.json
   │
   ▼  convert_to_training_v3.py（合規檢查 + 章節/書目過濾）
training_data_v5.jsonl
   │
   ▼  peft_train_v3.py（LoRA, 4-bit NF4）
lora_weights/full_train/final_lora/
   │
   ├──▶ evaluate_v1.py     → eval_report_v5.json（正式評估）
   ├──▶ few_shot_inference.py（基線對照）
   └──▶ Ablation Study.py（純 LoRA / 純 HSK 字典 / LoRA+HSK）
```

---

## 技術選型

| 項目 | 選用 |
|:---|:---|
| 基礎模型 | `meta-llama/Llama-3.1-8B-Instruct` |
| 微調方法 | LoRA (PEFT 0.7.1) |
| 量化 | 4-bit NF4 + Double Quantization (bitsandbytes) |
| 硬體 | RTX 4070 12GB |
| 評估指標 | SARI（中文字元級）、BERTScore (`bert-base-chinese`)、句長合規率 |
| 中文處理 | jieba（分詞）、OpenCC（繁簡 tw2sp / s2twp） |
| GUI | tkinter + ttkbootstrap |

### LoRA 設定

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05
learning_rate = 2e-4
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
num_train_epochs = 3
per_device_train_batch_size = 1
gradient_accumulation_steps = 8   # 有效 batch size = 8
```

### Prompt 格式（Llama 3.1 chat template）

```
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{instruction}\n\n{input}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}
<|eot_id|>
```

訓練時對 user 段落（至 `<|eot_id|>` 為止）的 labels 設為 `-100`，僅對 assistant 輸出計算 loss。

---

## 簡化結果範例

以下範例為**自編歷史故事素材**（非教科書內容），展示模型實際輸出：

| 原文 | L1 (≤20 字) | L2 (≤15 字) | L3 (≤12 字) |
|:---|:---|:---|:---|
| 蔡倫不斷的觀察和思考，終於用樹皮、破布、廢漁網和麻布等東西，經過切斷、搗爛、攪拌、晾乾等步驟，成功的製造出既輕便又便宜的紙張。 | 蔡倫不斷觀察和思考。最後他用樹皮、破布、廢漁網和麻布製造出紙張。 | 蔡倫不斷觀察和思考。他用樹皮製造紙張。 | 蔡倫不斷思考研究。他用樹皮造紙。 |
| 戰爭過後，南丁格爾成立了世界上第一所護士學校，訓練了成千上萬的護理人員，也改變了大家對護士工作的印象。| 戰爭結束後南丁格爾建立護士學校。她訓練很多護士。 | 南丁格爾建立護士學校。她訓練了很多護士。 | 南丁格爾建護校。 |
| 鄭成功率領軍隊驅逐荷蘭人，收復台灣。 | 鄭成功帶兵打走荷蘭人。他收回了台灣。 | 鄭成功打走了敵人。他收回台灣。 | 他收回台灣。 |

---

## 目錄結構

```
專題/
├── config.py                      # 統一路徑與常數
├── core_analyzer.py               # 句長分析引擎
├── hsk_utils.py                   # HSK 字典、同義詞替換、合規率
├── model_utils.py                 # 模型載入、4-bit 量化設定
│
├── batch_processor.py             # TXT → 標註模板
├── gui_annotator.py               # 標註 GUI（ttkbootstrap）
├── simplification_gui.py          # 主推論 GUI
├── convert_to_training_v3.py      # 標註 JSON → 訓練 JSONL
│
├── peft_train_v3.py               # LoRA 微調訓練
├── evaluate_v1.py                 # 正式評估腳本
├── few_shot_inference.py          # Few-Shot 基線
├── Ablation Study.py              # 消融實驗
│
├── hsk_coverage.py                # HSK 覆蓋率分析
├── tbcl_analyzer.py               # TBCL 詞彙黃金比例
├── readability_tool.py            # 可讀性對比
│
├── New HSK (2025)/HSK Words/
│   ├── new/{1-7}.json             # 新版 HSK 詞庫（含繁體）
│   └── HSK_Level_*.txt            # 舊版 TXT
│
├── lora_weights/full_train/final_lora/
│   ├── adapter_config.json
│   └── adapter_model.safetensors  # 27MB，最終 LoRA 權重
│
├── eval_report_v5_summary.json    # 評估彙總（無原文）
├── ablation_results.json          # 消融實驗結果
└── hsk_coverage_report_final.json # HSK 覆蓋率分析
```

---

## 資料公開性說明（Data Availability）

本專案的訓練資料來源為**台灣國小四年級教科書**與**PRIORI 補救教材**，因教科書內容受版權保護，**完整訓練資料（`完成標註/` 與 `training_data_v5.jsonl`）不公開**。

公開內容包含：
- ✅ 全部程式碼（資料前處理、訓練、評估、推論、GUI）
- ✅ 訓練後的 LoRA 權重（27MB，可直接用於推論）
- ✅ HSK 詞庫（公開資源）
- ✅ 評估指標彙總（去除原文 source 欄位）
- ✅ 消融實驗結果（使用自編歷史故事素材）

如需獲取完整訓練資料以重現研究，請聯繫作者並提供合理的學術用途說明。

### 訓練資料 JSONL Schema

```json
{
  "instruction": "請將以下課文簡化為適合閱讀障礙學生閱讀的版本，每句不超過20個字...",
  "input": "原文內容...",
  "output": "簡化後的版本...",
  "metadata": {
    "lesson_id": "課文識別碼",
    "level": 1,
    "limit": 20,
    "source_file": "來源檔案"
  }
}
```

---

## 快速開始

### 環境需求

- Python 3.9+
- CUDA-capable GPU（≥ 8GB VRAM；4-bit 量化下可降至 ~6GB）
- Hugging Face token（`huggingface-cli login`，需取得 Llama 3.1 存取權）

### 安裝

```bash
pip install -r requirements.txt
huggingface-cli login   # 輸入你的 HF token，需先在 HF 網站申請 Llama 3.1 存取權
```

### 推論（GUI）

```bash
python simplification_gui.py
```

### 推論（命令列）

```python
from model_utils import load_finetuned_model
from config import LEVEL_INSTRUCTION

model, tokenizer = load_finetuned_model()

text = "蔡倫不斷的觀察和思考，終於用樹皮、破布、廢漁網等材料製造出紙張。"
prompt = (
    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    f"{LEVEL_INSTRUCTION[1]}\n\n{text}"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 重現訓練（需自備標註資料）

```bash
# 1. 將標註 JSON 轉為 JSONL
python convert_to_training_v3.py

# 2. LoRA 微調
python peft_train_v3.py

# 3. 評估
python evaluate_v1.py
```

---

## 設計決策與已知陷阱

- **`repetition_penalty=1.3` 禁用**：實驗顯示會使 BERTScore 從 0.82 崩潰至 0.62，僅可使用 `1.1`。
- **HSK 為簡體標準**：對繁體文本有系統性偏差，HSK 覆蓋率僅作為輔助指標。
- **Few-Shot L1 高合規率（99.7%）為過度簡化造成**，BERTScore 僅 0.35，非公平基線。
- **小樣本工程妥協**：80 筆資料、L1 僅 26 筆，採用 2-fold 交叉驗證而非標準 3-fold。
- **章節標題/書目標籤行不計入句長**：合規檢查時自動過濾 `一、xxx`、`書名：xxx` 等行。

---

## 參考資料

- [MCTS: Mandarin Chinese Text Simplification Dataset](https://github.com/blcuicall/mcts) — 公開中文簡化資料集（本專案僅作對照，未用於訓練）
- [新版 HSK (2025) 詞表](https://github.com/krmanik/HSK-3.0) — HSK 1-7 級官方詞彙
- TBCL 台灣基礎華語詞彙表（14,452 詞）

---

## 作者

[**willy40567**](https://github.com/willy40567)

本專題為大學畢業專題作品，研究方向為**閱讀障礙友善的中文文本自動簡化**。

---

## 授權

本專案程式碼採用 [MIT License](LICENSE)。

LoRA 權重基於 `meta-llama/Llama-3.1-8B-Instruct`，使用上應遵循 [Meta 的 Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)。
