# evaluate_v1.py
# 放置於 F:\download\專題\
# 安裝依賴：pip install bert-score sacrebleu --break-system-packages

import json
import re
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from bert_score import score as bert_score

# ============================================================
# 0. 設定
# ============================================================
BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH   = r"F:\download\專題\lora_weights\full_train\final_lora"
OFFLOAD_DIR = r"F:\download\專題\offload_tmp"
DATA_PATH   = r"F:\download\專題\完成標註\training_data_v5.jsonl"
OUTPUT_PATH = r"F:\download\專題\eval_report_v5.json" 

# ============================================================
# 1. SARI 實作（中文字元級）
# ============================================================
def tokenize_zh(text):
    """中文以字元為單位分詞"""
    return list(re.sub(r'\s+', '', text))

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def sari_score(source, prediction, reference):
    """
    計算單筆 SARI（字元級，n=1,2,3,4 平均）
    SARI = (F_add + F_keep + P_del) / 3
    """
    def f1(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    src_tok  = tokenize_zh(source)
    pred_tok = tokenize_zh(prediction)
    ref_tok  = tokenize_zh(reference)

    add_scores, keep_scores, del_scores = [], [], []

    for n in range(1, 5):
        src_ng  = set(ngrams(src_tok,  n))
        pred_ng = set(ngrams(pred_tok, n))
        ref_ng  = set(ngrams(ref_tok,  n))

        # ADD：pred 中有、src 中沒有的 n-gram，與 ref 的重疊
        added_pred = pred_ng - src_ng
        added_ref  = ref_ng  - src_ng
        if added_ref:
            p = len(added_pred & added_ref) / len(added_pred) if added_pred else 0
            r = len(added_pred & added_ref) / len(added_ref)
            add_scores.append(f1(p, r))
        else:
            add_scores.append(1.0 if not added_pred else 0.0)

        # KEEP：src 中有、ref 中也有的 n-gram，pred 是否保留
        keep_ref  = src_ng & ref_ng
        if keep_ref:
            p = len(pred_ng & keep_ref) / len(pred_ng & src_ng) if (pred_ng & src_ng) else 0
            r = len(pred_ng & keep_ref) / len(keep_ref)
            keep_scores.append(f1(p, r))
        else:
            keep_scores.append(1.0)

        # DELETE：src 中有、ref 中沒有的 n-gram，pred 是否刪除
        del_src = src_ng - ref_ng
        if del_src:
            p = len(del_src - pred_ng) / len(del_src)
            del_scores.append(p)
        else:
            del_scores.append(1.0)

    F_add  = np.mean(add_scores)
    F_keep = np.mean(keep_scores)
    P_del  = np.mean(del_scores)

    return (F_add + F_keep + P_del) / 3, F_add, F_keep, P_del

# ============================================================
# 2. 句長合規率
# ============================================================
def compliance_rate(text, limit):
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    compliant = sum(1 for s in sentences if len(re.sub(r'[^\u4e00-\u9fff\w]', '', s)) <= limit)
    return compliant / len(sentences)

# ============================================================
# 3. 載入模型
# ============================================================
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        max_memory          = {0: "11GiB", "cpu": "30GiB"},
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, instruction, input_text):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}\n\n{input_text}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=400,
            temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in full:
        return full.split(marker)[-1].replace("<|eot_id|>","").strip()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================
# 4. 主評估流程
# ============================================================
def main():
    # 載入測試資料（使用全部 30 筆）
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    model, tokenizer = load_model()

    LIMIT_MAP = {1: 20, 2: 15, 3: 12}
    results = {1: [], 2: [], 3: []}

    print(f"開始評估 {len(samples)} 筆樣本...")

    for i, sample in enumerate(samples):
        level = sample["metadata"]["level"]
        instruction = sample["instruction"]
        source  = sample["input"]
        reference = sample["output"]

        prediction = generate(model, tokenizer, instruction, source)

        # SARI
        sari, f_add, f_keep, p_del = sari_score(source, prediction, reference)

        # 句長合規率
        comp = compliance_rate(prediction, LIMIT_MAP[level])

        results[level].append({
            "id"          : i,
            "source"      : source[:50] + "...",
            "reference"   : reference[:50] + "...",
            "prediction"  : prediction[:50] + "...",
            "sari"        : round(sari,  4),
            "f_add"       : round(f_add, 4),
            "f_keep"      : round(f_keep,4),
            "p_del"       : round(p_del, 4),
            "compliance"  : round(comp,  4),
        })

        print(f"[{i+1:02d}/{len(samples)}] L{level} SARI={sari:.3f} Comp={comp:.1%}")

    # BERTScore（全部樣本一次計算）
    all_preds = []
    all_refs  = []
    for lv in [1, 2, 3]:
        for r in results[lv]:
            all_preds.append(r["prediction"])
            all_refs.append(r["reference"])

    print("\n計算 BERTScore（中文 BERT）...")
    P, R, F1 = bert_score(
        all_preds, all_refs,
        lang="zh", model_type="bert-base-chinese",
        verbose=True
    )

    # 彙整報告
    report = {}
    idx = 0
    for lv in [1, 2, 3]:
        n = len(results[lv])
        f1_slice = F1[idx:idx+n].tolist()
        idx += n
        for j, r in enumerate(results[lv]):
            r["bertscore_f1"] = round(f1_slice[j], 4)

        report[f"level{lv}"] = {
            "sari_mean"        : round(np.mean([r["sari"]         for r in results[lv]]), 4),
            "compliance_mean"  : round(np.mean([r["compliance"]   for r in results[lv]]), 4),
            "bertscore_mean"   : round(np.mean(f1_slice), 4),
            "details"          : results[lv]
        }

    # 輸出
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"評估報告摘要")
    print(f"{'='*60}")
    for lv in [1, 2, 3]:
        d = report[f"level{lv}"]
        print(f"  Level {lv}｜SARI={d['sari_mean']:.4f}｜"
              f"BERTScore={d['bertscore_mean']:.4f}｜"
              f"句長合規率={d['compliance_mean']:.1%}")
    print(f"{'='*60}")
    print(f"完整報告 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()