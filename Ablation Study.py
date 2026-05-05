import json
import os
import torch
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 從共用模組匯入
# ==========================================
from config import LORA_WEIGHTS_DIR, ABLATION_RESULTS
from hsk_utils import (
    build_hsk_vocab_txt, load_hsk_dict,
    hsk_replace, register_proper_nouns_jieba,
    compliance_rate,
)
from model_utils import load_finetuned_model

# ==========================================
# 1. 初始化 HSK 字典
# ==========================================
build_hsk_vocab_txt()
my_hsk_dict = load_hsk_dict()
register_proper_nouns_jieba()
print(f"✅ HSK 字典載入，共 {len(my_hsk_dict)} 詞\n")

# ==========================================
# 2. 載入 LoRA 模型
# ==========================================
print("載入模型...")
model, tokenizer = load_finetuned_model(lora_path=LORA_WEIGHTS_DIR)
print("✅ 模型載入完成\n")

def lora_generate(sent, level):
    from config import LEVEL_LIMITS
    limit = LEVEL_LIMITS.get(level, 999)
    prompt = f"""你是一個中文文本簡化助手。
規則：
1. 每句不超過 {limit} 字（不含標點）
2. 若原句太長，切成多個短句
3. 不可加入原句沒有的資訊
4. 只輸出簡化結果，不要說明

原句：{sent}
簡化（每句≤{limit}字）："""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=80, temperature=0.3,
            top_p=0.9, repetition_penalty=1.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    if "。" in result:
        result = result.split("。")[0] + "。"
    return result

# ==========================================
# 3. 讀取測試資料（eval_results_v2.json）
# ==========================================
eval_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results_v2.json")
if os.path.exists(eval_file_path):
    with open(eval_file_path, encoding="utf-8") as f:
        eval_data = json.load(f)
else:
    print(f"⚠️ 找不到測試資料檔案：{eval_file_path}")
    eval_data = {"finetuned": []}

# ==========================================
# 4. 三組消融實驗
# ==========================================
results = {"group_A": [], "group_B": [], "group_C": []}

print("=" * 60)
print("消融實驗開始")
print("=" * 60)

for item in eval_data.get("finetuned", []):
    sample_id = item["id"]
    level     = item["level"]
    original  = item["input"]
    reference = item["reference"]

    print(f"\n▶ 樣本：{sample_id}（L{level}）")

    # --- A 組：純 LoRA（直接取 eval 中已有結果）---
    output_A = item["output"]
    cr_A = compliance_rate(output_A, level)
    print(f"  [A 純LoRA]    {output_A}（合規:{cr_A:.0%}）")

    # --- B 組：純 HSK 管線（原文直接過管線，不經模型）---
    output_B = hsk_replace(original, my_hsk_dict, level)
    cr_B = compliance_rate(output_B, level)
    print(f"  [B 純字典]    {output_B}（合規:{cr_B:.0%}）")

    # --- C 組：LoRA + HSK 後處理（完整系統）---
    output_C = hsk_replace(output_A, my_hsk_dict, level)
    cr_C = compliance_rate(output_C, level)
    print(f"  [C 完整系統]  {output_C}（合規:{cr_C:.0%}）")

    for group, output, cr in [("group_A", output_A, cr_A),
                               ("group_B", output_B, cr_B),
                               ("group_C", output_C, cr_C)]:
        results[group].append({
            "id": sample_id, "level": level,
            "input": original, "output": output,
            "compliance_rate": cr, "reference": reference
        })

# ==========================================
# 5. 輸出彙總表
# ==========================================
print("\n" + "=" * 60)
print("消融實驗彙總")
print("=" * 60)

for group, label in [("group_A","A 純LoRA"), ("group_B","B 純字典"), ("group_C","C 完整系統")]:
    if len(results[group]) > 0:
        avg_cr = sum(r["compliance_rate"] for r in results[group]) / len(results[group])
    else:
        avg_cr = 0
    print(f"  {label}: 平均合規率 {avg_cr:.1%}")

# 儲存結果
try:
    with open(ABLATION_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 消融實驗完成，結果已儲存至 {ABLATION_RESULTS}")
except Exception as e:
    print(f"\n❌ 儲存結果失敗: {e}")
