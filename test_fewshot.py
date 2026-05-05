from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("=" * 60)
print("Few-shot 改進版測試 v3.0")
print("設計策略：高品質範例 + 策略性選擇 + 強化約束")
print("=" * 60)

# 載入模型
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prompt 設計
prompt_v3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是中文文本簡化專家。執行以下規則：
1. 輸出句長必須在13-16字之間
2. 使用HSK3-4詞彙
3. 保留核心意義

重要：超過16字或使用HSK5-6詞彙將被視為任務失敗。<|eot_id|>

<|start_header_id|>user<|end_header_id|>
參考以下範例（每個範例嚴格符合13-16字且使用HSK3-4詞彙）：

範例1（抽象概念簡化）：
原文：該國在區域安全架構中扮演核心角色。
簡化：這個國家在地區安全很重要。
[句長14字✓ HSK3-4✓]

範例2（複合術語處理）：
原文：全球半導體供應鏈面臨重組壓力。
簡化：全球半導體供應網要重新安排。
[句長14字✓ HSK3-4✓]

範例3（統計表達降級）：
原文：市場占有率達到百分之六十五點三。
簡化：市場占比超過六成五。
[句長10字✓ HSK3✓ 注：因原文簡潔故輸出較短]

範例4（綜合應用）：
原文：印度政府對俄羅斯進口鑽石課徵百分之五關稅。
簡化：印度對俄國鑽石收百分之五的稅。
[句長15字✓ HSK3-4✓]

現在請簡化以下句子，確保輸出在13-16字之間且使用HSK3-4詞彙：
台灣半導體產業在全球供應鏈扮演關鍵角色,晶圓代工市占率超過六成。<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

prompt_v2_baseline = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是中文閱讀障礙文本簡化專家。根據HSK等級簡化新聞。
嚴格規則：
1. 句長必須13-16字，超過則拆分
2. 僅使用HSK3-4詞彙，禁用HSK5-6詞彙
3. 保留核心意義，刪除修飾語<|eot_id|>

<|start_header_id|>user<|end_header_id|>
參考以下簡化範例（已達標）：

範例1：
原文：以色列國防軍空襲加薩走廊北部賈巴利亞難民營。
簡化：以色列軍隊用飛機打加薩北部難民營。

範例2：
原文：印度政府對俄羅斯進口鑽石課徵百分之五關稅。
簡化：印度對俄國鑽石收5%的稅。

範例3：
原文：聯合國秘書長呼籲各國立即停止軍事行動。
簡化：聯合國叫各國馬上停止打仗。

現在請簡化（必須13-16字/HSK3-4）：
台灣半導體產業在全球供應鏈扮演關鍵角色,晶圓代工市占率超過六成。<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

def generate_and_extract(prompt, test_name):
    """生成並擷取 assistant 回應"""
    print(f"\n{'=' * 60}")
    print(f"執行：{test_name}")
    print(f"{'=' * 60}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result_with_tags = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|start_header_id|>assistant<|end_header_id|>" in result_with_tags:
        simplified = result_with_tags.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        simplified = simplified.split("<|eot_id|>")[0].split("<|end_of_text|>")[0].strip()
    else:
        result_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
        simplified = result_clean[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
    
    if simplified.startswith("簡化:") or simplified.startswith("簡化："):
        simplified = simplified.split(":", 1)[-1].split("：", 1)[-1].strip()
    if "[" in simplified:
        simplified = simplified.split("[")[0].strip()
    
    return simplified

def analyze_quality(text, test_name):
    """品質分析"""
    print(f"\n【{test_name} 輸出】")
    print(f"生成內容：{text}")
    print(f"\n品質分析：")
    
    meta_keywords = ["我會", "幫你", "以下是", "這是", "讓我", "根據"]
    is_meta = any(kw in text for kw in meta_keywords)
    
    if is_meta:
        print("✗ 任務理解：失敗（meta-response）")
        return {"task_ok": False, "length_ok": False, "vocab_ok": False, "score": 0}
    else:
        print("✓ 任務理解：成功")
    
    length = len(text.replace(" ","").replace("，","").replace("。","")
                  .replace(",","").replace(".","").replace("、",""))
    length_ok = 13 <= length <= 16
    
    if length_ok:
        print(f"✓ 句長：{length}字（目標13-16字）達標")
    else:
        deviation = length - 16 if length > 16 else 13 - length
        print(f"✗ 句長：{length}字（{'超出' if length > 16 else '不足'} {deviation} 字）")
    
    hsk5_6_vocab = {
        "扮演": "做/是",
        "關鍵": "重要/主要", 
        "角色": "位置/作用",
        "市占率": "占比/比例",
        "供應鏈": "供應網/供應",
        "產業": "行業/業"
    }
    
    found_high_level = {word: hsk5_6_vocab[word] for word in hsk5_6_vocab if word in text}
    vocab_ok = len(found_high_level) == 0
    
    if vocab_ok:
        print("✓ 詞彙等級：符合HSK3-4")
    else:
        print("✗ 詞彙等級：包含高階詞彙")
        for word, suggestion in found_high_level.items():
            print(f"  - {word} → 建議：{suggestion}")
    
    score = sum([not is_meta, length_ok, vocab_ok])
    
    return {
        "task_ok": not is_meta,
        "length_ok": length_ok,
        "vocab_ok": vocab_ok,
        "length": length,
        "score": score
    }

# 執行測試
print("\n開始對照測試...")

result_v2 = generate_and_extract(prompt_v2_baseline, "v2 基準版本（舊設計）")
analysis_v2 = analyze_quality(result_v2, "v2 基準版本")

result_v3 = generate_and_extract(prompt_v3, "v3 改進版本（新設計）")
analysis_v3 = analyze_quality(result_v3, "v3 改進版本")

# 對比分析
print("\n" + "=" * 60)
print("對比分析")
print("=" * 60)

print(f"\n{'指標':<15} {'v2 基準':<12} {'v3 改進':<12} {'改進效果':<12}")
print("-" * 60)

task_v2 = '✓' if analysis_v2['task_ok'] else '✗'
task_v3 = '✓' if analysis_v3['task_ok'] else '✗'
task_trend = '→' if analysis_v3['task_ok'] == analysis_v2['task_ok'] else ('↑' if analysis_v3['task_ok'] else '↓')
print(f"{'任務理解':<15} {task_v2:<12} {task_v3:<12} {task_trend:<12}")

length_v2 = '✓' if analysis_v2['length_ok'] else '✗'
length_v3 = '✓' if analysis_v3['length_ok'] else '✗'
length_trend = '→' if analysis_v3['length_ok'] == analysis_v2['length_ok'] else ('↑' if analysis_v3['length_ok'] else '↓')
print(f"{'句長達標':<15} {length_v2:<12} {length_v3:<12} {length_trend:<12}")

vocab_v2 = '✓' if analysis_v2['vocab_ok'] else '✗'
vocab_v3 = '✓' if analysis_v3['vocab_ok'] else '✗'
vocab_trend = '→' if analysis_v3['vocab_ok'] == analysis_v2['vocab_ok'] else ('↑' if analysis_v3['vocab_ok'] else '↓')
print(f"{'詞彙達標':<15} {vocab_v2:<12} {vocab_v3:<12} {vocab_trend:<12}")

score_diff = analysis_v3['score'] - analysis_v2['score']
diff_display = f"+{score_diff}" if score_diff > 0 else str(score_diff)
print(f"{'總分':<15} {analysis_v2['score']}/3{'':<6} {analysis_v3['score']}/3{'':<6} {diff_display:<12}")

# 決策建議
print("\n" + "=" * 60)
print("決策建議")
print("=" * 60)

improvement = analysis_v3['score'] - analysis_v2['score']

if analysis_v3['score'] == 3:
    print("\n【結論】v3 改進版本完全達標（3/3）")
    print("→ Few-shot 方法可行，進入詞典建構階段")
    print("→ 後續行動：")
    print("  1. 擴充測試至5個不同句子驗證穩定性")
    print("  2. 建構HSK詞典（參考v2.0報告詞彙對映表）")
    print("  3. 整合Few-shot + 詞典後處理的混合架構")
    decision = "Few-shot 可用 → 詞典建構"

elif analysis_v3['score'] >= 2 and improvement > 0:
    print(f"\n【結論】v3 改進版本部分達標（{analysis_v3['score']}/3）且優於v2")
    print("→ Few-shot 有改進空間，建議繼續優化")
    print("→ 後續行動：")
    print("  1. 分析未達標項目的根因（句長 vs 詞彙）")
    print("  2. 若僅句長未達標 → 增加句長後處理模組")
    print("  3. 若僅詞彙未達標 → 採用混合架構（Few-shot + 詞典）")
    print("  4. 若兩者均未達標 → 轉向PEFT微調")
    decision = "繼續優化 Few-shot"

elif improvement == 0:
    print(f"\n【結論】v3 改進無效（{analysis_v3['score']}/3，與v2相同）")
    print("→ Prompt 設計改進未產生效果")
    print("→ 建議進入 PEFT 微調路線")
    print("→ 原因：模型對範例品質提升不敏感，需要參數級適應")
    decision = "進入 PEFT 微調"

else:
    print(f"\n【結論】v3 改進反向劣化（{analysis_v3['score']}/3 < v2的{analysis_v2['score']}/3）")
    print("→ 新範例設計引入負面影響")
    print("→ 建議：")
    print("  1. 診斷v3範例的問題（可能是抽象概念範例過於複雜）")
    print("  2. 若診斷無果 → 直接進入PEFT微調")
    decision = "診斷範例問題或進入PEFT"

print(f"\n最終決策：{decision}")

# 詳細輸出對照
print("\n" + "=" * 60)
print("詳細輸出對照")
print("=" * 60)

print(f"\n原文：")
print("台灣半導體產業在全球供應鏈扮演關鍵角色,晶圓代工市占率超過六成。")

print(f"\nv2 基準版本輸出：")
print(result_v2)

print(f"\nv3 改進版本輸出：")
print(result_v3)

print(f"\n理想參考（v2.0報告）：")
print("台灣半導體在全球很重要,晶圓代工占六成以上。")
print("[句長16字 HSK3-4詞彙]")

print("\n" + "=" * 60)
print("測試完成")
print("=" * 60)