import os
import torch
import jieba
import opencc
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 1. HSK 字典
# ==========================================
folder_path = "F:/download/專題/New HSK (2025)/HSK Words"
output_file = "hsk_vocab.txt"

level_mapping = {
    "HSK_Level_1_words.txt": 1, "HSK_Level_2_words.txt": 2,
    "HSK_Level_3_words.txt": 3, "HSK_Level_4_words.txt": 4,
    "HSK_Level_5_words.txt": 5, "HSK_Level_6_words.txt": 6,
    "HSK_Level_7-9_words.txt": 7
}

with open(output_file, "w", encoding="utf-8") as out_f:
    for filename, level in level_mapping.items():
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    parts = line.strip().split()
                    if parts:
                        out_f.write(f"{parts[0]}\t{level}\n")

def load_hsk_dict(path="hsk_vocab.txt"):
    hsk = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                hsk[parts[0]] = int(parts[1])
    return hsk

my_hsk_dict = load_hsk_dict(output_file)
print(f"✅ HSK 字典載入完成，共 {len(my_hsk_dict)} 個詞彙\n")

# ==========================================
# 2. 初始化設定
# ==========================================
tw2sp = opencc.OpenCC('tw2sp')
sp2tw = opencc.OpenCC('s2twp')

PROPER_NOUNS_SIMP = {"南丁格尔", "蔡伦", "郑成功", "安平古堡", "台湾", "中国"}

SUPPLEMENT_DICT = {
    "第一": 2, "所": 2, "和": 1, "的": 1, "了": 1,
    "在": 1, "是": 1, "有": 1, "不": 1, "也": 1,
    "都": 2, "很": 1, "就": 2, "把": 3, "被": 3,
    "做成": 2, "做好": 2,
}

SYNONYM_TABLE = {
    # 動詞類
    "成立": ("建立", 3), "制造": ("做", 1), "观察": ("看", 1),
    "思考": ("想", 1), "奉献": ("付出", 4), "训练": ("教导", 3),
    "改变": ("变化", 3), "发现": ("找到", 2), "建造": ("建", 2),
    "完成": ("做好", 2), "保护": ("爱护", 3), "利用": ("使用", 3),
    "产生": ("出现", 3), "发展": ("成长", 3), "进行": ("做", 1),
    "表示": ("说", 1), "提供": ("给", 1), "影响": ("关系", 3),
    "解决": ("处理", 4), "努力": ("用心", 3), "坚持": ("一直做", 2),
    "创造": ("做出", 1), "研究": ("学习", 1), "收集": ("收", 2),
    "保存": ("留下", 2), "传播": ("传出", 3), "消失": ("不见", 2),
    "减少": ("变少", 2), "增加": ("变多", 2), "帮助": ("帮", 1),
    "创立": ("建立", 3),
    # 名詞類
    "印象": ("感觉", 3), "材料": ("东西", 1), "步骤": ("方法", 3),
    "技术": ("方法", 3), "价格": ("价钱", 3), "原因": ("因为", 1),
    "结果": ("后来", 2), "目的": ("想要", 1), "特点": ("特别", 3),
    "优点": ("好处", 3), "缺点": ("不好", 1), "经验": ("方法", 3),
    "力量": ("力气", 3), "作用": ("用处", 3), "问题": ("事情", 1),
    # 形容詞類
    "珍贵": ("很好", 1), "丰富": ("很多", 1), "严重": ("很大", 1),
    "困难": ("不容易", 2), "危险": ("不安全", 2), "美丽": ("美", 2),
    "特殊": ("特别", 3), "普通": ("一般", 3), "传统": ("以前的", 2),
    # 副詞/連接詞
    "因此": ("所以", 2), "然而": ("但是", 2), "逐渐": ("慢慢", 2),
    "终于": ("最后", 2), "究竟": ("到底", 4), "仍然": ("还是", 2),
    "甚至": ("还有", 2), "成千上万": ("很多", 1),
}

for cw_simp in PROPER_NOUNS_SIMP:
    jieba.add_word(cw_simp)
    jieba.add_word(sp2tw.convert(cw_simp))

jieba.suggest_freq(('护士', '学校'), True)
jieba.suggest_freq(('第一', '所'), True)

# ==========================================
# 3. 動態跳過判斷（取代靜態白名單）
# ==========================================
def should_skip(w_simp, hsk_dict):
    if w_simp in PROPER_NOUNS_SIMP:
        return True
    if len(w_simp) == 1:           # 單字跳過
        return True
    level_val = hsk_dict.get(w_simp, SUPPLEMENT_DICT.get(w_simp, 999))
    if level_val == 999 and w_simp not in SYNONYM_TABLE:
        return True                 # 查不到且無替換詞 → 跳過
    return False

# ==========================================
# 4. 靜默替換函數
# ==========================================
def process_sentence_silent(sent, hsk_dict, target_level):
    simplified_sent = tw2sp.convert(sent)
    tokens_simp = list(jieba.cut(simplified_sent))
    result_tokens = []

    for w_simp in tokens_simp:
        if not w_simp.strip() or w_simp in ["。","，","、","！","？"]:
            result_tokens.append(sp2tw.convert(w_simp))
            continue

        w_trad = sp2tw.convert(w_simp)

        if should_skip(w_simp, hsk_dict):
            result_tokens.append(w_trad)
            continue

        level_val = hsk_dict.get(w_simp, SUPPLEMENT_DICT.get(w_simp, 999))
        if level_val > target_level and w_simp in SYNONYM_TABLE:
            repl_simp, new_level = SYNONYM_TABLE[w_simp]
            if new_level <= target_level:
                result_tokens.append(sp2tw.convert(repl_simp))
                continue

        result_tokens.append(w_trad)

    return "".join(result_tokens)

# ==========================================
# 5. 載入 LoRA 模型
# ==========================================
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH  = "F:/download/專題/lora_weights"

print("【載入模型】載入 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("【載入模型】載入基底模型（4-bit 量化）...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

print("【載入模型】掛載 LoRA 權重...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("✅ 模型載入完成！\n")

# ==========================================
# 6. 模型推論函數
# ==========================================
def model_generate(sent, level):
    limits = {1: 20, 2: 15, 3: 12}
    limit = limits[level]
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
            **inputs,
            max_new_tokens=80,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    if "。" in generated:
        generated = generated.split("。")[0] + "。"
    return generated

# ==========================================
# 7. 幻覺檢查
# ==========================================
def is_hallucination(original, generated):
    """
    只過濾新增的專有名詞/地名（真正的幻覺）
    不過濾同義詞替換
    """
    orig_simp = tw2sp.convert(original)
    gen_simp  = tw2sp.convert(generated)
    gen_words = [w for w in jieba.cut(gen_simp) if len(w) >= 2]
    
    # 只檢查：生成詞是大寫開頭或已知地名模式
    suspicious = {"伦敦", "北京", "上海", "美国", "英国", "日本",
                  "法国", "德国", "俄罗斯", "印度", "荷兰人"}
    
    for w in gen_words:
        if w in suspicious and w not in orig_simp:
            return True, sp2tw.convert(w)
    
    return False, None

# ==========================================
# 8. 完整管線控制器
# ==========================================
def postprocess_with_retry(sent, hsk_dict, target_level, model_fn=None, max_retry=3):
    limits = {1: 20, 2: 15, 3: 12}
    char_limit = limits.get(target_level, 999)
    punctuations = ["。","，","、","！","？","：","；","「","」"]
    result = sent

    for attempt in range(max_retry):
        result = process_sentence_silent(result, hsk_dict, target_level)
        clean = result
        for p in punctuations:
            clean = clean.replace(p, "")
        is_compliant = len(clean) <= char_limit
        print(f"  [嘗試 {attempt+1}/{max_retry}] {result}（{len(clean)} 字）{'✅' if is_compliant else '❌ 超標'}")

        if is_compliant:
            break
        if model_fn and attempt < max_retry - 1:
            print(f"  🤖 呼叫 LLM 進行重寫...")
            new_result = model_fn(sent, target_level)
            hallucinated, word = is_hallucination(sent, new_result)
            if hallucinated:
                print(f"  ⚠️ 偵測到幻覺詞彙：{word}，跳過此次重寫")
            else:
                result = new_result
        else:
            break

    return result

# ==========================================
# 9. 執行測試
# ==========================================
test_cases = [
    ("最後他用樹皮、破布、廢漁網和麻布製造出紙張。", 1),
    ("南丁格爾成立了世界上第一所護士學校，訓練了成千上萬的護理人員。", 1),
]

if __name__ == "__main__":
    print("============= 完整管線測試 =============")
    for sent, level in test_cases:
        print(f"\n📝 輸入：{sent}")
        result = postprocess_with_retry(sent, my_hsk_dict, level, model_fn=model_generate)
        print(f"🌟 最終輸出：{result}")
        print("-" * 50)
