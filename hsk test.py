import os
import jieba
import opencc
import warnings

# 🤫 忽略 Jieba 引起的 pkg_resources 棄用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 1. 產生並載入 HSK 字典
# ==========================================
folder_path = "F:/download/專題/New HSK (2025)/HSK Words"
output_file = "hsk_vocab.txt"

level_mapping = {
    "HSK_Level_1_words.txt": 1, "HSK_Level_2_words.txt": 2, "HSK_Level_3_words.txt": 3,
    "HSK_Level_4_words.txt": 4, "HSK_Level_5_words.txt": 5, "HSK_Level_6_words.txt": 6,
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
print(f"✅ 成功載入 HSK 字典，共 {len(my_hsk_dict)} 個詞彙！\n")

# ==========================================
# 2. 初始化轉換器與優化配置
# ==========================================
print("正在載入 Jieba 與優化配置...")
tw2sp = opencc.OpenCC('tw2sp')
sp2tw = opencc.OpenCC('s2twp')

PROPER_NOUNS_SIMP = {"南丁格尔", "蔡伦", "郑成功", "安平古堡", "台湾", "中国"}
SUPPLEMENT_DICT = {
    "第一": 2, "所": 2, "和": 1, "的": 1, "了": 1,
    "在": 1, "是": 1, "有": 1, "不": 1, "也": 1,
    "都": 2, "很": 1, "就": 2, "把": 3, "被": 3
}

for cw_simp in PROPER_NOUNS_SIMP:
    jieba.add_word(cw_simp)
    jieba.add_word(sp2tw.convert(cw_simp)) 

jieba.suggest_freq(('护士', '学校'), True)
jieba.suggest_freq(('第一', '所'), True)

# ==========================================
# 3. 建立同義詞替換表
# ==========================================
SYNONYM_TABLE = {
    "成立": ("建立", 3),
    "制造": ("做", 1),
    "观察": ("看", 1),
    "思考": ("想", 1),
    "奉献": ("付出", 4),
    "训练": ("教导", 3),
    "改变": ("变化", 3),
    "发现": ("找到", 2),
    "建造": ("建", 2),
    "完成": ("做好", 2),
    "保护": ("爱护", 3),
    "利用": ("使用", 3),
    "产生": ("出现", 3),
    "发展": ("成长", 3),
    "进行": ("做", 1),
    "表示": ("说", 1),
    "提供": ("给", 1),
    "影响": ("影响到", 3),
    "解决": ("处理", 4),
    "努力": ("用心", 3),
    "坚持": ("一直做", 2),
    "创造": ("做出", 1),
    "研究": ("学习", 1),
    "收集": ("收", 2),
    "保存": ("留下", 2),
    "传播": ("传出", 3),
    "消失": ("不见", 2),
    "减少": ("变少", 2),
    "增加": ("变多", 2),
    "帮助": ("帮", 1),
    "工作": ("做事", 1),
    # ===== 名詞類 =====
    "产业": ("工作", 5),
    "供应链": ("供货", 5),
    "印象": ("感觉", 3),
    "环境": ("地方", 1),
    "材料": ("东西", 1),
    "步骤": ("方法", 3),
    "技术": ("方法", 3),
    "价格": ("价钱", 3),
    "原因": ("因为", 1),
    "结果": ("后来", 2),
    "目的": ("想要", 1),
    "特点": ("特别", 3),
    "优点": ("好处", 3),
    "缺点": ("不好", 1),
    "经验": ("方法", 3),
    "力量": ("力气", 3),
    "作用": ("用处", 3),
    "方向": ("地方", 1),
    "问题": ("事情", 1),
    # ===== 形容詞類 =====
    "珍贵": ("很好", 1),
    "丰富": ("很多", 1),
    "严重": ("很大", 1),
    "困难": ("不容易", 2),
    "危险": ("不安全", 2),
    "美丽": ("美", 2),
    "特殊": ("特别", 3),
    "普通": ("一般", 3),
    "传统": ("以前的", 2),
    # ===== 副詞/連接詞 =====
    "因此": ("所以", 2),
    "然而": ("但是", 2),
    "逐渐": ("慢慢", 2),
    "终于": ("最后", 2),
    "究竟": ("到底", 4),
    "仍然": ("还是", 2),
    "甚至": ("还有", 2),
}

# ==========================================
# 4. 核心處理函數：靜默替換版
# ==========================================
def process_sentence_silent(sent, hsk_dict, target_level):
    """不印 log 的替換版本（供重試迴圈使用）"""
    simplified_sent = tw2sp.convert(sent)
    tokens_simp = list(jieba.cut(simplified_sent))
    result_tokens = []
    
    for w_simp in tokens_simp:
        if not w_simp.strip() or w_simp in ["。", "，", "、", "！", "？"]:
            result_tokens.append(sp2tw.convert(w_simp))
            continue
            
        w_trad = sp2tw.convert(w_simp)
        
        if w_simp in PROPER_NOUNS_SIMP:
            result_tokens.append(w_trad)
            continue
            
        level = hsk_dict.get(w_simp, SUPPLEMENT_DICT.get(w_simp, 999))
        
        # 精簡寫法：符合條件且字典有替換詞，且替換後難度達標
        if level > target_level and w_simp in SYNONYM_TABLE:
            repl_simp, new_level = SYNONYM_TABLE[w_simp]
            if new_level <= target_level:
                result_tokens.append(sp2tw.convert(repl_simp))
                continue
                
        # 否則保留原字(轉繁體)
        result_tokens.append(w_trad)
    
    return "".join(result_tokens)

# ==========================================
# 5. 管線控制器：重試與字數合規
# ==========================================
def postprocess_with_retry(sent, hsk_dict, target_level, model_fn=None, max_retry=3):
    """
    完整後處理管線：HSK 替換 + 字數合規重試
    model_fn: 你的模型生成函數（若為 None 則跳過重生成，僅做替換）
    """
    # 擴充字數限制，找不到對應等級時預設為 999 (無限制)
    limits = {1: 20, 2: 15, 3: 12, 4: 10} 
    char_limit = limits.get(target_level, 999) 
    
    result = sent
    punctuations = ["。", "，", "、", "！", "？", "：", "；", "「", "」"]
    
    for attempt in range(max_retry):
        # 執行 HSK 詞彙替換
        result = process_sentence_silent(result, hsk_dict, target_level)
        
        # 檢查字數合規 (精準去除所有常見標點)
        clean = result
        for p in punctuations:
            clean = clean.replace(p, "")
            
        is_compliant = len(clean) <= char_limit
        
        print(f"  第 {attempt+1} 次嘗試：{result}（{len(clean)} 字）{'✅' if is_compliant else '❌ 超標'}")
        
        if is_compliant:
            break
            
        # 若有模型函數且未達上限，重新生成後再試
        if model_fn and attempt < max_retry - 1:
            print(f"  ⚠️ 字數仍超標，呼叫模型進行第 {attempt+2} 次改寫...")
            result = model_fn(sent, target_level) 
        elif not model_fn:
            break  # 無模型函數，直接跳出保留當前結果
    
    return result

# ==========================================
# 6. 執行最終測試
# ==========================================
test_cases = [
    ("南丁格爾建立第一所護士學校。", 1),   # 應該合規
    ("最後他用樹皮、破布、廢漁網和麻布製造出紙張。", 1),  # 預期超標
]

print("\n============= 合規重試迴圈測試 =============")
for sent, level in test_cases:
    # 動態取得字數上限供 print 顯示
    max_chars = {1: 20, 2: 15, 3: 12, 4: 10}.get(level, 999)
    print(f"\n📝 輸入：{sent}（目標 L{level}，上限 {max_chars} 字）")
    
    result = postprocess_with_retry(sent, my_hsk_dict, level)
    
    print(f"📤 最終輸出：{result}")
    print("-" * 40)