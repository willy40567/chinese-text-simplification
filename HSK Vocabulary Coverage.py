import json
import os
import jieba
import opencc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===== HSK 字典載入 =====
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
print(f"✅ HSK 字典載入，共 {len(my_hsk_dict)} 詞")

# 載入工具（沿用之前設定）
tw2sp = opencc.OpenCC('tw2sp')
my_hsk_dict  # 已載入

SUPPLEMENT_DICT = {
    "第一":2,"所":2,"和":1,"的":1,"了":1,"在":1,"是":1,
    "有":1,"不":1,"也":1,"都":2,"很":1,"就":2,"把":3,"被":3,
}

def get_hsk_coverage(sent, hsk_dict, target_levels=(1,2,3,4)):
    """計算句子中目標 HSK 等級詞彙佔比"""
    simp = tw2sp.convert(sent)
    words = [w for w in jieba.cut(simp) 
             if w.strip() and w not in "。，、！？：；「」"]
    
    if not words:
        return 0, 0, 0
    
    in_target = 0
    unknown   = 0
    
    for w in words:
        lv = hsk_dict.get(w, SUPPLEMENT_DICT.get(w, 999))
        if lv in target_levels:
            in_target += 1
        elif lv == 999:
            unknown += 1
    
    coverage = in_target / len(words)
    return coverage, in_target, len(words)

# 載入消融結果
with open("F:/download/專題/ablation_results.json", encoding="utf-8") as f:
    ablation = json.load(f)

# 計算各組覆蓋率
print("=" * 60)
print("HSK 1-4 詞彙覆蓋率：A 組 vs C 組")
print("=" * 60)

a_coverages = []
c_coverages = []

for a_item, c_item in zip(ablation["group_A"], ablation["group_C"]):
    sample_id = a_item["id"]
    level     = a_item["level"]
    
    cov_a, in_a, total_a = get_hsk_coverage(a_item["output"], my_hsk_dict)
    cov_c, in_c, total_c = get_hsk_coverage(c_item["output"], my_hsk_dict)
    
    diff = cov_c - cov_a
    flag = "⬆️ 提升" if diff > 0 else ("➡️ 持平" if diff == 0 else "⬇️ 下降")
    
    print(f"\n▶ {sample_id} (L{level})")
    print(f"  A 純LoRA   : {cov_a:.1%} ({in_a}/{total_a} 詞在 HSK 1-4)")
    print(f"  C 完整系統 : {cov_c:.1%} ({in_c}/{total_c} 詞在 HSK 1-4)")
    print(f"  變化       : {diff:+.1%} {flag}")
    
    a_coverages.append(cov_a)
    c_coverages.append(cov_c)

avg_a = sum(a_coverages) / len(a_coverages)
avg_c = sum(c_coverages) / len(c_coverages)

print("\n" + "=" * 60)
print("彙總")
print("=" * 60)
print(f"  A 組平均 HSK 1-4 覆蓋率：{avg_a:.1%}")
print(f"  C 組平均 HSK 1-4 覆蓋率：{avg_c:.1%}")
print(f"  平均提升幅度            ：{avg_c - avg_a:+.1%}")

# 儲存結果
summary = {
    "group_A_avg_hsk14_coverage": round(avg_a, 4),
    "group_C_avg_hsk14_coverage": round(avg_c, 4),
    "improvement": round(avg_c - avg_a, 4),
    "details": [
        {
            "id": a["id"],
            "level": a["level"],
            "A_coverage": round(get_hsk_coverage(a["output"], my_hsk_dict)[0], 4),
            "C_coverage": round(get_hsk_coverage(c["output"], my_hsk_dict)[0], 4),
        }
        for a, c in zip(ablation["group_A"], ablation["group_C"])
    ]
}

with open("F:/download/專題/hsk_coverage_analysis.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n✅ 結果已儲存至 hsk_coverage_analysis.json")