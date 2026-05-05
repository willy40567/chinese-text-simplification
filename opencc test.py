from datasets import load_dataset
import opencc

print("正在讀取資料集...")
base_path = "F:/download/專題/mcts-main"

train_complex_ds = load_dataset("text", data_files=f"{base_path}/pseudo_data/zh_selected.ori", split="train")
train_simple_ds = load_dataset("text", data_files=f"{base_path}/pseudo_data/zh_selected.sim", split="train")

print("資料集讀取完成！開始進行過濾與繁簡轉換...\n")

# 初始化 OpenCC
converter = opencc.OpenCC('s2twp')  # 簡體→繁體（台灣用語）

# 定義大陸特有詞彙黑名單 (簡體)
blacklist = ['人民币', '公安', '高铁', '人大', '政协', '党委', 
             '省委', '市委', '央视', '新华社', '解放军']

# 定義教育語域白名單 (繁體)
edu_keywords = ['動物', '植物', '科學', '自然', '歷史', '人物', 
                '發明', '身體', '食物', '學校', '運動', '地理',
                '製造', '工具', '生活', '環境', '水', '空氣']

def is_valid(text):
    return not any(word in text for word in blacklist)

# 進行篩選與轉換
filtered = []
for orig, simp in zip(train_complex_ds, train_simple_ds):
    orig_text = orig['text'].strip()
    simp_text = simp['text'].strip()
    
    # 關卡 1：黑名單過濾 (簡體狀態下檢查)
    if not is_valid(orig_text) or not is_valid(simp_text):
        continue
        
    # 關卡 2：長度篩選
    if len(orig_text) > 20 and len(simp_text) <= 20:
        
        # 先將句子轉換成繁體，因為我們的白名單關鍵字是繁體
        complex_tw = converter.convert(orig_text)
        simple_tw = converter.convert(simp_text)
        
        # 關卡 3：教育語域白名單過濾 (繁體狀態下檢查)
        if any(kw in complex_tw for kw in edu_keywords):
            filtered.append({
                "complex": complex_tw,
                "simple": simple_tw
            })
    
    # 只要收集滿 200 筆符合所有條件的資料就停止
    if len(filtered) >= 200:  
        break

# 印出結果
print(f"符合所有條件筆數：{len(filtered)} 筆")
print("\n教育語域範例預覽：")
for item in filtered[:5]:
    print(f"原句：{item['complex']}")
    print(f"簡化：{item['simple']}")
    print("-" * 30)