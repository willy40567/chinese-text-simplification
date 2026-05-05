import json
import re
import os
import jieba
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 設定路徑 (從 config.py 統一管理) ---
from config import TRAINING_DATA as JSONL_PATH, HSK_FOLDER_JSON as HSK_FOLDER
TXT_FOLDER = "F:/download/Documents/國文課本/txt_output"  # 外部路徑，不由 config 管理
OUTPUT_JSON = "hsk_coverage_report_final.json"

# --- 2. 設定 matplotlib 支援中文顯示 (Windows 預設微軟正黑體) ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

from hsk_utils import load_hsk_dict_json as load_hsk_dict

def get_hsk_stats(text, hsk_dict):
    """計算文本的 HSK 覆蓋率與低級詞比例 (直接使用繁體文本)"""
    if not text: return {"coverage": 0, "low_level_ratio": 0, "total_tokens": 0}
    
    # 移除 opencc，直接對繁體原文進行分詞 (只保留中文字)
    tokens = [w for w in jieba.cut(text) if re.search(r'[\u4e00-\u9fff]', w)]
    
    counts = defaultdict(int)
    for w in tokens:
        lv = hsk_dict.get(w)
        if lv: counts[lv] += 1
        else:  counts['unknown'] += 1
    
    total = sum(counts.values())
    known = total - counts['unknown']
    low_level_sum = sum(counts[i] for i in [1, 2, 3])
    
    return {
        "coverage": known / total if total > 0 else 0.0,
        "low_level_ratio": low_level_sum / total if total > 0 else 0.0,
        "total_tokens": total,
    }

def draw_chart(summary):
    """繪製 Before/After 對比長條圖"""
    mapping = {
        "Before_Original": "原始課文\n(Before)",
        "After_L1": "L1 標註\n(After)",
        "After_L2": "L2 標註\n(After)",
        "After_L3": "L3 標註\n(After)"
    }
    
    keys_to_plot = [k for k in ["Before_Original", "After_L1", "After_L2", "After_L3"] if k in summary]
    x_ticks = [mapping[k] for k in keys_to_plot]
    coverage_data = [summary[k]["avg_coverage"] * 100 for k in keys_to_plot]
    low_level_data = [summary[k]["avg_low_level_ratio"] * 100 for k in keys_to_plot]

    x = np.arange(len(keys_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, coverage_data, width, label='總 HSK 覆蓋率 (%)', color='#4CAF50')
    rects2 = ax.bar(x + width/2, low_level_data, width, label='低級詞 (L1-L3) 比例 (%)', color='#2196F3')

    ax.set_ylabel('百分比 (%)')
    ax.set_title('文本簡化成效：原始課文 vs 各級標註結果 (Before / After)', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.legend(loc='lower right')

    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')

    fig.tight_layout()
    plt.savefig("hsk_before_after_chart.png", dpi=300)
    print("\n📊 圖表已生成並儲存為：hsk_before_after_chart.png")
    plt.show()

def main():
    print("🧠 最強大腦啟動中...")
    hsk_dict = load_hsk_dict()
    print(f"✅ HSK 繁體/簡體 詞典載入完成：共彙整 {len(hsk_dict)} 個詞彙型態")

    # --- Step 1: 讀取 Before (原始課文) ---
    raw_texts = []
    target_pattern = re.compile(r'chi-(3-[1-9]|4-[1-3])-book-1.*\.txt')
    
    if os.path.exists(TXT_FOLDER):
        for fname in os.listdir(TXT_FOLDER):
            if target_pattern.search(fname):
                with open(os.path.join(TXT_FOLDER, fname), encoding="utf-8") as f:
                    raw_texts.append(f.read())
    print(f"📚 讀取 Before (原始課文)：{len(raw_texts)} 篇")

    # --- Step 2: 讀取 After (標註後成果) ---
    buckets = {1: [], 2: [], 3: []}
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line.strip())
                lv = s.get("metadata", {}).get("level")
                out = s.get("output", "").strip()
                if lv in buckets and out:
                    buckets[lv].append(out)
    else:
        print(f"⚠️ 找不到 JSONL 檔案：{JSONL_PATH}")

    # --- Step 3: 進行數據運算與成長幅度計算 ---
    summary = {}
    
    orig_cov_avg = 0.0
    orig_low_avg = 0.0
    if raw_texts:
        orig_stats = [get_hsk_stats(t, hsk_dict) for t in raw_texts]
        orig_cov_avg = sum(x["coverage"] for x in orig_stats) / len(orig_stats)
        orig_low_avg = sum(x["low_level_ratio"] for x in orig_stats) / len(orig_stats)
        summary["Before_Original"] = {
            "avg_coverage": orig_cov_avg,
            "avg_low_level_ratio": orig_low_avg,
            "n": len(orig_stats)
        }

    for lv in [1, 2, 3]:
        if not buckets[lv]: continue
        stats = [get_hsk_stats(out_text, hsk_dict) for out_text in buckets[lv]]
        lv_cov_avg = sum(x["coverage"] for x in stats) / len(stats)
        lv_low_avg = sum(x["low_level_ratio"] for x in stats) / len(stats)
        
        summary[f"After_L{lv}"] = {
            "avg_coverage": lv_cov_avg,
            "avg_low_level_ratio": lv_low_avg,
            "n": len(stats),
            "diff_coverage": lv_cov_avg - orig_cov_avg,
            "diff_low_level": lv_low_avg - orig_low_avg
        }

    # --- Step 4: 輸出具有衝擊力的終端機表格 ---
    print("\n" + "="*75)
    print(f"{'比較項目 (Before vs After)':<22} {'HSK覆蓋率':>10} {'(成長幅度)':>10} {'低級詞比例':>10} {'(成長幅度)':>10}")
    print("-" * 75)
    
    if "Before_Original" in summary:
        s = summary["Before_Original"]
        print(f"['Before'] 原始課文{'':<8} {s['avg_coverage']:>9.2%} {'-':>10} {s['avg_low_level_ratio']:>9.2%} {'-':>10}")
    
    for lv in [1, 2, 3]:
        field = f"After_L{lv}"
        if field in summary:
            s = summary[field]
            cov_sign = "+" if s['diff_coverage'] > 0 else ""
            low_sign = "+" if s['diff_low_level'] > 0 else ""
            
            print(f"['After']  標註為 L{lv}{'':<6} {s['avg_coverage']:>9.2%} {cov_sign+str(round(s['diff_coverage']*100, 2))+'%':>10} "
                  f"{s['avg_low_level_ratio']:>9.2%} {low_sign+str(round(s['diff_low_level']*100, 2))+'%':>10}")
    print("="*75)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # --- Step 5: 繪製圖表 ---
    if summary:
        draw_chart(summary)

if __name__ == "__main__":
    main()