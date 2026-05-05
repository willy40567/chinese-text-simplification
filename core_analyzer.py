"""
core_analyzer.py - 中文文本句長分析核心引擎
用於閱讀障礙文本簡化研究 (台灣國小四年級課本)
"""
import re
import json
from datetime import datetime

# 三級標準定義
LEVEL_STANDARDS = {
    "level1": {"max_chars": 20, "hsk_range": "1-4", "desc": "閱讀障礙友善基礎版"},
    "level2": {"max_chars": 15, "hsk_range": "1-3", "desc": "進階簡化版"},
    "level3": {"max_chars": 12, "hsk_range": "1-2", "desc": "極簡版"},
}

def split_sentences(text):
    """
    依標點符號切分句子，排除標點後計算字符數。
    支援：。！？…… 及換行
    """
    # 清理空白
    text = text.strip()
    # 切分句子（保留句末標點）
    sentences = re.split(r'(?<=[。！？])', text)
    # 清理空字串與空白
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def count_chars(sentence):
    """
    計算有效字符數（排除標點符號、空格、引號）
    """
    # 排除標點、空格、引號、書名號等
    punctuation = r'[，。！？、；：「」『』《》〈〉【】（）…—\s\-\u2014\u2026""'']'
    clean = re.sub(punctuation, '', sentence)
    return len(clean)

def analyze_text(text):
    """
    分析文本，回傳各句字符數與合規情況
    
    Returns:
        dict: {
            "sentences": [{"text": str, "char_count": int}],
            "total_sentences": int,
            "avg_length": float,
            "max_length": int,
            "compliance": {"level1": bool, "level2": bool, "level3": bool}
        }
    """
    sentences = split_sentences(text)
    analyzed = []
    
    for s in sentences:
        count = count_chars(s)
        analyzed.append({
            "text": s,
            "char_count": count
        })
    
    if not analyzed:
        return None
    
    lengths = [s["char_count"] for s in analyzed]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    
    compliance = {}
    for level, std in LEVEL_STANDARDS.items():
        compliance[level] = all(l <= std["max_chars"] for l in lengths)
    
    return {
        "sentences": analyzed,
        "total_sentences": len(analyzed),
        "avg_length": round(avg_len, 1),
        "max_length": max_len,
        "compliance": compliance
    }

def suggest_splits(sentence, max_chars=20):
    """
    智能切分建議：基於逗號、頓號識別語意邊界
    
    Args:
        sentence: 原始句子
        max_chars: 目標最大字符數
    
    Returns:
        list: 切分後的句子列表
    """
    if count_chars(sentence) <= max_chars:
        return [sentence]
    
    # 依逗號、頓號切分
    parts = re.split(r'[，、]', sentence)
    parts = [p.strip() for p in parts if p.strip()]
    
    result = []
    current = ""
    
    for part in parts:
        candidate = current + part if not current else current + "，" + part
        if count_chars(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                # 補句號
                result.append(current + "。")
            current = part
    
    if current:
        # 保留原句尾標點
        last_char = sentence[-1] if sentence else ""
        if last_char in "。！？":
            result.append(current + last_char)
        else:
            result.append(current + "。")
    
    return result if result else [sentence]

def create_annotation_template(lesson_id, original_text, source_file=""):
    """
    建立標準化標註模板 (JSON格式)
    """
    analysis = analyze_text(original_text)
    
    template = {
        "lesson_id": lesson_id,
        "source_file": source_file,
        "original_text": original_text,
        "original_analysis": analysis,
        "simplified_versions": {
            "level1": {
                "text": "",
                "standard": LEVEL_STANDARDS["level1"],
                "compliance": None,
                "sentence_count": None,
                "avg_sentence_length": None,
                "annotator_notes": ""
            },
            "level2": {
                "text": "",
                "standard": LEVEL_STANDARDS["level2"],
                "compliance": None,
                "sentence_count": None,
                "avg_sentence_length": None,
                "annotator_notes": ""
            },
            "level3": {
                "text": "",
                "standard": LEVEL_STANDARDS["level3"],
                "compliance": None,
                "sentence_count": None,
                "avg_sentence_length": None,
                "annotator_notes": ""
            }
        },
        "metadata": {
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "annotation_status": "pending",  # pending / in_progress / completed
            "quality_score": None
        }
    }
    return template

def validate_simplified(text, level="level1"):
    """
    驗證簡化版本是否符合指定級別標準
    
    Returns:
        dict: {
            "compliant": bool,
            "violations": list,
            "analysis": dict
        }
    """
    std = LEVEL_STANDARDS.get(level, LEVEL_STANDARDS["level1"])
    max_chars = std["max_chars"]
    
    analysis = analyze_text(text)
    if not analysis:
        return {"compliant": False, "violations": ["文本為空"], "analysis": None}
    
    violations = []
    for s in analysis["sentences"]:
        if s["char_count"] > max_chars:
            violations.append(
                f"「{s['text'][:15]}...」({s['char_count']}字 > {max_chars}字上限)"
            )
    
    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "analysis": analysis
    }


if __name__ == "__main__":
    # 測試範例
    print("=== core_analyzer.py 功能測試 ===\n")
    
    test_text = "才走到夜市附近，空氣中已飄來好多的香味——蚵仔煎、臭豆腐、炸雞排，還有紅豆餅的香甜味，哇！我的口水都快流出來了。"
    
    print(f"原文：{test_text}")
    print(f"原文字數：{count_chars(test_text)} 字\n")
    
    analysis = analyze_text(test_text)
    print(f"句子數：{analysis['total_sentences']}")
    print(f"平均句長：{analysis['avg_length']}")
    print(f"最長句：{analysis['max_length']}")
    print(f"合規性：{analysis['compliance']}\n")
    
    print("--- 智能切分建議 (Level 1, ≤20字) ---")
    for s in analysis["sentences"]:
        suggestions = suggest_splits(s["text"], max_chars=20)
        if len(suggestions) > 1:
            print(f"原句({s['char_count']}字)：{s['text']}")
            for i, sug in enumerate(suggestions, 1):
                print(f"  切分{i}({count_chars(sug)}字)：{sug}")
        else:
            print(f"合規({s['char_count']}字)：{s['text']}")
    
    print("\n✅ 測試完成")
