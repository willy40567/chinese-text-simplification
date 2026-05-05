#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSK 詞彙工具模組 (hsk_utils.py)
統一管理 HSK 字典載入、同義詞替換、專有名詞、輔助詞典等共用邏輯。
"""

import os
import re
import json

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import opencc
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False

from config import (
    HSK_FOLDER, HSK_FOLDER_JSON, HSK_VOCAB_TXT, LEVEL_MAPPING
)

# ==========================================
# 專有名詞 (簡體)
# ==========================================
PROPER_NOUNS_SIMP = {
    "南丁格尔", "蔡伦", "郑成功", "安平古堡", "台湾", "中国"
}

# ==========================================
# 輔助詞典 (未收錄於 HSK 但常見的基礎詞)
# ==========================================
SUPPLEMENT_DICT = {
    "第一": 2, "所": 2, "和": 1, "的": 1, "了": 1, "在": 1, "是": 1,
    "有": 1, "不": 1, "也": 1, "都": 2, "很": 1, "就": 2, "把": 3, "被": 3,
    "做成": 2, "做好": 2,
}

# ==========================================
# 同義詞替換表 (簡體 → (替換詞, HSK等級))
# ==========================================
SYNONYM_TABLE = {
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
    "创立": ("建立", 3), "印象": ("感觉", 3), "材料": ("东西", 1),
    "步骤": ("方法", 3), "技术": ("方法", 3), "价格": ("价钱", 3),
    "原因": ("因为", 1), "结果": ("后来", 2), "目的": ("想要", 1),
    "特点": ("特别", 3), "优点": ("好处", 3), "缺点": ("不好", 1),
    "经验": ("方法", 3), "力量": ("力气", 3), "作用": ("用处", 3),
    "问题": ("事情", 1), "珍贵": ("很好", 1), "丰富": ("很多", 1),
    "严重": ("很大", 1), "困难": ("不容易", 2), "危险": ("不安全", 2),
    "美丽": ("美", 2), "特殊": ("特别", 3), "普通": ("一般", 3),
    "传统": ("以前的", 2), "因此": ("所以", 2), "然而": ("但是", 2),
    "逐渐": ("慢慢", 2), "终于": ("最后", 2), "究竟": ("到底", 4),
    "仍然": ("还是", 2), "甚至": ("还有", 2), "成千上万": ("很多", 1),
}

# ==========================================
# OpenCC 轉換器 (延遲初始化)
# ==========================================
_tw2sp = None
_sp2tw = None


def get_tw2sp():
    """取得繁→簡轉換器 (lazy init)"""
    global _tw2sp
    if _tw2sp is None:
        if not OPENCC_AVAILABLE:
            raise ImportError("opencc 未安裝，請執行: pip install opencc-python-reimplemented")
        _tw2sp = opencc.OpenCC('tw2sp')
    return _tw2sp


def get_sp2tw():
    """取得簡→繁轉換器 (lazy init)"""
    global _sp2tw
    if _sp2tw is None:
        if not OPENCC_AVAILABLE:
            raise ImportError("opencc 未安裝，請執行: pip install opencc-python-reimplemented")
        _sp2tw = opencc.OpenCC('s2twp')
    return _sp2tw


# ==========================================
# HSK 字典載入
# ==========================================
def build_hsk_vocab_txt(folder_path=HSK_FOLDER, output_path=HSK_VOCAB_TXT):
    """從 HSK TXT 檔案建立合併詞表 (hsk_vocab.txt)"""
    with open(output_path, "w", encoding="utf-8") as out_f:
        for filename, level in LEVEL_MAPPING.items():
            filepath = os.path.join(folder_path, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        parts = line.strip().split()
                        if parts:
                            out_f.write(f"{parts[0]}\t{level}\n")
    return output_path


def load_hsk_dict(path=HSK_VOCAB_TXT):
    """從 hsk_vocab.txt 載入 HSK 詞典 {詞: 等級}"""
    hsk = {}
    if not os.path.exists(path):
        return hsk
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                hsk[parts[0]] = int(parts[1])
    return hsk


def load_hsk_dict_json(folder=HSK_FOLDER_JSON):
    """從新版 JSON 檔案載入 HSK 詞典 (含繁體字)"""
    hsk = {}
    for lv in range(1, 8):
        path = os.path.join(folder, f"{lv}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            words_data = json.load(f)
            for item in words_data:
                word_forms = set()
                simp = item.get("simplified")
                if simp:
                    word_forms.add(simp)
                for form in item.get("forms", []):
                    trad = form.get("traditional")
                    if trad:
                        word_forms.add(trad)
                for w in word_forms:
                    if w not in hsk:
                        hsk[w] = lv
    return hsk


# ==========================================
# HSK 替換管線
# ==========================================
def should_skip(w_simp, hsk_dict):
    """判斷詞彙是否應跳過替換 (專有名詞、單字、未收錄)"""
    if w_simp in PROPER_NOUNS_SIMP:
        return True
    if len(w_simp) == 1:
        return True
    lv = hsk_dict.get(w_simp, SUPPLEMENT_DICT.get(w_simp, 999))
    if lv == 999 and w_simp not in SYNONYM_TABLE:
        return True
    return False


def hsk_replace(sent, hsk_dict, target_level):
    """對句子進行 HSK 等級同義詞替換"""
    if not JIEBA_AVAILABLE:
        return sent
    tw2sp = get_tw2sp()
    sp2tw = get_sp2tw()
    tokens = list(jieba.cut(tw2sp.convert(sent)))
    out = []
    for w in tokens:
        if not w.strip() or w in "。，、！？：；「」":
            out.append(sp2tw.convert(w))
            continue
        wt = sp2tw.convert(w)
        if should_skip(w, hsk_dict):
            out.append(wt)
            continue
        lv = hsk_dict.get(w, SUPPLEMENT_DICT.get(w, 999))
        if lv > target_level and w in SYNONYM_TABLE:
            rs, rl = SYNONYM_TABLE[w]
            if rl <= target_level:
                out.append(sp2tw.convert(rs))
                continue
        out.append(wt)
    return "".join(out)


def register_proper_nouns_jieba():
    """將專有名詞加入 jieba 詞典 (繁簡雙版本)"""
    if not JIEBA_AVAILABLE:
        return
    sp2tw = get_sp2tw()
    for cw in PROPER_NOUNS_SIMP:
        jieba.add_word(cw)
        jieba.add_word(sp2tw.convert(cw))


# ==========================================
# 合規率計算
# ==========================================
def char_count(sent):
    """計算句子有效字數 (不含標點)"""
    puncts = "。，、！？：；「」『』（）"
    return len("".join(c for c in sent if c not in puncts))


def compliance_rate(output, level, limits=None):
    """計算句子長度合規率"""
    from config import LEVEL_LIMITS
    if limits is None:
        limits = LEVEL_LIMITS
    limit = limits.get(level, 999)
    sentences = [s + "。" for s in output.replace("。", "\n").split("\n") if s.strip()]
    if not sentences:
        sentences = [output]
    ok = sum(1 for s in sentences if char_count(s) <= limit)
    return ok / len(sentences) if sentences else 0
