#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py 純邏輯單元測試（不需載入模型／GPU）。

兩種跑法皆可：
  pytest tests/                       # 有裝 pytest
  python tests/test_pipeline_logic.py # 無 pytest，直接執行
"""

import os
import sys

# 讓測試可在 tests/ 子目錄下找到專案根的 pipeline.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    split_sentences,
    is_heading_like,
    is_valid_chinese_output,
    normalize_quote_punctuation,
    apply_text_corrections,
    similarity_filter,
    is_hallucination,
)


def test_split_sentences_basic():
    out = split_sentences("今天天氣很好。我們去公園。")
    assert out == ["今天天氣很好。", "我們去公園。"]


def test_split_sentences_long_splits_on_comma():
    # 一句超過 40 字 → 以逗號/頓號再切，且保留分隔符
    long = ("小明今天早上起床之後先去刷牙然後洗臉，"
            "接著他吃了一頓非常豐盛的早餐然後背起書包，"
            "最後沿著大馬路慢慢走到學校準備上課。")
    assert len(long) > 40  # 確保會觸發切分
    out = split_sentences(long)
    assert len(out) >= 2
    assert any("，" in seg for seg in out[:-1])


def test_split_sentences_drops_empty():
    assert split_sentences("   ") == []


def test_is_heading_like():
    assert is_heading_like("第一章 緒論") is True
    assert is_heading_like("一、研究背景") is True
    assert is_heading_like("（二）研究目的") is True
    assert is_heading_like("書名：") is True
    assert is_heading_like("短標題") is True          # 短、無句末標點、無逗號
    assert is_heading_like("今天天氣很好，我們去公園玩耍。") is False


def test_is_valid_chinese_output():
    assert is_valid_chinese_output("這是一段中文") is True
    assert is_valid_chinese_output("") is False
    assert is_valid_chinese_output("好") is False                 # 太短
    assert is_valid_chinese_output("hello world test") is False   # 連續英文單字
    assert is_valid_chinese_output("這是中文 # 註解") is False     # 含 #


def test_normalize_quote_punctuation_fixes_question():
    # 句末為疑問語氣詞 + 。」→ 應改成 ？」
    assert normalize_quote_punctuation("他問你好嗎。」").endswith("好嗎？」")


def test_apply_text_corrections_typo():
    assert apply_text_corrections("一撇即分的香菇") == "一撥即分的香菇"
    assert apply_text_corrections("幹香菇很好吃") == "乾香菇很好吃"


def test_similarity_filter_keeps_quoted():
    # 含引號的句子應原樣保留（不過濾）
    src = "老師說我們要努力讀書"
    pred = "老師說「要努力」"
    assert "「" in similarity_filter(src, pred, semantic_model=None)


def test_similarity_filter_no_src_returns_pred():
    assert similarity_filter("", "任意內容。", semantic_model=None) == "任意內容。"


def test_is_hallucination_flags_new_place():
    # 原文沒有地名、輸出冒出「北京」→ 判為幻覺
    flagged, word = is_hallucination("我們去公園散步", "我們去北京散步")
    assert flagged is True
    assert word == "北京"


def test_is_hallucination_passes_when_place_in_source():
    # 原文本就有「北京」→ 不算幻覺
    flagged, _ = is_hallucination("我們去北京玩", "我們去北京玩")
    assert flagged is False


# ---------- 無 pytest 時的執行入口 ----------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed (共 {len(tests)} 項)")
    sys.exit(1 if failed else 0)
