#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py 端到端煙霧測試（需 GPU / 會載入 Llama-3.1-8B + LoRA，約 30–60 秒）。

用途：驗證抽取後的 pipeline.simplify_text 行為與原 GUI 一致。
跑法：  python smoke_test.py
"""

from pipeline import SimplificationPipeline

SAMPLE = (
    "南丁格爾是一位偉大的護士，她在戰爭期間照顧許多受傷的士兵，"
    "並且改善了醫院的衛生環境，因此被後人尊稱為提燈天使。"
)


def main():
    pipe = SimplificationPipeline()
    pipe.load()  # 預設用 print 當 log

    print("\n========== 輸入 ==========")
    print(SAMPLE)

    out = pipe.simplify_text(SAMPLE)

    print("\n========== 三級簡化結果 ==========")
    print(f"L1（≤20字）：{out['l1']}")
    print(f"L2（≤15字）：{out['l2']}")
    print(f"L3（≤12字）：{out['l3']}")

    # 基本健全性檢查
    assert out["l1"] and out["l2"] and out["l3"], "三級輸出不應為空"
    print("\n✅ 煙霧測試通過：三級皆有輸出。")


if __name__ == "__main__":
    main()
