#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Web UI (ui.py)

把 Gradio 介面掛進 app.py 的 FastAPI 服務（路徑 /ui），與 API 共用同一個
SimplificationPipeline 與同一把推論鎖 → 單張 GPU 也安全。

啟動（同時提供 API 與 Web UI）：
  uvicorn ui:app --host 0.0.0.0 --port 8000
  - Web UI:  http://localhost:8000/ui
  - API 文件: http://localhost:8000/docs
"""

import gradio as gr

from app import app, run_simplify

SAMPLE = (
    "南丁格爾是一位偉大的護士，她在戰爭期間照顧許多受傷的士兵，"
    "並且改善了醫院的衛生環境，因此被後人尊稱為提燈天使。"
)


def _simplify(text):
    pipe = getattr(app.state, "pipeline", None)
    if pipe is None or not getattr(pipe, "ready", False):
        return "⚠️ 模型尚未就緒，請稍候再試。", "", ""
    if not text or not text.strip():
        return "⚠️ 請先輸入文字。", "", ""
    out = run_simplify(pipe, text)
    return out["l1"], out["l2"], out["l3"]


with gr.Blocks(title="中文文本三級簡化系統") as demo:
    gr.Markdown(
        "# 中文文本三級簡化系統\n"
        "輸入課文，產出三個難度遞減的簡化版本（L1 ≤20 字、L2 ≤15 字、L3 ≤12 字／句）。"
    )
    inp = gr.Textbox(lines=6, label="原文", value=SAMPLE)
    btn = gr.Button("開始簡化", variant="primary")
    with gr.Row():
        o1 = gr.Textbox(label="L1（輕度，≤20 字）", lines=6)
        o2 = gr.Textbox(label="L2（中度，≤15 字）", lines=6)
        o3 = gr.Textbox(label="L3（重度，≤12 字）", lines=6)
    btn.click(_simplify, inputs=inp, outputs=[o1, o2, o3])


# 掛進 FastAPI，回傳同一個（已掛載）app 供 uvicorn 使用
app = gr.mount_gradio_app(app, demo, path="/ui")
