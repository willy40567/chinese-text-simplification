#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 推論服務 (app.py)

對外端點：
  GET  /health            服務與模型就緒狀態
  POST /simplify          {text} -> {l1, l2, l3} 三級簡化

設計：
  - lifespan 啟動時載入 SimplificationPipeline 一次，存進 app.state.pipeline。
  - 單張 GPU，用 asyncio.Lock 序列化推論，避免並發爆顯存。
  - simplify_text 為同步阻塞，丟到 executor 執行，不卡住事件迴圈。

啟動：  uvicorn app:app --host 0.0.0.0 --port 8000
測試時可先設定 app.state.pipeline = <stub>，lifespan 偵測到已存在即略過載入。
"""

import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

from pipeline import SimplificationPipeline

# 單 GPU：序列化推論。用 threading.Lock 讓 API 與 Gradio UI 兩條路徑共用同一把鎖
# （兩者最終都在背景執行緒呼叫 simplify_text）。
_inference_lock = threading.Lock()


def run_simplify(pipe, text):
    """在鎖保護下執行推論，供 API 與 Gradio UI 共用。"""
    with _inference_lock:
        return pipe.simplify_text(text)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 若測試已注入 stub pipeline，則略過真實模型載入
    if getattr(app.state, "pipeline", None) is None:
        pipe = SimplificationPipeline()
        pipe.load()
        app.state.pipeline = pipe
    yield


app = FastAPI(title="中文文本三級簡化服務", version="1.0", lifespan=lifespan)


def get_pipeline():
    """取得已載入的 pipeline。"""
    pipe = getattr(app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型尚未就緒")
    return pipe


# ==========================================
# 資料模型
# ==========================================
class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待簡化的中文文本")


class SimplifyResponse(BaseModel):
    l1: str
    l2: str
    l3: str


# ==========================================
# 端點
# ==========================================
@app.get("/health")
def health():
    pipe = getattr(app.state, "pipeline", None)
    ready = bool(pipe is not None and getattr(pipe, "ready", False))
    return {"status": "ok", "ready": ready}


@app.post("/simplify", response_model=SimplifyResponse)
async def simplify(req: SimplifyRequest, pipe=Depends(get_pipeline)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不可為空白")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_simplify, pipe, req.text)
    return result
