# 服務層說明（FastAPI + Gradio）

本文件說明在原研究系統之上新增的**服務化層**：把三級簡化推論封裝成
FastAPI 服務，並提供 Gradio Web 介面。原桌面工具
（`simplification_gui.py`）維持不變。

## 架構

```
                       ┌──────────────────────────────┐
                       │        uvicorn ui:app         │
                       │      (單一進程 / 單張 GPU)      │
   瀏覽器  ─HTTP─▶  /ui │  Gradio Web UI                │
                       │     │                         │
   API 客戶端 ─HTTP─▶ /simplify  /health  /docs         │
                       │     │                         │
                       │     ▼   run_simplify()         │
                       │   threading.Lock（序列化）     │
                       │     │                         │
                       │     ▼                         │
                       │  SimplificationPipeline        │
                       │  （啟動時 load() 一次，常駐）   │
                       │   Llama-3.1-8B + LoRA (4-bit)  │
                       └──────────────────────────────┘
```

- **`pipeline.py`** — 推論核心。`SimplificationPipeline.load()` 啟動時載入
  HSK 字典、Jieba、Llama-3.1-8B + LoRA（4-bit NF4 量化）、Sentence-BERT；
  `simplify_text(text)` 回傳三級結果 `{"l1","l2","l3"}`（原文 → L1 → L2 → L3 鏈式）。
- **`app.py`** — FastAPI。lifespan 啟動載入模型一次；`/simplify` 把同步推論丟到
  executor 執行，並以 `threading.Lock` 序列化（單張 GPU 不可並發）。
- **`ui.py`** — Gradio 介面，透過 `gr.mount_gradio_app` 掛在 `/ui`，與 API
  共用同一個 pipeline 與同一把鎖。

## 快速開始

```bash
pip install -r requirements.txt

# 同時提供 API 與 Web UI
uvicorn ui:app --host 0.0.0.0 --port 8000
```

- Web UI：<http://localhost:8000/ui>
- API 文件（Swagger）：<http://localhost:8000/docs>
- 純 API（不要 Web UI）：`uvicorn app:app --port 8000`

> 首次啟動會載入 8B 模型（4-bit 量化，約 30–60 秒）。`/health` 回傳
> `{"ready": true}` 即代表模型就緒。

## API 參考

### `GET /health`
```json
{ "status": "ok", "ready": true }
```

### `POST /simplify`
請求：
```json
{ "text": "南丁格爾是一位偉大的護士，她在戰爭期間照顧許多受傷的士兵。" }
```
回應：
```json
{
  "l1": "……（每句 ≤20 字）",
  "l2": "……（每句 ≤15 字）",
  "l3": "……（每句 ≤12 字）"
}
```
- `text` 為空字串 → `422`；純空白 → `400`。

## 測試

```bash
# 免模型（路由 + 純文字邏輯，CI 可跑）
python tests/test_pipeline_logic.py      # 或 pytest tests/test_pipeline_logic.py
python tests/test_api.py                 # 或 pytest tests/test_api.py

# 需 GPU（端到端載入真模型）
python smoke_test.py
```

測試分兩層：免模型的單元/路由測試（注入 stub pipeline）可在無 GPU 環境跑；
`smoke_test.py` 則載入真實模型驗證端到端。

## 設計重點

- **模型常駐**：lifespan 啟動載一次，避免每次請求重載（8B 載入需數十秒）。
- **單 GPU 序列化**：API 與 Web UI 兩條路徑共用同一把 `threading.Lock`，
  確保同一時間只有一個推論在 GPU 上執行。
- **不阻塞事件迴圈**：同步推論透過 `run_in_executor` 在背景執行緒執行。
- **可測試性**：推論狀態收進 `SimplificationPipeline`，測試以 stub 注入即可
  在無 GPU 環境驗證服務邏輯。
