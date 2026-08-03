#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py FastAPI 路由測試（注入 stub pipeline，不需載入模型／GPU）。

跑法：
  pytest tests/test_api.py
  python tests/test_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import app as app_module


class StubPipeline:
    """假管線：行為可預測，不載入任何模型。"""
    ready = True

    def simplify_text(self, text):
        return {"l1": text + "-L1", "l2": text + "-L2", "l3": text + "-L3"}


def make_client():
    # 先注入 stub，lifespan 偵測到已存在即略過真實載入
    app_module.app.state.pipeline = StubPipeline()
    return TestClient(app_module.app)


def test_health_ready():
    client = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ready"] is True


def test_simplify_ok():
    client = make_client()
    r = client.post("/simplify", json={"text": "測試文本"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"l1": "測試文本-L1", "l2": "測試文本-L2", "l3": "測試文本-L3"}


def test_simplify_empty_string_422():
    client = make_client()
    r = client.post("/simplify", json={"text": ""})
    assert r.status_code == 422  # pydantic min_length=1


def test_simplify_whitespace_400():
    client = make_client()
    r = client.post("/simplify", json={"text": "   "})
    assert r.status_code == 400  # 顯式空白檢查


def test_simplify_missing_field_422():
    client = make_client()
    r = client.post("/simplify", json={})
    assert r.status_code == 422


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
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
