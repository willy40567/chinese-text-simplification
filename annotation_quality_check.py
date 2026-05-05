# annotation_quality_check.py
import json, re
import numpy as np
from bert_score import score as bert_score

DATA_PATH = r"F:\download\專題\完成標註\training_data_v5.jsonl"
LIMIT_MAP = {1: 20, 2: 15, 3: 12}

samples = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line.strip()))

def compliance_rate(text, limit):
    sents = [s.strip() for s in re.split(r'[。！？\n]', text) if s.strip()]
    if not sents: return 0.0
    ok = sum(1 for s in sents if len(re.sub(r'[^\u4e00-\u9fff\w]','',s)) <= limit)
    return ok / len(sents)

# 1. 合規率
for lv in [1,2,3]:
    lvs = [s for s in samples if s["metadata"]["level"] == lv]
    rates = [compliance_rate(s["output"], LIMIT_MAP[lv]) for s in lvs]
    print(f"L{lv} 標註合規率: {np.mean(rates):.1%}  (n={len(lvs)})")

# 2. BERTScore（標註 vs 原文）
for lv in [1,2,3]:
    lvs = [s for s in samples if s["metadata"]["level"] == lv]
    refs  = [s["input"]  for s in lvs]
    hyps  = [s["output"] for s in lvs]
    _, _, F1 = bert_score(hyps, refs, lang="zh",
                          model_type="bert-base-chinese", verbose=False)
    print(f"L{lv} BERTScore(標註↔原文): {F1.mean().item():.4f}")
