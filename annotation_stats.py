# annotation_stats.py
import json, re
import numpy as np

DATA_PATH = r"F:\download\專題\完成標註\training_data_v5.jsonl"

samples = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line.strip()))

def avg_sent_len(text):
    sents = [s.strip() for s in re.split(r'[。！？\n]', text) if s.strip()]
    if not sents: return 0
    lens = [len(re.sub(r'[^\u4e00-\u9fff]', '', s)) for s in sents]
    return np.mean(lens)

def ttr(text):
    chars = re.sub(r'[^\u4e00-\u9fff]', '', text)
    if not chars: return 0
    return len(set(chars)) / len(chars)

def total_chars(text):
    return len(re.sub(r'[^\u4e00-\u9fff]', '', text))

print(f"{'層級':<6} {'平均句長':>8} {'平均總字數':>10} {'TTR':>8} {'n':>4}")
print("-" * 45)

# 原文
srcs = [s["input"] for s in samples if s["metadata"]["level"] == 1]
print(f"{'原文':<6} {np.mean([avg_sent_len(t) for t in srcs]):>8.1f} "
      f"{np.mean([total_chars(t) for t in srcs]):>10.1f} "
      f"{np.mean([ttr(t) for t in srcs]):>8.4f} {len(srcs):>4}")

for lv in [1, 2, 3]:
    lvs = [s for s in samples if s["metadata"]["level"] == lv]
    texts = [s["output"] for s in lvs]
    print(f"{'L'+str(lv):<6} {np.mean([avg_sent_len(t) for t in texts]):>8.1f} "
          f"{np.mean([total_chars(t) for t in texts]):>10.1f} "
          f"{np.mean([ttr(t) for t in texts]):>8.4f} {len(texts):>4}")