# convert_to_training_v4.py
# 放置於 F:\download\專題\
# 變更紀錄 v4：
#   - 新增 check_compliance() 合規驗證（skip_non_compliant 參數）
#   - 標題行過濾：十一、粒粒皆辛苦——牡蠣養殖 等章節標題不計入句長
#   - 書目標籤行過濾：書名/作者/譯者/出版社/大意/心得感想 不計入句長

import json
import re
from pathlib import Path
from datetime import datetime

from config import LEVEL_LIMITS as LIMITS, LEVEL_INSTRUCTION_TRAINING as LEVEL_INSTRUCTION

# 標題行 pattern（如 十一、粒粒皆辛苦——牡蠣養殖）
TITLE_PATTERN = re.compile(r'^[零一二三四五六七八九十百千]+[、．]')
# 書目標籤 pattern（如 書名：大象艾瑪）
LABEL_PATTERN = re.compile(r'^(書名|作者|譯者|出版社|大意|心得感想|心得|標題)：')


def check_compliance(text, level):
    """
    回傳 (is_compliant, violations_list)
    自動過濾：章節標題行、書目標籤行
    """
    limit = LIMITS[level]
    violations = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 過濾標題行
        if TITLE_PATTERN.match(line):
            continue
        # 按句號/問號/驚嘆號分句
        sentences = [s.strip() for s in re.split(r'[。！？]', line) if s.strip()]
        for sent in sentences:
            # 過濾書目標籤行
            if LABEL_PATTERN.match(sent):
                continue
            if len(sent) > limit:
                violations.append(sent)

    return len(violations) == 0, violations


def build_training_samples(json_path, include_levels=(1, 2, 3), skip_non_compliant=True):
    """將單一 JSON 標註檔轉換為訓練樣本列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lesson_id = data.get('lesson_id', json_path.stem)
    source_file = data.get('source_file', '')
    original_text = data.get('original_text', '').strip()
    simplified = data.get('simplified_versions', {})

    samples = []
    skipped = []

    # L1：以原文為輸入
    if 1 in include_levels:
        level_data = simplified.get('level1', {})
        text = level_data.get('text', '') if isinstance(level_data, dict) else level_data
        if original_text and text:
            is_ok, violations = check_compliance(text, 1)
            if skip_non_compliant and not is_ok:
                skipped.append(f"[SKIP] {lesson_id} L1 超標句：{violations}")
            else:
                samples.append({
                    "instruction": LEVEL_INSTRUCTION[1],
                    "input": original_text,
                    "output": text.strip(),
                    "metadata": {
                        "lesson_id": lesson_id,
                        "source_file": source_file,
                        "level": 1,
                        "limit": LIMITS[1]
                    }
                })

    # L2：以 L1 輸出為輸入
    if 2 in include_levels:
        l1_data = simplified.get('level1', {})
        l1_text = l1_data.get('text', '') if isinstance(l1_data, dict) else l1_data
        l2_data = simplified.get('level2', {})
        l2_text = l2_data.get('text', '') if isinstance(l2_data, dict) else l2_data
        if l1_text and l2_text:
            is_ok, violations = check_compliance(l2_text, 2)
            if skip_non_compliant and not is_ok:
                skipped.append(f"[SKIP] {lesson_id} L2 超標句：{violations}")
            else:
                samples.append({
                    "instruction": LEVEL_INSTRUCTION[2],
                    "input": l1_text.strip(),
                    "output": l2_text.strip(),
                    "metadata": {
                        "lesson_id": lesson_id,
                        "source_file": source_file,
                        "level": 2,
                        "limit": LIMITS[2]
                    }
                })

    # L3：以 L2 輸出為輸入
    if 3 in include_levels:
        l2_data = simplified.get('level2', {})
        l2_text = l2_data.get('text', '') if isinstance(l2_data, dict) else l2_data
        l3_data = simplified.get('level3', {})
        l3_text = l3_data.get('text', '') if isinstance(l3_data, dict) else l3_data
        if l2_text and l3_text:
            is_ok, violations = check_compliance(l3_text, 3)
            if skip_non_compliant and not is_ok:
                skipped.append(f"[SKIP] {lesson_id} L3 超標句：{violations}")
            else:
                samples.append({
                    "instruction": LEVEL_INSTRUCTION[3],
                    "input": l2_text.strip(),
                    "output": l3_text.strip(),
                    "metadata": {
                        "lesson_id": lesson_id,
                        "source_file": source_file,
                        "level": 3,
                        "limit": LIMITS[3]
                    }
                })

    return samples, skipped


def convert_all(annotation_dir, output_path, include_levels=(1, 2, 3), skip_non_compliant=True):
    path = Path(annotation_dir)
    files = sorted(path.glob("*.json"))

    all_samples = []
    all_skipped = []
    file_report = []

    for f in files:
        samples, skipped = build_training_samples(f, include_levels, skip_non_compliant)
        all_samples.extend(samples)
        all_skipped.extend(skipped)
        file_report.append(f"  {f.name}：{len(samples)} 筆")

    # 輸出 SKIP 訊息
    for msg in all_skipped:
        print(msg)

    # 寫出 JSONL
    out_path = Path(output_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 報告
    print(f"{'='*60}")
    print(f"資料轉換報告 v4｜{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    for line in file_report:
        print(line)
    print(f"{'='*60}")
    print(f"輸出路徑：{out_path}")
    print(f"總樣本數：{len(all_samples)}")
    level_counts = {}
    for s in all_samples:
        lv = s['metadata']['level']
        level_counts[lv] = level_counts.get(lv, 0) + 1
    for lv in sorted(level_counts):
        print(f"  Level {lv}：{level_counts[lv]} 筆")
    print(f"跳過（不合規）：{len(all_skipped)} 筆")
    print(f"{'='*60}")

    return len(all_samples)


if __name__ == "__main__":
    import os
    from config import PROJECT_ROOT
    convert_all(
        annotation_dir=os.path.join(PROJECT_ROOT, "完成標註"),
        output_path=os.path.join(PROJECT_ROOT, "完成標註", "training_data_v4.jsonl"),
        include_levels=(1, 2, 3),
        skip_non_compliant=True
    )