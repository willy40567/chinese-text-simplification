# quality_checker_v2.py
# 放置於 F:\download\專題\

import json
import re
from pathlib import Path

LIMITS = {1: 20, 2: 15, 3: 12}

def check_text(text, level):
    """分句並檢查字數上限"""
    limit = LIMITS[level]
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if s.strip() and s.strip() != '—']
    
    violations = []
    for sent in sentences:
        # 移除標點符號後計算純漢字+英數字長度
        clean = re.sub(r'[^\u4e00-\u9fff\w]', '', sent)
        count = len(clean)
        if count > limit:
            violations.append((sent, count))
    return violations

def check_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lesson_id = data.get('lesson_id', json_path.stem)
    simplified = data.get('simplified_versions', {})
    
    issues = []
    for level in [1, 2, 3]:
        key = f'level{level}'
        level_data = simplified.get(key, {})
        
        # 相容 text 在 dict 內或直接是字串
        if isinstance(level_data, dict):
            text = level_data.get('text', '')
        elif isinstance(level_data, str):
            text = level_data
        else:
            text = ''
        
        if not text:
            issues.append(f"  ⚠️  L{level} 文本為空")
            continue
        
        violations = check_text(text, level)
        for sent, count in violations:
            issues.append(
                f"  ❌ L{level} 超標 {count}字（上限{LIMITS[level]}）：「{sent[:20]}{'...' if len(sent)>20 else ''}」"
            )
    
    return lesson_id, issues

def run_check(annotation_dir):
    path = Path(annotation_dir)
    files = sorted(path.glob("*.json"))
    
    print(f"{'='*60}")
    print(f"標註品質掃描報告 v2")
    print(f"掃描檔案數：{len(files)}")
    print(f"{'='*60}\n")
    
    total_violations = 0
    clean_count = 0
    
    for f in files:
        lesson_id, issues = check_file(f)
        if issues:
            print(f"📄 {f.name}")
            print(f"   lesson_id: {lesson_id}")
            for issue in issues:
                print(issue)
            total_violations += len(issues)
            print()
        else:
            print(f"✅ {f.name}")
            clean_count += 1
    
    print(f"\n{'='*60}")
    print(f"掃描完成")
    print(f"  合規檔案：{clean_count}/{len(files)}")
    print(f"  總違規數：{total_violations}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_check(r"F:\download\專題\完成標註")