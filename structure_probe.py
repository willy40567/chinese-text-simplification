# structure_probe.py
# 放置於 F:\download\專題\

import json
from pathlib import Path

def probe_structure(annotation_dir):
    path = Path(annotation_dir)
    json_files = list(path.glob("*.json"))
    
    for json_file in sorted(json_files)[:2]:  # 只看前兩個檔
        print(f"\n{'='*50}")
        print(f"檔案：{json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"頂層型別：{type(data).__name__}")
        
        if isinstance(data, list):
            print(f"元素數量：{len(data)}")
            print(f"第一個元素型別：{type(data[0]).__name__}")
            print(f"第一個元素內容（前200字）：")
            print(str(data[0])[:200])
            
        elif isinstance(data, dict):
            print(f"頂層鍵值：{list(data.keys())}")
            for k, v in data.items():
                print(f"  [{k}] 型別：{type(v).__name__}，內容（前100字）：{str(v)[:100]}")

if __name__ == "__main__":
    probe_structure(r"F:\download\專題\完成標註")