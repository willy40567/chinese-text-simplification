"""
batch_processor.py - 批次自動化處理模組
自動遍歷目錄內所有txt檔案，生成標註模板
"""
import os
import json
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_analyzer import analyze_text, create_annotation_template, LEVEL_STANDARDS

def extract_lessons_from_txt(filepath):
    """
    從課本txt提取課文段落
    Returns: list of dict {"lesson_name": str, "paragraphs": [str]}
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='big5') as f:
            content = f.read()
    
    lessons = []
    # 嘗試按課文標題切分（一、二、三...或數字開頭）
    pattern = r'((?:第?[一二三四五六七八九十百]+[課、]|[一二三四五六七八九十]+[、.．])[^\n]+)'
    titles = list(re.finditer(pattern, content))
    
    if titles:
        for i, match in enumerate(titles):
            title = match.group().strip()
            start = match.end()
            end = titles[i+1].start() if i+1 < len(titles) else len(content)
            lesson_content = content[start:end].strip()
            
            # 切分段落（按換行或句群）
            paragraphs = [p.strip() for p in lesson_content.split('\n') if p.strip() and len(p.strip()) > 5]
            
            if paragraphs:
                lessons.append({
                    "lesson_name": title,
                    "paragraphs": paragraphs
                })
    else:
        # 無法識別課文結構，按段落分割
        paragraphs = [p.strip() for p in content.split('\n') if p.strip() and len(p.strip()) > 5]
        filename = os.path.basename(filepath).replace('.txt', '')
        lessons.append({
            "lesson_name": filename,
            "paragraphs": paragraphs
        })
    
    return lessons

def process_directory(input_dir, output_dir):
    """
    批次處理目錄內所有txt檔案
    
    Args:
        input_dir: 課本txt所在目錄
        output_dir: 輸出模板的目錄
    
    Returns:
        dict: 統計報告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        "processed_files": 0,
        "total_lessons": 0,
        "total_paragraphs": 0,
        "compliance_summary": {"level1": 0, "level2": 0, "level3": 0},
        "files": []
    }
    
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    print(f"\n{'='*50}")
    print(f"批次處理開始 | 找到 {len(txt_files)} 個txt檔案")
    print(f"{'='*50}\n")
    
    for filename in sorted(txt_files):
        filepath = os.path.join(input_dir, filename)
        print(f"處理：{filename}...")
        
        try:
            lessons = extract_lessons_from_txt(filepath)
            file_stats = {"filename": filename, "lessons": [], "paragraph_count": 0}
            
            for lesson in lessons:
                lesson_templates = []
                
                for i, para in enumerate(lesson["paragraphs"]):
                    if len(para) < 10:  # 跳過太短的段落
                        continue
                    
                    lesson_id = f"{lesson['lesson_name']}-p{i+1:02d}"
                    template = create_annotation_template(
                        lesson_id=lesson_id,
                        original_text=para,
                        source_file=filename
                    )
                    lesson_templates.append(template)
                    stats["total_paragraphs"] += 1
                    file_stats["paragraph_count"] += 1
                    
                    # 統計原文合規率
                    analysis = template["original_analysis"]
                    if analysis:
                        for level, ok in analysis["compliance"].items():
                            if ok:
                                stats["compliance_summary"][level] += 1
                
                if lesson_templates:
                    # 儲存課文模板
                    safe_name = re.sub(r'[\\/:*?"<>|]', '_', lesson["lesson_name"])
                    out_filename = f"{os.path.splitext(filename)[0]}_{safe_name}_template.json"
                    out_path = os.path.join(output_dir, out_filename)
                    
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "source_file": filename,
                            "lesson_name": lesson["lesson_name"],
                            "paragraph_count": len(lesson_templates),
                            "templates": lesson_templates
                        }, f, ensure_ascii=False, indent=2)
                    
                    file_stats["lessons"].append({
                        "name": lesson["lesson_name"],
                        "paragraphs": len(lesson_templates),
                        "output": out_filename
                    })
                    stats["total_lessons"] += 1
            
            stats["files"].append(file_stats)
            stats["processed_files"] += 1
            print(f"  ✅ {len(lessons)} 課，{file_stats['paragraph_count']} 段落")
        
        except Exception as e:
            print(f"  ❌ 錯誤：{e}")
    
    # 生成統計報告
    report_path = os.path.join(output_dir, f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    _write_report(stats, report_path)
    
    print(f"\n{'='*50}")
    print(f"批次處理完成！")
    print(f"處理檔案：{stats['processed_files']}/{len(txt_files)}")
    print(f"總課數：{stats['total_lessons']}")
    print(f"總段落：{stats['total_paragraphs']}")
    print(f"報告：{report_path}")
    print(f"{'='*50}\n")
    
    return stats

def _write_report(stats, report_path):
    total = stats["total_paragraphs"]
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"批次處理報告\n")
        f.write(f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"總覽：\n")
        f.write(f"  處理檔案：{stats['processed_files']}\n")
        f.write(f"  課文數量：{stats['total_lessons']}\n")
        f.write(f"  段落數量：{total}\n\n")
        
        f.write(f"原文合規率（不需簡化即合規的比例）：\n")
        for level, count in stats["compliance_summary"].items():
            rate = count / total * 100 if total > 0 else 0
            f.write(f"  {level.upper()}(≤{20 if level=='level1' else 15 if level=='level2' else 12}字)：{rate:.1f}% ({count}/{total})\n")
        
        f.write("\n各檔案詳情：\n")
        for file_info in stats["files"]:
            f.write(f"\n  {file_info['filename']}：{file_info['paragraph_count']} 段落\n")
            for lesson in file_info["lessons"]:
                f.write(f"    └─ {lesson['name']}：{lesson['paragraphs']} 段\n")


if __name__ == "__main__":
    # 預設路徑
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "F:/download/專題/課本txt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "F:/download/專題/標註模板"
    
    if not os.path.exists(input_dir):
        print(f"❌ 找不到輸入目錄：{input_dir}")
        print("用法：python batch_processor.py [輸入目錄] [輸出目錄]")
        sys.exit(1)
    
    process_directory(input_dir, output_dir)
