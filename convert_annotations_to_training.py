#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能分割標註資料轉換腳本 v2.0
功能：將多課合併JSON檔案智能分割為單課訓練樣本
作者：閱讀障礙文本簡化研究專案
日期：2026-02-25
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

class MultiLessonSplitter:
    """多課JSON智能分割器"""
    
    # 課文標題模式（一、二、三、四、五...）
    LESSON_PATTERN = r'([一二三四五六七八九十]+)、([^\n]+)'
    
    def __init__(self, input_dir: str, output_file: str):
        """
        初始化分割器
        
        Args:
            input_dir: 標註JSON檔案所在目錄
            output_file: 輸出的訓練資料檔案路徑
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.training_samples = []
    
    def split_by_lesson_markers(self, text: str) -> List[Tuple[str, str]]:
        """
        按課文標題分割文本
        
        Args:
            text: 合併的多課文本
            
        Returns:
            [(lesson_title, lesson_content), ...]
        """
        # 找到所有課文標題位置
        matches = list(re.finditer(self.LESSON_PATTERN, text))
        
        if not matches:
            # 無課文標記，返回整體文本
            return [("未知課文", text)]
        
        lessons = []
        
        for i, match in enumerate(matches):
            # 提取課文標題
            lesson_num = match.group(1)
            lesson_name = match.group(2)
            lesson_title = f"{lesson_num}、{lesson_name}"
            
            # 確定課文內容範圍
            start_pos = match.start()
            
            # 下一課的起始位置（或文本結尾）
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            # 提取課文內容
            lesson_content = text[start_pos:end_pos].strip()
            
            lessons.append((lesson_title, lesson_content))
        
        return lessons
    
    def align_original_and_simplified(self, 
                                      original_lessons: List[Tuple[str, str]], 
                                      simplified_lessons: List[Tuple[str, str]]) -> List[Dict]:
        """
        對齊原文與簡化版本
        
        Args:
            original_lessons: 原文分割結果
            simplified_lessons: 簡化版分割結果
            
        Returns:
            對齊後的訓練樣本列表
        """
        aligned_samples = []
        
        # 取較短的列表長度（防止索引越界）
        min_length = min(len(original_lessons), len(simplified_lessons))
        
        for i in range(min_length):
            orig_title, orig_content = original_lessons[i]
            simp_title, simp_content = simplified_lessons[i]
            
            # 驗證標題一致性
            if orig_title != simp_title:
                print(f"⚠️  警告：標題不一致 - 原文:{orig_title} vs 簡化:{simp_title}")
            
            aligned_samples.append({
                'lesson_title': orig_title,
                'original': orig_content,
                'simplified': simp_content
            })
        
        return aligned_samples
    
    def process_json_file(self, json_path: Path) -> List[Dict]:
        """
        處理單一JSON檔案，提取多個訓練樣本
        
        Args:
            json_path: JSON檔案路徑
            
        Returns:
            訓練樣本列表
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取原文與Level 1簡化版本
        original_text = data.get('original_text', '')
        
        simplified_versions = data.get('simplified_versions', {})
        level1_data = simplified_versions.get('level1', {})
        
        # 提取Level 1文本
        if isinstance(level1_data, dict):
            simplified_text = level1_data.get('text', '')
        elif isinstance(level1_data, str):
            simplified_text = level1_data
        else:
            print(f"❌ 錯誤：{json_path.name} 的level1格式異常")
            return []
        
        # 驗證數據完整性
        if not original_text or not simplified_text:
            print(f"❌ 錯誤：{json_path.name} 缺少必要欄位")
            return []
        
        # 分割原文與簡化版本
        original_lessons = self.split_by_lesson_markers(original_text)
        simplified_lessons = self.split_by_lesson_markers(simplified_text)
        
        # 對齊並生成訓練樣本
        aligned = self.align_original_and_simplified(
            original_lessons, 
            simplified_lessons
        )
        
        # 構建訓練樣本
        samples = []
        for item in aligned:
            sample = {
                'input': item['original'].strip(),
                'output': item['simplified'].strip(),
                'metadata': {
                    'source_file': json_path.name,
                    'lesson_title': item['lesson_title'],
                    'lesson_id': data.get('lesson_id', '')
                }
            }
            samples.append(sample)
        
        return samples
    
    def convert_all(self) -> int:
        """
        轉換所有JSON檔案
        
        Returns:
            成功轉換的樣本數量
        """
        json_files = list(self.input_dir.glob("*.json"))
        
        # 過濾掉checkpoint等非標註檔案
        json_files = [
            f for f in json_files 
            if 'checkpoint' not in f.name.lower() and
               'adapter_config' not in f.name.lower() and
               'trainer_state' not in f.name.lower() and
               'tokenizer' not in f.name.lower()
        ]
        
        print(f"🔍 發現 {len(json_files)} 個標註JSON檔案")
        
        if not json_files:
            print("❌ 未發現任何標註JSON檔案")
            return 0
        
        print("\n開始轉換...")
        print("=" * 60)
        
        total_samples = 0
        
        for json_file in sorted(json_files):
            try:
                samples = self.process_json_file(json_file)
                
                if samples:
                    self.training_samples.extend(samples)
                    total_samples += len(samples)
                    
                    print(f"✅ {json_file.name}")
                    print(f"   提取 {len(samples)} 筆訓練樣本")
                    
                    # 顯示課文標題
                    for sample in samples:
                        print(f"      - {sample['metadata']['lesson_title']}")
                    
            except Exception as e:
                print(f"❌ {json_file.name}: {str(e)}")
        
        print("=" * 60)
        print(f"\n📊 轉換完成：{total_samples} 筆訓練樣本（來自 {len(json_files)} 個JSON檔案）")
        
        return total_samples
    
    def save_training_data(self):
        """儲存為JSONL格式"""
        if not self.training_samples:
            print("⚠️  沒有可儲存的訓練資料")
            return
        
        # 確保輸出目錄存在
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 寫入JSONL格式
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for sample in self.training_samples:
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"\n💾 訓練資料已儲存：{self.output_file}")
        print(f"   檔案大小：{self.output_file.stat().st_size / 1024:.2f} KB")
    
    def generate_report(self):
        """生成轉換報告"""
        if not self.training_samples:
            return
        
        total_samples = len(self.training_samples)
        
        # 計算平均長度
        input_lengths = [len(s['input']) for s in self.training_samples]
        output_lengths = [len(s['output']) for s in self.training_samples]
        
        avg_input = sum(input_lengths) / total_samples
        avg_output = sum(output_lengths) / total_samples
        compression_rate = avg_output / avg_input if avg_input > 0 else 0
        
        # 按來源檔案分組統計
        source_counts = {}
        for sample in self.training_samples:
            source = sample['metadata']['source_file']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # 按課文標題統計
        lesson_titles = {}
        for sample in self.training_samples:
            title = sample['metadata']['lesson_title']
            lesson_titles[title] = lesson_titles.get(title, 0) + 1
        
        # 生成報告
        report = f"""
{'=' * 60}
訓練資料轉換報告 v2.0（智能分割版）
{'=' * 60}

基本資訊：
  訓練樣本數：{total_samples}
  輸出檔案：{self.output_file.name}

長度統計：
  原文平均長度：{avg_input:.1f} 字符
  簡化版平均長度：{avg_output:.1f} 字符
  壓縮率：{compression_rate*100:.1f}%

來源檔案分布：
"""
        for source, count in sorted(source_counts.items()):
            report += f"  {source}: {count} 筆\n"
        
        report += f"\n課文標題列表：\n"
        for title, count in sorted(lesson_titles.items()):
            report += f"  {title}: {count} 次\n"
        
        report += f"\n{'=' * 60}\n"
        
        print(report)
        
        # 儲存報告
        report_file = self.output_file.parent / 'conversion_report_v2.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 報告已儲存：{report_file}")


def main():
    """主程式"""
    print("""
╔══════════════════════════════════════════════════════════╗
║    智能分割標註資料轉換工具 v2.0                         ║
║    閱讀障礙文本簡化研究專案                              ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 設定路徑
    input_dir = r"F:\download\專題\完成標註"
    output_file = r"F:\download\專題\完成標註\training_data_v2.jsonl"
    
    print(f"輸入目錄：{input_dir}")
    print(f"輸出檔案：{output_file}")
    print(f"\n功能說明：")
    print(f"  - 智能識別課文標題標記（一、二、三...）")
    print(f"  - 自動分割多課合併JSON")
    print(f"  - 對齊原文與簡化版本")
    print(f"  - 生成單課訓練樣本\n")
    
    # 執行轉換
    splitter = MultiLessonSplitter(input_dir, output_file)
    
    # 轉換所有檔案
    success_count = splitter.convert_all()
    
    if success_count > 0:
        # 儲存訓練資料
        splitter.save_training_data()
        
        # 生成報告
        splitter.generate_report()
        
        print("\n✅ 智能分割轉換作業完成！")
        print(f"\n下一步：使用 training_data_v2.jsonl 進行PEFT微調訓練")
        print(f"預期樣本數：24筆（8個JSON × 3課/JSON）")
    else:
        print("\n❌ 轉換失敗，請檢查檔案路徑與格式")


if __name__ == '__main__':
    main()