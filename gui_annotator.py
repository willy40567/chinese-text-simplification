"""
gui_annotator.py - 圖形化標註介面 (現代化 UI 版)
中文文本簡化標註工具 v2.0
依賴：ttkbootstrap, core_analyzer.py (同目錄)
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.scrolledtext as st
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import json
import os
import sys
from datetime import datetime

# 確保可以匯入同目錄的 core_analyzer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_analyzer import (
    analyze_text, suggest_splits, validate_simplified,
    create_annotation_template, LEVEL_STANDARDS, count_chars
)

class AnnotationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📝 中文文本簡化標註工具 v2.0")
        self.root.geometry("1400x850")
        
        # 狀態變數
        self.current_file = None
        self.current_lesson = ttk.StringVar(value="")
        self.current_original = ttk.StringVar(value="")
        self.annotations = []  # 已完成的標註列表
        self.current_template = None
        
        # 🤖 設定你的 LoRA 模型路徑
        self.model_path = r"F:\download\專題\lora_weights\full_train\final_lora\\"
        
        # 建立介面
        self._build_ui()
        
        # 顯示歡迎訊息
        self._log("🚀 標註工具啟動完成。請載入 txt 課本檔案開始標註。")
        self._log(f"標準：Level1≤20字 | Level2≤15字 | Level3≤12字")
        self._log(f"🧠 已設定 LoRA 模型路徑：{self.model_path}")
    
    # ==========================================
    # UI 建構 (現代化設計)
    # ==========================================
    def _build_ui(self):
        # 頂部工具列
        self._build_toolbar()
        
        # 主內容區域（三欄）
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=BOTH, expand=YES)
        
        # 左欄：原文分析
        self._build_left_panel(main_frame)
        
        # 中欄：簡化輸入
        self._build_center_panel(main_frame)
        
        # 右欄：日誌與統計
        self._build_right_panel(main_frame)
        
        # 底部狀態列
        self._build_statusbar()
    
    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, bootstyle=DARK, padding=10)
        toolbar.pack(fill=X)
        
        title = ttk.Label(toolbar, text="📝 中文文本簡化標註工具", 
                         font=("Microsoft JhengHei", 16, "bold"), bootstyle="inverse-dark")
        title.pack(side=LEFT, padx=(10, 20))
        
        # 工具按鈕 (使用 ttkbootstrap 樣式)
        ttk.Button(toolbar, text="📂 載入txt", bootstyle=PRIMARY,
                  command=self._load_file).pack(side=LEFT, padx=5)
        
        ttk.Button(toolbar, text="💾 儲存JSON", bootstyle=SUCCESS,
                  command=self._save_annotation).pack(side=LEFT, padx=5)
        
        ttk.Button(toolbar, text="📋 儲存全部", bootstyle=INFO,
                  command=self._save_all).pack(side=LEFT, padx=5)
        
        ttk.Button(toolbar, text="🔄 清除", bootstyle=SECONDARY,
                  command=self._clear_form).pack(side=LEFT, padx=5)
        
        # 計數器
        self.count_label = ttk.Label(toolbar, text="已完成：0 組", 
                                    font=("Microsoft JhengHei", 12, "bold"), bootstyle="inverse-dark")
        self.count_label.pack(side=RIGHT, padx=15)
    
    def _build_left_panel(self, parent):
        frame = ttk.Labelframe(parent, text="📄 原文分析", bootstyle=INFO, padding=15)
        frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 5))
        
        # 課文名稱
        ttk.Label(frame, text="課文名稱：", font=("Microsoft JhengHei", 10, "bold")).pack(anchor=W, pady=(0, 5))
        ttk.Entry(frame, textvariable=self.current_lesson, font=("Microsoft JhengHei", 10)).pack(fill=X, pady=(0, 15))
        
        # 原文輸入
        ttk.Label(frame, text="原文（貼上後點擊分析）：", font=("Microsoft JhengHei", 10, "bold")).pack(anchor=W, pady=(0, 5))
        self.original_text = st.ScrolledText(frame, height=6, font=("Microsoft JhengHei", 11), wrap=WORD)
        self.original_text.pack(fill=X, pady=(0, 10))
        
        # 分析與 AI 簡化按鈕區域
        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=X, pady=(0, 15))
        
        ttk.Button(action_frame, text="🔍 分析原文", bootstyle=PRIMARY,
                  command=self._analyze_original).pack(side=LEFT, expand=YES, fill=X, padx=(0, 5))
                 
        ttk.Button(action_frame, text="🤖 AI 自動簡化 (LoRA)", bootstyle=(WARNING, OUTLINE),
                  command=self._ai_generate).pack(side=LEFT, expand=YES, fill=X, padx=(5, 0))
        
        # 分析結果顯示
        ttk.Label(frame, text="📊 分析結果：", font=("Microsoft JhengHei", 10, "bold")).pack(anchor=W, pady=(0, 5))
        self.analysis_result = st.ScrolledText(frame, height=12, font=("Consolas", 10), wrap=WORD, state=DISABLED)
        self.analysis_result.pack(fill=BOTH, expand=YES, pady=(0, 10))
        
        # 切分建議按鈕
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=X)
        btn_styles = {"level1": PRIMARY, "level2": INFO, "level3": WARNING}
        
        for level, std in LEVEL_STANDARDS.items():
            max_c = std["max_chars"]
            ttk.Button(btn_frame, text=f"💡 {level.upper()}建議(≤{max_c}字)", bootstyle=(btn_styles[level], OUTLINE),
                      command=lambda l=level, m=max_c: self._show_split_suggestion(l, m)
                      ).pack(side=LEFT, expand=YES, fill=X, padx=2)
    
    def _build_center_panel(self, parent):
        frame = ttk.Labelframe(parent, text="✏️ 簡化版本輸入", bootstyle=SUCCESS, padding=15)
        frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=5)
        
        self.level_texts = {}
        self.level_validations = {}
        
        levels_info = [
            ("level1", "Level 1 ≤20字（基礎版）", PRIMARY),
            ("level2", "Level 2 ≤15字（進階版）", INFO),
            ("level3", "Level 3 ≤12字（極簡版）", WARNING)
        ]
        
        for level, desc, style in levels_info:
            # 標題與驗證標籤
            header_frame = ttk.Frame(frame)
            header_frame.pack(fill=X, pady=(10, 5) if level != "level1" else (0, 5))
            
            ttk.Label(header_frame, text=desc, font=("Microsoft JhengHei", 11, "bold"), bootstyle=style).pack(side=LEFT)
            
            val_label = ttk.Label(header_frame, text="⏳ 待輸入", font=("Microsoft JhengHei", 9), bootstyle=SECONDARY)
            val_label.pack(side=RIGHT)
            self.level_validations[level] = val_label
            
            # 文字輸入框
            txt = st.ScrolledText(frame, height=5, font=("Microsoft JhengHei", 11), wrap=WORD)
            txt.pack(fill=X, pady=(0, 5))
            self.level_texts[level] = txt
            
            # 操作按鈕
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(fill=X)
            ttk.Button(btn_frame, text=f"✓ 驗證 {level.upper()}", bootstyle=(style, OUTLINE),
                      command=lambda l=level: self._validate_level(l)).pack(side=LEFT, padx=(0, 5))
            ttk.Button(btn_frame, text="📋 貼上建議", bootstyle=(SECONDARY, OUTLINE),
                      command=lambda l=level: self._apply_suggestion(l)).pack(side=LEFT)
        
        # 儲存按鈕
        ttk.Button(frame, text="💾 完成並儲存此組標註", bootstyle=SUCCESS, padding=10,
                  command=self._save_annotation).pack(fill=X, pady=(20, 0))
    
    def _build_right_panel(self, parent):
        frame = ttk.Labelframe(parent, text="📊 日誌與統計", bootstyle=SECONDARY, padding=15)
        frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(5, 0))
        
        # 進度統計卡片
        stats_frame = ttk.Frame(frame)
        stats_frame.pack(fill=X, pady=(0, 15))
        
        ttk.Label(stats_frame, text="📈 標註進度", font=("Microsoft JhengHei", 11, "bold")).pack(anchor=W, pady=(0, 5))
        
        self.progress_labels = {}
        for item in ["已完成組數", "Level1完成", "Level2完成", "Level3完成"]:
            row = ttk.Frame(stats_frame)
            row.pack(fill=X, pady=2)
            ttk.Label(row, text=f"{item}：", font=("Microsoft JhengHei", 10), width=12).pack(side=LEFT)
            lbl = ttk.Label(row, text="0", font=("Consolas", 11, "bold"), bootstyle=PRIMARY)
            lbl.pack(side=LEFT)
            self.progress_labels[item] = lbl
        
        ttk.Separator(frame, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # 操作日誌
        ttk.Label(frame, text="📋 操作日誌：", font=("Microsoft JhengHei", 10, "bold")).pack(anchor=W, pady=(0, 5))
        self.log_box = st.ScrolledText(frame, height=15, font=("Consolas", 9), wrap=WORD, state=DISABLED)
        self.log_box.pack(fill=BOTH, expand=YES, pady=(0, 15))
        
        # 切分建議顯示區
        ttk.Label(frame, text="💡 切分建議（可複製）：", font=("Microsoft JhengHei", 10, "bold")).pack(anchor=W, pady=(0, 5))
        self.suggestion_box = st.ScrolledText(frame, height=8, font=("Microsoft JhengHei", 10), wrap=WORD)
        self.suggestion_box.pack(fill=BOTH)
    
    def _build_statusbar(self):
        status_frame = ttk.Frame(self.root, bootstyle=SECONDARY, padding=5)
        status_frame.pack(fill=X, side=BOTTOM)
        
        self.status_var = ttk.StringVar(value="就緒")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Microsoft JhengHei", 9), bootstyle="inverse-secondary").pack(side=LEFT, padx=10)

    # ==========================================
    # 以下保留原本的核心邏輯，完全不變！
    # ==========================================
    
    def _load_file(self):
        path = filedialog.askopenfilename(
            title="選擇課本txt檔案",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir="F:/download/專題/課本txt"
        )
        if path:
            self.current_file = path
            filename = os.path.basename(path)
            self.status_var.set(f"已載入：{filename}")
            self._log(f"📂 載入檔案：{filename}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(500)
                self._log(f"檔案預覽（前500字）：\n{content}")
            except UnicodeDecodeError:
                try:
                    with open(path, 'r', encoding='big5') as f:
                        content = f.read(500)
                    self._log(f"檔案預覽（big5編碼）：\n{content}")
                except:
                    self._log("⚠️ 讀取檔案時發生編碼問題，請確認檔案為UTF-8或Big5")
    
    def _analyze_original(self):
        text = self.original_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "請先輸入或貼上原文")
            return
        
        analysis = analyze_text(text)
        if not analysis:
            self._log("❌ 分析失敗：無法識別句子")
            return
        
        result_text = "=" * 40 + "\n"
        result_text += f"句子總數：{analysis['total_sentences']}\n"
        result_text += f"平均句長：{analysis['avg_length']} 字\n"
        result_text += f"最長句子：{analysis['max_length']} 字\n\n"
        
        for level, ok in analysis["compliance"].items():
            std = LEVEL_STANDARDS[level]
            icon = "✅" if ok else "❌"
            result_text += f"{icon} {level.upper()}(≤{std['max_chars']}字)：{'合規' if ok else '超標'}\n"
        
        result_text += "\n" + "-" * 40 + "\n各句分析：\n"
        for i, s in enumerate(analysis["sentences"], 1):
            result_text += f"\n[{i}] ({s['char_count']}字) {s['text']}\n"
            for level, std in LEVEL_STANDARDS.items():
                if s["char_count"] > std["max_chars"]:
                    result_text += f"    ⚠️ 超過{level.upper()}上限({std['max_chars']}字)\n"
        
        self._update_text(self.analysis_result, result_text)
        self._log(f"✅ 分析完成 | 共{analysis['total_sentences']}句 | 均長{analysis['avg_length']}字")

    def _ai_generate(self):
        text = self.original_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "請先輸入或貼上原文以供 AI 分析")
            return
            
        self._log("=====================================")
        self._log("🧠 準備呼叫 LoRA AI 進行自動推論...")
        self._log(f"📂 模型路徑: {self.model_path}")
        self._log("⚠️ 提示: 請在此函式實作 transformers/peft 載入與推論邏輯。")
        self._log("=====================================")
    
    def _show_split_suggestion(self, level, max_chars):
        text = self.original_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "請先輸入原文")
            return
        
        analysis = analyze_text(text)
        if not analysis:
            return
        
        suggestion_text = f"=== {level.upper()} 切分建議 (≤{max_chars}字) ===\n\n"
        all_suggestions = []
        
        for i, s in enumerate(analysis["sentences"], 1):
            splits = suggest_splits(s["text"], max_chars)
            if len(splits) > 1:
                suggestion_text += f"原句[{i}]({s['char_count']}字)：{s['text']}\n"
                suggestion_text += "→ 建議切分：\n"
                for j, split in enumerate(splits, 1):
                    chars = count_chars(split)
                    suggestion_text += f"  {j}. ({chars}字) {split}\n"
            else:
                suggestion_text += f"合規[{i}]({s['char_count']}字)：{s['text']}\n"
            all_suggestions.extend(splits)
            suggestion_text += "\n"
        
        suggestion_text += "-" * 40 + "\n完整建議版本（可直接複製）：\n"
        suggestion_text += "".join(all_suggestions)
        
        self.suggestion_box.delete("1.0", tk.END)
        self.suggestion_box.insert("1.0", suggestion_text)
        self._current_suggestions = {level: "".join(all_suggestions)}
        self._log(f"💡 已生成 {level.upper()} 切分建議")
    
    def _apply_suggestion(self, level):
        if hasattr(self, "_current_suggestions") and level in self._current_suggestions:
            text_widget = self.level_texts[level]
            text_widget.delete("1.0", tk.END)
            text_widget.insert("1.0", self._current_suggestions[level])
            self._log(f"📋 已套用 {level.upper()} 切分建議到輸入框")
        else:
            messagebox.showinfo("提示", f"請先點擊「{level.upper()}建議」按鈕")
    
    def _validate_level(self, level):
        text = self.level_texts[level].get("1.0", tk.END).strip()
        if not text:
            self.level_validations[level].config(text="⏳ 待輸入", bootstyle=SECONDARY)
            return
        
        result = validate_simplified(text, level)
        val_label = self.level_validations[level]
        
        if result["compliant"]:
            analysis = result["analysis"]
            val_label.config(text=f"✅ 合規 | 均{analysis['avg_length']}字", bootstyle=SUCCESS)
            self._log(f"✅ {level.upper()} 驗證通過：{analysis['total_sentences']}句，均長{analysis['avg_length']}字")
        else:
            val_label.config(text=f"❌ 超標 {len(result['violations'])} 處", bootstyle=DANGER)
            msg = f"❌ {level.upper()} 驗證失敗：\n"
            for v in result["violations"]:
                msg += f"  · {v}\n"
            self._log(msg)
    
    def _save_annotation(self):
        lesson = self.current_lesson.get().strip()
        original = self.original_text.get("1.0", tk.END).strip()
        
        if not lesson or not original:
            messagebox.showwarning("提示", "請填寫課文名稱與原文")
            return
        
        record = create_annotation_template(
            lesson_id=f"{lesson}-{len(self.annotations)+1:03d}",
            original_text=original,
            source_file=self.current_file or ""
        )
        
        all_compliant = True
        for level in ["level1", "level2", "level3"]:
            text = self.level_texts[level].get("1.0", tk.END).strip()
            if text:
                validation = validate_simplified(text, level)
                analysis = validation["analysis"]
                record["simplified_versions"][level].update({
                    "text": text,
                    "compliance": validation["compliant"],
                    "sentence_count": analysis["total_sentences"] if analysis else 0,
                    "avg_sentence_length": analysis["avg_length"] if analysis else 0
                })
                if not validation["compliant"]:
                    all_compliant = False
        
        record["metadata"]["annotation_status"] = "completed" if all_compliant else "in_progress"
        self.annotations.append(record)
        self._update_progress()
        
        save_path = filedialog.asksaveasfilename(
            title="儲存標註JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"{lesson}_annotation_{datetime.now().strftime('%m%d_%H%M')}.json",
            initialdir="F:/download/專題"
        )
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            self._log(f"💾 已儲存：{os.path.basename(save_path)}")
            messagebox.showinfo("成功", f"標註已儲存！\n路徑：{save_path}")
    
    def _save_all(self):
        if not self.annotations:
            messagebox.showwarning("提示", "尚無已完成的標註")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="儲存所有標註",
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("JSON files", "*.json")],
            initialfile=f"all_annotations_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
            initialdir="F:/download/專題"
        )
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                for ann in self.annotations:
                    f.write(json.dumps(ann, ensure_ascii=False) + "\n")
            self._log(f"📋 已儲存全部 {len(self.annotations)} 組標註到：{os.path.basename(save_path)}")
            messagebox.showinfo("成功", f"已儲存 {len(self.annotations)} 組標註！")
    
    def _clear_form(self):
        self.current_lesson.set("")
        self.original_text.delete("1.0", tk.END)
        self._update_text(self.analysis_result, "")
        for level in ["level1", "level2", "level3"]:
            self.level_texts[level].delete("1.0", tk.END)
            self.level_validations[level].config(text="⏳ 待輸入", bootstyle=SECONDARY)
        self._log("🔄 表單已清除，可開始下一組標註")
    
    def _log(self, message):
        self.log_box.config(state=NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=DISABLED)
    
    def _update_text(self, widget, text):
        widget.config(state=NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state=DISABLED)
    
    def _update_progress(self):
        total = len(self.annotations)
        l1 = sum(1 for a in self.annotations if a["simplified_versions"]["level1"]["text"])
        l2 = sum(1 for a in self.annotations if a["simplified_versions"]["level2"]["text"])
        l3 = sum(1 for a in self.annotations if a["simplified_versions"]["level3"]["text"])
        
        self.progress_labels["已完成組數"].config(text=str(total))
        self.progress_labels["Level1完成"].config(text=str(l1))
        self.progress_labels["Level2完成"].config(text=str(l2))
        self.progress_labels["Level3完成"].config(text=str(l3))
        self.count_label.config(text=f"已完成：{total} 組")

if __name__ == "__main__":
    # 使用 flatly 主題啟動，具有乾淨、現代的扁平化視覺感
    root = ttk.Window(themename="flatly")
    app = AnnotationGUI(root)
    root.mainloop()