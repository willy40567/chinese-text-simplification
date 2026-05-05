#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox, filedialog
import re
import os

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# 這裡換成你剛剛成功載入的 CSV 或 Excel 檔案路徑
DEFAULT_TBCL_PATH = r"F:\download\專題\hsk詞彙\14452詞語表202504.xlsx - 14452詞語表202504.csv"

class GoldenRatioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("最強大腦 - 閱讀障礙黃金比例檢測器 (TBCL 對標版)")
        self.root.geometry("1000x750")
        self.root.resizable(False, False)
        
        self.tbcl_dict = {}
        self.create_widgets()
        
        # 啟動時自動載入預設詞庫
        if PANDAS_AVAILABLE:
            self.load_dictionary(DEFAULT_TBCL_PATH, silent=True)
        else:
            messagebox.showerror("錯誤", "缺少 pandas 套件！請執行: pip install pandas")

    def create_widgets(self):
        header_frame = tk.Frame(self.root)
        header_frame.pack(fill="x", pady=10)
        tk.Label(header_frame, text="🎯 閱讀障礙友善：詞彙黃金比例檢測器", font=("Arial", 16, "bold")).pack()
        tk.Label(header_frame, text="對標程度：國小四年級閱讀障礙學童", font=("Arial", 11), fg="#555555").pack()
        
        self.lbl_dict_status = tk.Label(header_frame, text="詞庫狀態：尚未載入", fg="red", font=("Arial", 10))
        self.lbl_dict_status.pack()
        tk.Button(header_frame, text="手動載入 詞庫檔案 (CSV/Excel)", command=self.manual_load_dict).pack(pady=5)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(left_frame, text="【原始文本】", font=("Arial", 12, "bold")).pack()
        self.text_orig = tk.Text(left_frame, height=14, width=40, font=("Arial", 11))
        self.text_orig.pack(fill="both", expand=True, pady=5)
        self.lbl_result_orig = tk.Label(left_frame, text="等待計算...", font=("Arial", 11), justify="left", fg="blue")
        self.lbl_result_orig.pack(anchor="w", pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)
        tk.Label(right_frame, text="【簡化版本】", font=("Arial", 12, "bold")).pack()
        self.text_simp = tk.Text(right_frame, height=14, width=40, font=("Arial", 11))
        self.text_simp.pack(fill="both", expand=True, pady=5)
        self.lbl_result_simp = tk.Label(right_frame, text="等待計算...", font=("Arial", 11), justify="left", fg="#D35400")
        self.lbl_result_simp.pack(anchor="w", pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=15)
        tk.Button(btn_frame, text="🚀 檢測黃金比例", font=("Arial", 14, "bold"), bg="#4CAF50", fg="black", command=self.analyze).pack()

    def load_dictionary(self, filepath, silent=False):
        if not os.path.exists(filepath):
            if not silent:
                messagebox.showwarning("找不到檔案", f"找不到詞庫：\n{filepath}\n請點擊手動載入。")
            return

        try:
            self.lbl_dict_status.config(text="詞庫載入中，請稍候...", fg="blue")
            self.root.update()

            if filepath.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='big5')
            else:
                df = pd.read_excel(filepath)
            
            def _find_col(columns, keywords):
                lower_map = {str(c).strip().lower(): c for c in columns}
                for k in keywords:
                    if k.lower() in lower_map:
                        return lower_map[k.lower()]
                for col in columns:
                    name = str(col).strip().lower()
                    if any(k.lower() in name for k in keywords):
                        return col
                return None

            if df.empty:
                raise ValueError("Excel/CSV 內容為空")

            word_col = _find_col(df.columns, ["word", "詞", "词"])
            level_col = _find_col(df.columns, ["ji", "級", "级", "level", "deng"])
            if not word_col or not level_col:
                cols = ", ".join([str(c) for c in df.columns])
                raise ValueError(f"找不到必要欄位（詞/級）。目前欄位：{cols}")

            self.tbcl_dict.clear()
            for index, row in df.iterrows():
                word = str(row[word_col]).strip()
                level_val = str(row[level_col]).strip()
                
                match = re.search(r'\d+', level_val)
                if word and match:
                    self.tbcl_dict[word] = int(match.group())

            self.lbl_dict_status.config(text=f"✅ 詞庫載入成功！共 {len(self.tbcl_dict)} 個詞彙", fg="green")
            if not silent:
                messagebox.showinfo("成功", f"成功載入詞庫！共 {len(self.tbcl_dict)} 個詞彙。")
                
        except Exception as e:
            self.lbl_dict_status.config(text="❌ 詞庫載入失敗", fg="red")
            if not silent:
                messagebox.showerror("載入失敗", f"讀取檔案發生錯誤：\n{str(e)}")

    def manual_load_dict(self):
        filepath = filedialog.askopenfilename(
            title="選擇 TBCL/HSK 詞表", 
            filetypes=[("CSV/Excel files", "*.csv *.xlsx *.xls"), ("All files", "*.*")]
        )
        if filepath:
            self.load_dictionary(filepath)

    def analyze_text(self, text):
        if not text.strip():
            return None

        num_words = 0
        counts = {"basic": 0, "inter": 0, "adv": 0, "oov": 0}
        
        if JIEBA_AVAILABLE:
            words = list(jieba.cut(text))
            clean_words = [w for w in words if re.fullmatch(r'[\w\u4e00-\u9fff]+', w)]
            num_words = max(len(clean_words), 1)
            
            for word in clean_words:
                level = self.tbcl_dict.get(word, 0)
                
                # 依據國小四年級黃金比例重新分類
                if 1 <= level <= 3:
                    counts["basic"] += 1   # 基礎詞 (1~3級)
                elif level == 4:
                    counts["inter"] += 1   # 進階詞 (4級)
                elif 5 <= level <= 7:
                    counts["adv"] += 1     # 精熟詞 (5~7級)
                else:
                    counts["oov"] += 1     # 未收錄詞 (OOV)

        p_basic = (counts["basic"] / num_words) * 100
        p_inter = (counts["inter"] / num_words) * 100
        p_adv = (counts["adv"] / num_words) * 100
        p_oov = (counts["oov"] / num_words) * 100

        return {
            "total_words": num_words,
            "counts": counts,
            "basic_pct": p_basic,
            "inter_pct": p_inter,
            "adv_pct": p_adv,
            "oov_pct": p_oov
        }

    def format_result(self, metrics):
        if not metrics:
            return "（請輸入文本）"
        
        counts = metrics['counts']
        p_basic = metrics['basic_pct']
        p_adv = metrics['adv_pct']
        p_oov = metrics['oov_pct']

        # 判定是否達標
        chk_basic = "✅ 達標" if p_basic >= 80 else "❌ 偏低"
        chk_adv = "✅ 達標" if (p_adv + p_oov) <= 15 else "❌ 偏高" # 精熟+OOV 最好壓在 15% 內，嚴格需 5%

        return (
            f"📌 系統有效詞數： {metrics['total_words']} 詞\n"
            f"{'-'*35}\n"
            f"🟢 基礎舒適區 (第1~3級) ： {counts['basic']:>3} 個 ({p_basic:5.1f} %) | 目標 ≥ 80%\n"
            f"🟡 些微挑戰區 (第 4 級)  ： {counts['inter']:>3} 個 ({metrics['inter_pct']:5.1f} %) | 目標 10~15%\n"
            f"🔴 艱澀精熟區 (第5級以上)： {counts['adv']:>3} 個 ({p_adv:5.1f} %) | 目標 ≤ 5%\n"
            f"⚫ 未收錄生字 (OOV)      ： {counts['oov']:>3} 個 ({p_oov:5.1f} %)\n"
            f"{'-'*35}\n"
            f"🎯 國小四年級友善度檢測：\n"
            f" ► 基礎詞彙覆蓋率 (≥80%)： {chk_basic}\n"
            f" ► 困難詞彙控制度 (≤15%)： {chk_adv}"
        )

    def analyze(self):
        if not self.tbcl_dict:
            messagebox.showwarning("警告", "請先確保詞庫已成功載入！")
            return

        orig_text = self.text_orig.get("1.0", tk.END)
        simp_text = self.text_simp.get("1.0", tk.END)

        orig_metrics = self.analyze_text(orig_text)
        simp_metrics = self.analyze_text(simp_text)

        self.lbl_result_orig.config(text=self.format_result(orig_metrics))
        self.lbl_result_simp.config(text=self.format_result(simp_metrics))

if __name__ == "__main__":
    root = tk.Tk()
    app = GoldenRatioAnalyzerApp(root)
    root.mainloop()