#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox
import re

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

class ReadabilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("最強大腦 - 文本難易度對比分析 (長詞評估版)")
        self.root.geometry("900x650")
        self.root.resizable(False, False)
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="📊 文本難易度對比 (結構與詞彙長度)", font=("Arial", 16, "bold"), pady=10).pack()

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10)

        # 左側
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        tk.Label(left_frame, text="【原始文本】", font=("Arial", 12, "bold")).pack()
        self.text_orig = tk.Text(left_frame, height=15, width=40, font=("Arial", 11))
        self.text_orig.pack(fill="both", expand=True, pady=5)
        self.lbl_result_orig = tk.Label(left_frame, text="等待計算...", font=("Arial", 11), justify="left", fg="blue")
        self.lbl_result_orig.pack(anchor="w", pady=5)

        # 右側
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)
        tk.Label(right_frame, text="【簡化版本】", font=("Arial", 12, "bold")).pack()
        self.text_simp = tk.Text(right_frame, height=15, width=40, font=("Arial", 11))
        self.text_simp.pack(fill="both", expand=True, pady=5)
        self.lbl_result_simp = tk.Label(right_frame, text="等待計算...", font=("Arial", 11), justify="left", fg="#D35400")
        self.lbl_result_simp.pack(anchor="w", pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=15)
        tk.Button(btn_frame, text="🚀 開始分析比對", font=("Arial", 14, "bold"), bg="#4CAF50", fg="black", command=self.analyze).pack()

    def calculate_metrics(self, text):
        if not text.strip():
            return None

        # 計算句數
        sentences = re.split(r'[。！？；\!\?]+', text)
        sentences = [s for s in sentences if s.strip()]
        num_sentences = max(len(sentences), 1)

        # 計算有效中文字
        clean_text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text)
        num_chars = len(clean_text)

        num_words = 0
        long_words_count = 0
        
        if JIEBA_AVAILABLE:
            words = list(jieba.cut(text))
            # 確保只計算真正的詞彙，排除標點與空白
            clean_words = [w for w in words if re.fullmatch(r'[\w\u4e00-\u9fff]+', w)]
            num_words = max(len(clean_words), 1)
            
            # 🌟 新的詞彙難度邏輯：計算長詞 (>= 3個字) 的數量
            for w in clean_words:
                if len(w) >= 3:
                    long_words_count += 1

        avg_sent_len = num_chars / num_sentences if num_sentences > 0 else 0
        # 計算長詞佔比
        long_word_ratio = (long_words_count / num_words) * 100 if num_words > 0 else 0

        return {
            "chars": num_chars,
            "sentences": num_sentences,
            "avg_sent_len": avg_sent_len,
            "long_ratio": long_word_ratio
        }

    def format_result(self, metrics):
        if not metrics:
            return "（請輸入文本）"
        
        return (
            f"📌 有效字數　： {metrics['chars']} 字\n"
            f"📌 總句數　　： {metrics['sentences']} 句\n"
            f"📏 平均句長　： {metrics['avg_sent_len']:.1f} 字/句\n"
            f"⚠️ 長詞比例　： {metrics['long_ratio']:.1f} % (>=3字, 越低越簡單)"
        )

    def analyze(self):
        if not JIEBA_AVAILABLE:
            messagebox.showerror("錯誤", "找不到 jieba 套件")
            return

        orig_text = self.text_orig.get("1.0", tk.END)
        simp_text = self.text_simp.get("1.0", tk.END)

        orig_metrics = self.calculate_metrics(orig_text)
        simp_metrics = self.calculate_metrics(simp_text)

        self.lbl_result_orig.config(text=self.format_result(orig_metrics))
        self.lbl_result_simp.config(text=self.format_result(simp_metrics))

if __name__ == "__main__":
    root = tk.Tk()
    app = ReadabilityApp(root)
    root.mainloop()