#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter

# 嘗試載入 jieba
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


# ==========================================
# 核心 SARI 演算法邏輯
# ==========================================
def tokenize(text, language='en'):
    """將文字轉為 token 列表"""
    if language == 'zh':
        if not JIEBA_AVAILABLE:
            raise ImportError("請先安裝 jieba：pip install jieba")
        return list(jieba.cut(text))
    else:
        return text.split()

def get_ngrams(tokens, n):
    """從 token 序列中產生 n-gram"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_sari(original, references, output, language='en'):
    """計算 SARI 分數與三大細項 (整篇文章層級)"""
    orig_tokens = tokenize(original, language)
    refs_tokens = [tokenize(ref, language) for ref in references]
    out_tokens = tokenize(output, language)

    max_n = 4
    r = len(refs_tokens)
    if r == 0:
        return {"sari": 0.0, "add": 0.0, "keep": 0.0, "del": 0.0}

    add_prec, add_rec = [], []
    keep_prec, keep_rec = [], []
    del_prec = []

    for n in range(1, max_n + 1):
        orig_ngrams = Counter(get_ngrams(orig_tokens, n))
        out_ngrams = Counter(get_ngrams(out_tokens, n))
        refs_ngrams = [Counter(get_ngrams(ref, n)) for ref in refs_tokens]
        ref_counts = sum(refs_ngrams, Counter())

        # ---------- 新增 (addition) ----------
        out_not_in_orig = {ng: cnt for ng, cnt in out_ngrams.items() if ng not in orig_ngrams}
        ref_not_in_orig = {ng: cnt for ng, cnt in ref_counts.items() if ng not in orig_ngrams}

        if out_not_in_orig:
            numerator = sum(min(cnt, ref_counts.get(ng, 0)) for ng, cnt in out_not_in_orig.items())
            denominator = sum(cnt for cnt in out_not_in_orig.values())
            p_add = numerator / denominator if denominator > 0 else 0
        else:
            p_add = 0

        if ref_not_in_orig:
            numerator = sum(min(out_ngrams.get(ng, 0), cnt) for ng, cnt in ref_not_in_orig.items())
            denominator = sum(cnt for cnt in ref_not_in_orig.values())
            r_add = numerator / denominator if denominator > 0 else 0
        else:
            r_add = 0

        # ---------- 保留 (keep) ----------
        keep_target = {}
        for ng, cnt_in in orig_ngrams.items():
            cnt_ref = ref_counts.get(ng, 0)
            keep_target[ng] = min(cnt_in, cnt_ref / r)

        in_and_out = {ng: min(cnt_in, out_ngrams.get(ng, 0)) for ng, cnt_in in orig_ngrams.items()}

        denominator_keep_p = sum(in_and_out.values())
        if denominator_keep_p > 0:
            numerator_keep = sum(min(in_and_out[ng], keep_target[ng]) for ng in orig_ngrams)
            p_keep = numerator_keep / denominator_keep_p
        else:
            p_keep = 0

        denominator_keep_r = sum(keep_target.values())
        r_keep = numerator_keep / denominator_keep_r if denominator_keep_r > 0 else 0

        # ---------- 刪除 (deletion) ----------
        denominator_del = 0
        numerator_del = 0
        for ng, cnt_in in orig_ngrams.items():
            cnt_out = out_ngrams.get(ng, 0)
            del_cnt = max(cnt_in - cnt_out, 0)
            if del_cnt > 0:
                denominator_del += del_cnt
                target_del = max(cnt_in - ref_counts.get(ng, 0) / r, 0)
                numerator_del += min(del_cnt, target_del)

        p_del = numerator_del / denominator_del if denominator_del > 0 else 0

        add_prec.append(p_add)
        add_rec.append(r_add)
        keep_prec.append(p_keep)
        keep_rec.append(r_keep)
        del_prec.append(p_del)

    # 計算 F1 與精準度
    P_add = sum(add_prec) / max_n
    R_add = sum(add_rec) / max_n
    F_add = (2 * P_add * R_add / (P_add + R_add)) if (P_add + R_add) > 0 else 0

    P_keep = sum(keep_prec) / max_n
    R_keep = sum(keep_rec) / max_n
    F_keep = (2 * P_keep * R_keep / (P_keep + R_keep)) if (P_keep + R_keep) > 0 else 0

    P_del = sum(del_prec) / max_n

    sari_total = (F_add + F_keep + P_del) / 3
    
    # 回傳字典，包含總分與三個細項
    return {
        "sari": sari_total,
        "add": F_add,
        "keep": F_keep,
        "del": P_del
    }

def read_document(filename):
    """讀取整個檔案並合併為一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip().replace('\n', ' ')

# ==========================================
# GUI 介面設計
# ==========================================
class SARIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("最強大腦 - SARI 文本簡化深度評估工具")
        self.root.geometry("650x700")
        self.root.resizable(False, False)

        self.orig_file = ""
        self.out_file = ""
        self.ref_files = []
        self.language = tk.StringVar(value="en")

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="SARI 文本簡化深度評估 (含細項拆解)", font=("Arial", 16, "bold")).pack(pady=10)

        # 語言選擇
        lang_frame = tk.Frame(self.root)
        lang_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(lang_frame, text="語言：", font=("Arial", 10, "bold")).pack(side="left")
        tk.Radiobutton(lang_frame, text="英文", variable=self.language, value="en").pack(side="left", padx=10)
        tk.Radiobutton(lang_frame, text="中文 (需裝jieba)", variable=self.language, value="zh").pack(side="left")

        # 檔案選擇區塊
        orig_frame = tk.Frame(self.root)
        orig_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(orig_frame, text="1. 原始文件 (Original):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.lbl_orig = tk.Label(orig_frame, text="尚未選擇檔案...", fg="gray")
        self.lbl_orig.pack(side="left", fill="x", expand=True)
        tk.Button(orig_frame, text="瀏覽", command=self.load_orig).pack(side="right")

        out_frame = tk.Frame(self.root)
        out_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(out_frame, text="2. 系統輸出 (Output - 你的簡化版):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.lbl_out = tk.Label(out_frame, text="尚未選擇檔案...", fg="gray")
        self.lbl_out.pack(side="left", fill="x", expand=True)
        tk.Button(out_frame, text="瀏覽", command=self.load_out).pack(side="right")

        ref_frame = tk.Frame(self.root)
        ref_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(ref_frame, text="3. 參考文件 (References - 標準答案):", font=("Arial", 10, "bold")).pack(anchor="w")
        
        btn_frame = tk.Frame(ref_frame)
        btn_frame.pack(fill="x", pady=2)
        tk.Button(btn_frame, text="+ 新增參考檔", command=self.add_ref).pack(side="left", padx=(0, 10))
        tk.Button(btn_frame, text="清空參考檔", command=self.clear_refs).pack(side="left")

        self.listbox_refs = tk.Listbox(ref_frame, height=3)
        self.listbox_refs.pack(fill="x")

        # 計算按鈕
        tk.Button(self.root, text="🚀 開始計算 SARI 深度分析", font=("Arial", 12, "bold"), bg="#4CAF50", fg="black", command=self.run_calculation).pack(pady=15)
        
        # 結果顯示區塊 (使用 Text 元件以便排版)
        self.text_result = tk.Text(self.root, height=10, width=60, font=("Arial", 12), bg="#f4f4f4", state="disabled")
        self.text_result.pack(padx=20, pady=5)

    def load_orig(self):
        fp = filedialog.askopenfilename(title="選擇原始文件")
        if fp:
            self.orig_file = fp
            self.lbl_orig.config(text=fp, fg="black")

    def load_out(self):
        fp = filedialog.askopenfilename(title="選擇系統輸出文件")
        if fp:
            self.out_file = fp
            self.lbl_out.config(text=fp, fg="black")

    def add_ref(self):
        fps = filedialog.askopenfilenames(title="選擇參考文件")
        for fp in fps:
            if fp not in self.ref_files:
                self.ref_files.append(fp)
                self.listbox_refs.insert(tk.END, fp)

    def clear_refs(self):
        self.ref_files.clear()
        self.listbox_refs.delete(0, tk.END)

    def display_result(self, message):
        self.text_result.config(state="normal")
        self.text_result.delete("1.0", tk.END)
        self.text_result.insert(tk.END, message)
        self.text_result.config(state="disabled")

    def run_calculation(self):
        if not self.orig_file or not self.out_file or not self.ref_files:
            messagebox.showwarning("警告", "請確保原始檔、輸出檔、參考檔都已選擇！")
            return
        
        lang = self.language.get()
        if lang == 'zh' and not JIEBA_AVAILABLE:
            messagebox.showerror("錯誤", "需要安裝 jieba (pip install jieba)")
            return

        try:
            self.display_result("計算中，請稍候...\n這可能會花費幾秒鐘。")
            self.root.update()

            orig_text = read_document(self.orig_file)
            out_text = read_document(self.out_file)
            refs_texts = [read_document(f) for f in self.ref_files]

            # 計算分數
            scores = compute_sari(orig_text, refs_texts, out_text, lang)
            
            # 排版輸出結果
            result_str = (
                f"🏆 系統輸出總分 (SARI)： {scores['sari']:.4f}\n"
                f"{'-'*40}\n"
                f"📊 細項指標拆解：\n"
                f" 🟢 保留 (Keep) 分數 ： {scores['keep']:.4f}  (語意保留度)\n"
                f" 🔴 刪除 (Delete) 分數： {scores['del']:.4f}  (冗言刪除度)\n"
                f" 🔵 新增 (Add) 分數   ： {scores['add']:.4f}  (新詞補充度)\n"
                f"{'-'*40}\n"
                f"💡 數據解讀提示：\n"
                f"保留(Keep)高，代表成功維持了原文的核心意境與語意。\n"
                f"刪除(Del)高，代表成功移除了造成大腦負擔的冗長字句。"
            )

            self.display_result(result_str)

        except Exception as e:
            messagebox.showerror("程式錯誤", f"計算發生錯誤：\n{str(e)}")
            self.display_result("計算失敗，請檢查檔案格式。")

if __name__ == "__main__":
    root = tk.Tk()
    app = SARIApp(root)
    root.mainloop()