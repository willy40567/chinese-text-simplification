import os
import re
import threading
import warnings
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 從共用模組匯入
# ==========================================
from config import SBERT_MODEL, LEVEL_LIMITS, LEVEL_INSTRUCTION
from hsk_utils import (
    get_tw2sp, get_sp2tw,
    build_hsk_vocab_txt, load_hsk_dict,
    hsk_replace, register_proper_nouns_jieba,
)

# ==========================================
# 全域變數
# ==========================================
model          = None
tokenizer      = None
my_hsk_dict    = None
pipeline_ready = False
semantic_model = None

# ==========================================
# GUI 專用常數 (不適合放在共用模組)
# ==========================================
SUSPICIOUS_PLACES = {
    "伦敦","北京","上海","美国","英国","日本",
    "法国","德国","俄罗斯","印度","荷兰人"
}

NOISE_TRIGGERS = [
    "簡化", "答案", "原句", "規則", "注意", "請輸入",
    "每句", "改寫", "此功能", "→", "（合併", "輸出",
    "# ", "```", "The best", "Best answer",
    "翻譯：", "英文版", "原文"
]

TYPO_CORRECTIONS = {
    "一撇即分": "一撥即分",
    "幹香菇": "乾香菇",
}

CAUSAL_HINTS = ("因為", "所以", "因此", "由於", "導致")
# ==========================================
# 正則預編譯
# ==========================================
RE_CHINESE = re.compile(r'[\u4e00-\u9fff]')
RE_ENGLISH_WORDS = re.compile(r'[a-zA-Z]{3,}')
RE_REPEAT_QUOTE = re.compile(r'[」』]{3,}')
RE_REPEAT_PERIOD = re.compile(r'[。]{2,}')
RE_NEWLINE = re.compile(r'\n+')
RE_SPLIT_SENTENCE = re.compile(r'(?<=[。！？；])')
RE_SPLIT_COMMA = re.compile(r'(?<=[，])')
RE_HEADING = re.compile(
    r'^\s*(?:第[一二三四五六七八九十百千0-9]+[章節篇]|[一二三四五六七八九十0-9]+[、.）)]|（[一二三四五六七八九十0-9]+）|[壹貳參肆伍陸柒捌玖拾]+、)'
)
# ==========================================
# 管線工具函數
# ==========================================
def is_heading_like(text):
    """偵測章節/條列標題，避免在高壓縮層被誤刪。"""
    s = text.strip()
    if not s:
        return False

    if RE_HEADING.match(s):
        return True

    if s.endswith(("：", ":")) and len(s) <= 30:
        return True

    # 無句末標點、字數短的獨立片段，多半是小標題
    if len(s) <= 16 and not re.search(r'[。！？!?]$', s) and '，' not in s:
        return True

    return False

## load_hsk_dict, should_skip, hsk_replace 已移至 hsk_utils.py

def is_valid_chinese_output(text):
    """檢查輸出是否為有效中文，排除幻覺/雜訊"""
    if not text or len(text) < 2:
        return False
    # 英文字母佔比超過 15% → 無效
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0 or chinese_chars / total_chars < 0.85:
        return False
    # 含 # 或 ``` → 無效
    if '#' in text or '```' in text:
        return False
    # 含連續英文單字 → 無效
    if RE_ENGLISH_WORDS.search(text):
        return False
    return True

def is_hallucination(original, generated):
    import jieba
    tw2sp = get_tw2sp()
    sp2tw = get_sp2tw()
    orig_simp = tw2sp.convert(original)
    gen_words = [w for w in jieba.cut(tw2sp.convert(generated)) if len(w) >= 2]
    for w in gen_words:
        if w in SUSPICIOUS_PLACES and w not in orig_simp:
            return True, sp2tw.convert(w)
    return False, None
def _sbert_max_similarity(pred_sent, src_sents):
    """回傳 pred_sent 與 source 句群的最高 cosine 相似度。"""
    import torch

    if semantic_model is None:
        return None
    if not pred_sent or not src_sents:
        return None

    texts = [pred_sent] + src_sents
    emb = semantic_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = torch.matmul(emb[0], emb[1:].T)
    return float(torch.max(sims).item())


def similarity_filter(source, prediction, threshold=0.45):
    """移除與原文語意無關的幻覺句。優先使用 Sentence-BERT，失敗時回退字元相似度。"""
    pred_sents = [s.strip() for s in re.split(r'[。！？]', prediction) if s.strip()]
    src_sents = [s.strip() for s in re.split(r'[。！？\n，]', source) if s.strip()]
    if not src_sents:
        return prediction

    clean = []
    for ps in pred_sents:
        if '「' in ps or '」' in ps:
            clean.append(ps)
            continue

        max_sim = _sbert_max_similarity(ps, src_sents)

        if max_sim is None:
            overlap = [len(set(ps) & set(ss)) / max(1, len(set(ps) | set(ss))) for ss in src_sents]
            max_sim = max(overlap) if overlap else 0.0

        print(f"[SBERT] sim={max_sim:.3f} | pred={ps[:20]} | src={source[:20]}")

        if max_sim >= threshold:
            clean.append(ps)

    return '。'.join(clean) + '。' if clean else source

def normalize_quote_punctuation(text):
    """修正中文引號問句標點，並補上遺失的左引號。"""
    fixed = re.sub(r'([嗎呢吧呀麼？?])。」', r'\1？」', text)

    # 句尾是 ？」但沒有左引號時，自動補上「
    fixed = re.sub(
        r'(^|[。！？\n])([^「」\n]{1,40}[嗎呢吧呀麼])？」',
        lambda m: f"{m.group(1)}「{m.group(2)}？」",
        fixed,
    )
    return fixed


def apply_text_corrections(text):
    """固定常見錯字與標點問題。"""
    fixed = text
    for src, dst in TYPO_CORRECTIONS.items():
        fixed = fixed.replace(src, dst)
    fixed = normalize_quote_punctuation(fixed)
    return fixed


def lora_generate(sent, level):
    import torch
    instruction = LEVEL_INSTRUCTION.get(
        level, "請將以下文本簡化，且不可新增原文內容。"
    )

    messages = [
        {
            "role": "system",
            "content": "你是一個專業的中文語言學家，負責將複雜的中文句子改寫成簡單的初級中文。"
        },
        {
            "role": "user",
            "content": f"{instruction}\n原文：{sent}\n請只輸出改寫結果。"
        }
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"""{instruction}
原文：{sent}
請只輸出改寫結果。"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            repetition_penalty=1.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    raw = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # 清洗層 1：遇到雜訊關鍵字截斷
    for noise in NOISE_TRIGGERS:
        if noise in raw:
            raw = raw[:raw.index(noise)].strip()

    # 清洗層 2：移除連續重複標點
    raw = RE_REPEAT_QUOTE.sub('。', raw)
    raw = RE_REPEAT_PERIOD.sub('。', raw)
    raw = RE_NEWLINE.sub('。', raw)

    # 清洗層 3：只取第一個完整句
    if '。' in raw:
        raw = raw.split('。')[0] + '。'

    # 清洗層 4：長度保護
    raw = raw[:60]

    # 清洗層 5：補句號
    if raw and raw[-1] not in '。！？':
        raw += '。'

    # 清洗層 5.5：錯字與標點校正
    raw = apply_text_corrections(raw)

    # 清洗層 6：中文有效性檢查
    if not is_valid_chinese_output(raw):
        return sent

    # 清洗層 7：地名幻覺檢查
    hallucinated, _ = is_hallucination(sent, raw)
    if hallucinated:
        return sent

    # 清洗層 8：語意相似度過濾
    raw = similarity_filter(sent, raw)
    if not raw or not is_valid_chinese_output(raw):
        return sent

    return raw

def postprocess_sentence(sent, level, max_retry=3):
    if is_heading_like(sent):
        return apply_text_corrections(sent)

    char_limit = LEVEL_LIMITS[level]
    result = sent
    for attempt in range(max_retry):
        result = hsk_replace(result, my_hsk_dict, level)
        clean = "".join(c for c in result if c not in "。，、！？：；「」")
        if len(clean) <= char_limit:
            break
        if attempt < max_retry - 1 and model is not None:
            new_result = lora_generate(sent, level)
            if is_valid_chinese_output(new_result):
                result = new_result
    result = apply_text_corrections(result)
    return result

def split_sentences(text):
    sentences = RE_SPLIT_SENTENCE.split(text.strip())
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 超過 40 字才切，且切分時保留逗號/頓號
        if len(s) > 40:
            sub = re.split(r'(?<=[，、])', s)  # 保留分隔符
            result.extend([x.strip() for x in sub if x.strip()])
        else:
            result.append(s)
    return result

# ==========================================
# 管線初始化（背景執行）
# ==========================================
def init_pipeline(log_cb):
    global model, tokenizer, my_hsk_dict, pipeline_ready, semantic_model
    try:
        import jieba

        log_cb("初始化 OpenCC 轉換器...")
        get_tw2sp()  # 觸發 lazy init
        get_sp2tw()

        log_cb("建立 HSK 字典...")
        build_hsk_vocab_txt()
        my_hsk_dict = load_hsk_dict()
        log_cb(f"✅ HSK 字典載入，共 {len(my_hsk_dict)} 詞")

        log_cb("設定 Jieba 斷詞...")
        register_proper_nouns_jieba()
        jieba.suggest_freq(('护士', '学校'), True)
        jieba.suggest_freq(('第一', '所'), True)
        log_cb("✅ Jieba 設定完成")

        log_cb("載入 Tokenizer...")
        from model_utils import load_finetuned_model

        log_cb("載入基底模型 + LoRA（4-bit 量化，約需 30 秒）...")
        model, tokenizer = load_finetuned_model()

        log_cb("載入 Sentence-BERT 語意過濾器...")
        try:
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer(SBERT_MODEL)
            log_cb("✅ Sentence-BERT 載入完成")
        except Exception as sem_e:
            semantic_model = None
            log_cb(f"⚠️ Sentence-BERT 載入失敗，改用回退相似度：{sem_e}")

        log_cb("✅ 模型載入完成，管線就緒！")
        pipeline_ready = True

    except Exception as e:
        log_cb(f"❌ 初始化失敗：{e}")
        pipeline_ready = False
        semantic_model = None

# ==========================================
# GUI 主體
# ==========================================
class SimplificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("中文文本三級簡化系統 v2")
        self.root.geometry("1280x820")
        self.root.minsize(1120, 760)

        self.colors = {
            "bg": "#f4f7fb",
            "panel": "#eaf0f7",
            "surface": "#ffffff",
            "header": "#0f172a",
            "accent": "#0ea5e9",
            "accent_hover": "#0284c7",
            "success": "#16a34a",
            "success_hover": "#15803d",
            "muted": "#64748b",
            "text": "#0f172a",
            "subtext": "#94a3b8",
            "line": "#dbe5f0"
        }

        self.root.configure(bg=self.colors["bg"])
        self.style = ttk.Style()
        self._setup_styles()
        self._build_ui()
        self._start_init()

    def _setup_styles(self):
        self.style.theme_use("clam")
        self.style.configure(
            "Modern.TNotebook",
            background=self.colors["panel"],
            borderwidth=0,
            tabmargins=(8, 4, 8, 0)
        )
        self.style.configure(
            "Modern.TNotebook.Tab",
            background="#dbe7f3",
            foreground=self.colors["muted"],
            font=("Segoe UI", 10, "bold"),
            padding=(16, 8),
            borderwidth=0
        )
        self.style.map(
            "Modern.TNotebook.Tab",
            background=[("selected", self.colors["surface"]), ("active", "#e6edf6")],
            foreground=[("selected", self.colors["text"])]
        )
        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor="#dbe7f3",
            background=self.colors["accent"],
            bordercolor="#dbe7f3",
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
            thickness=12
        )

    def _create_button(self, parent, text, cmd, bg, hover, font=("Segoe UI", 10, "bold"), padx=12, pady=6):
        btn = tk.Button(
            parent,
            text=text,
            command=cmd,
            font=font,
            bg=bg,
            fg="white",
            activebackground=hover,
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=padx,
            pady=pady
        )

        def on_enter(_):
            if btn["state"] != tk.DISABLED:
                btn.configure(bg=hover)

        def on_leave(_):
            if btn["state"] != tk.DISABLED:
                btn.configure(bg=bg)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def _build_ui(self):
        # 頂部標題列
        title_bar = tk.Frame(self.root, bg=self.colors["header"], pady=10)
        title_bar.pack(fill=tk.X)
        tk.Label(
            title_bar,
            text="中文文本三級簡化系統",
            font=("Segoe UI Semibold", 16),
            bg=self.colors["header"],
            fg="white"
        ).pack(side=tk.LEFT, padx=18)
        tk.Label(
            title_bar,
            text="Chain Simplification: Original -> L1 -> L2 -> L3",
            font=("Segoe UI", 10),
            bg=self.colors["header"],
            fg=self.colors["subtext"]
        ).pack(side=tk.LEFT)

        # 主體框架
        main = tk.Frame(self.root, bg=self.colors["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # 左側輸入
        left = tk.LabelFrame(
            main,
            text=" Input Text ",
            font=("Segoe UI Semibold", 10),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            padx=8,
            pady=8,
            bd=0
        )
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.input_text = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            font=("Noto Sans TC", 11),
            height=28,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            relief=tk.FLAT,
            insertbackground=self.colors["text"],
            padx=10,
            pady=8
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(left, bg=self.colors["panel"])
        btn_row.pack(fill=tk.X, pady=(8, 0))

        load_btn = self._create_button(
            btn_row,
            "載入檔案",
            self._load_file,
            self.colors["accent"],
            self.colors["accent_hover"],
            font=("Segoe UI", 10)
        )
        load_btn.pack(side=tk.LEFT, padx=(0, 6))

        clear_btn = self._create_button(
            btn_row,
            "清除內容",
            lambda: self.input_text.delete("1.0", tk.END),
            self.colors["muted"],
            "#475569",
            font=("Segoe UI", 10)
        )
        clear_btn.pack(side=tk.LEFT)

        self.run_btn = self._create_button(
            btn_row,
            "開始鏈式簡化",
            self._run_pipeline,
            self.colors["success"],
            self.colors["success_hover"],
            font=("Segoe UI Semibold", 10),
            padx=16
        )
        self.run_btn.config(state=tk.DISABLED)
        self.run_btn.pack(side=tk.RIGHT)

        # 右側輸出
        right = tk.LabelFrame(
            main,
            text=" Simplified Output ",
            font=("Segoe UI Semibold", 10),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            padx=8,
            pady=8,
            bd=0
        )
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        nb = ttk.Notebook(right, style="Modern.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)

        self.output_texts = {}
        tab_cfg = [
            ("L1 <=20", 1, "#eefbf2"),
            ("L2 <=15", 2, "#eef7ff"),
            ("L3 <=12", 3, "#fff1f6"),
        ]
        for tab_name, level, color in tab_cfg:
            frame = tk.Frame(nb, bg=color)
            nb.add(frame, text=tab_name)

            out = scrolledtext.ScrolledText(
                frame,
                wrap=tk.WORD,
                font=("Noto Sans TC", 11),
                bg=color,
                fg=self.colors["text"],
                relief=tk.FLAT,
                insertbackground=self.colors["text"],
                padx=10,
                pady=8
            )
            out.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
            self.output_texts[level] = out

            save_btn = self._create_button(
                frame,
                "儲存",
                lambda lv=level: self._save_output(lv),
                self.colors["muted"],
                "#475569",
                font=("Segoe UI", 9),
                padx=10,
                pady=4
            )
            save_btn.pack(anchor=tk.E, padx=6, pady=(0, 6))

        # 底部狀態列
        bottom = tk.Frame(self.root, bg=self.colors["surface"], bd=0)
        bottom.pack(fill=tk.X, padx=14, pady=(0, 6))

        self.status_var = tk.StringVar(value="正在初始化管線，請稍候...")
        tk.Label(
            bottom,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            anchor=tk.W
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=6)

        self.progress = ttk.Progressbar(
            bottom,
            mode="determinate",
            length=240,
            style="Modern.Horizontal.TProgressbar"
        )
        self.progress.pack(side=tk.RIGHT, padx=10, pady=10)

        # 日誌視窗
        log_frame = tk.LabelFrame(
            self.root,
            text=" System Log ",
            font=("Segoe UI", 9),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            bd=0
        )
        log_frame.pack(fill=tk.X, padx=14, pady=(0, 8))

        self.log_text = tk.Text(
            log_frame,
            height=4,
            font=("Consolas", 9),
            bg="#0b1220",
            fg="#7dd3fc",
            insertbackground="#7dd3fc",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.X, padx=6, pady=6)

    # ── 初始化 ──
    def _start_init(self):
        threading.Thread(
            target=init_pipeline, args=(self._log,), daemon=True
        ).start()
        self.root.after(500, self._check_ready)

    def _check_ready(self):
        if pipeline_ready:
            self.run_btn.config(state=tk.NORMAL)
            self._set_status("✅ 管線就緒，可以開始簡化")
        else:
            self.root.after(1000, self._check_ready)

    # ── 鏈式簡化主流程 ──
    def _run_pipeline(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "請先輸入或載入文章")
            return
        if not pipeline_ready:
            messagebox.showwarning("提示", "管線尚未就緒")
            return

        self.run_btn.config(state=tk.DISABLED)
        for lv in [1, 2, 3]:
            self.output_texts[lv].delete("1.0", tk.END)
        self.progress["value"] = 0

        threading.Thread(
            target=self._pipeline_thread, args=(text,), daemon=True
        ).start()

    def _pipeline_thread(self, text):
        sentences = split_sentences(text)
        total = len(sentences)
        results = {1: [], 2: [], 3: []}

        self._log(f"--- 開始鏈式簡化，共 {total} 句 ---")

        for i, sent in enumerate(sentences):
            self._set_status(f"鏈式簡化：第 {i+1}/{total} 句")

            # 鏈式：原文 → L1 → L2 → L3
            l1 = postprocess_sentence(sent, 1)
            l2 = postprocess_sentence(l1,   2)
            l3 = postprocess_sentence(l2,   3)

            results[1].append(l1)
            results[2].append(l2)
            results[3].append(l3)

            self._log(f"[{i+1}/{total}]")
            self._log(f"  原：{sent[:30]}")
            self._log(f"  L1：{l1[:30]}")
            self._log(f"  L2：{l2[:30]}")
            self._log(f"  L3：{l3[:30]}")

            pct = (i + 1) / total * 100
            self.root.after(0, lambda v=pct: self.progress.configure(value=v))

        for level in [1, 2, 3]:
            final = "".join(results[level])
            self.root.after(
                0, lambda lv=level, txt=final: self._set_output(lv, txt)
            )

        self._set_status("✅ 鏈式簡化全部完成！")
        self._log("--- 完成 ---")
        self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.progress.configure(value=100))

    # ── 工具 ──
    def _set_output(self, level, text):
        w = self.output_texts[level]
        w.delete("1.0", tk.END)
        w.insert(tk.END, text)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _log(self, msg):
        def _do():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")]
        )
        if path:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, content)
            self._set_status(f"已載入：{os.path.basename(path)}")

    def _save_output(self, level):
        text = self.output_texts[level].get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("提示", "此等級尚無輸出內容")
            return
        names = {1: "L1", 2: "L2", 3: "L3"}
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"simplified_{names[level]}.txt",
            filetypes=[("文字檔", "*.txt")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._set_status(f"✅ 已儲存：{os.path.basename(path)}")

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SimplificationGUI(root)
    root.mainloop()











