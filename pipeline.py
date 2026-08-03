#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推論核心模組 (pipeline.py)

把原本散落在 simplification_gui.py 的三級簡化推論邏輯抽出，去除 tkinter 依賴，
供 FastAPI / Gradio 等服務層共用。

- 純文字處理函式為模組層級函式（不需模型，可單獨測試）。
- 模型相關狀態 (model / tokenizer / hsk_dict / semantic_model) 收進
  SimplificationPipeline，啟動時 load() 一次、重複使用。
- 對外主要介面：SimplificationPipeline.simplify_text(text) -> {"l1","l2","l3"}

行為與 simplification_gui.py 的鏈式簡化保持一致：
  原文 → L1 → L2 → L3（逐句鏈接，再以 "".join 合併）。
"""

import re

from config import SBERT_MODEL, LEVEL_LIMITS, LEVEL_INSTRUCTION
from hsk_utils import (
    get_tw2sp, get_sp2tw,
    build_hsk_vocab_txt, load_hsk_dict,
    hsk_replace, register_proper_nouns_jieba,
)

# ==========================================
# 常數
# ==========================================
SUSPICIOUS_PLACES = {
    "伦敦", "北京", "上海", "美国", "英国", "日本",
    "法国", "德国", "俄罗斯", "印度", "荷兰人"
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
RE_CHINESE = re.compile(r'[一-鿿]')
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
# 純文字處理函式（不需模型，可單獨測試）
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


def is_valid_chinese_output(text):
    """檢查輸出是否為有效中文，排除幻覺/雜訊。"""
    if not text or len(text) < 2:
        return False
    # 英文字母佔比超過 15% → 無效
    chinese_chars = len(re.findall(r'[一-鿿]', text))
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
    """偵測輸出是否憑空冒出原文沒有的地名（簡體比對）。"""
    import jieba
    tw2sp = get_tw2sp()
    sp2tw = get_sp2tw()
    orig_simp = tw2sp.convert(original)
    gen_words = [w for w in jieba.cut(tw2sp.convert(generated)) if len(w) >= 2]
    for w in gen_words:
        if w in SUSPICIOUS_PLACES and w not in orig_simp:
            return True, sp2tw.convert(w)
    return False, None


def _sbert_max_similarity(pred_sent, src_sents, semantic_model):
    """回傳 pred_sent 與 source 句群的最高 cosine 相似度；無模型時回傳 None。"""
    import torch

    if semantic_model is None:
        return None
    if not pred_sent or not src_sents:
        return None

    texts = [pred_sent] + src_sents
    emb = semantic_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    sims = torch.matmul(emb[0], emb[1:].T)
    return float(torch.max(sims).item())


def similarity_filter(source, prediction, semantic_model=None, threshold=0.45):
    """移除與原文語意無關的幻覺句。優先用 Sentence-BERT，失敗時回退字元相似度。"""
    pred_sents = [s.strip() for s in re.split(r'[。！？]', prediction) if s.strip()]
    src_sents = [s.strip() for s in re.split(r'[。！？\n，]', source) if s.strip()]
    if not src_sents:
        return prediction

    clean = []
    for ps in pred_sents:
        if '「' in ps or '」' in ps:
            clean.append(ps)
            continue

        max_sim = _sbert_max_similarity(ps, src_sents, semantic_model)

        if max_sim is None:
            overlap = [len(set(ps) & set(ss)) / max(1, len(set(ps) | set(ss))) for ss in src_sents]
            max_sim = max(overlap) if overlap else 0.0

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


def split_sentences(text):
    """以句末標點斷句；過長 (>40 字) 者再以逗號/頓號切。"""
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
# 推論管線（持有模型狀態）
# ==========================================
class SimplificationPipeline:
    """載入一次模型，重複用於三級鏈式簡化。"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.hsk_dict = None
        self.semantic_model = None
        self.ready = False

    # ---------- 載入 ----------
    def load(self, log=print):
        """載入 HSK 字典、Jieba、Llama+LoRA、Sentence-BERT。約需 30–60 秒。"""
        import jieba

        log("初始化 OpenCC 轉換器...")
        get_tw2sp()
        get_sp2tw()

        log("建立 HSK 字典...")
        build_hsk_vocab_txt()
        self.hsk_dict = load_hsk_dict()
        log(f"✅ HSK 字典載入，共 {len(self.hsk_dict)} 詞")

        log("設定 Jieba 斷詞...")
        register_proper_nouns_jieba()
        jieba.suggest_freq(('护士', '学校'), True)
        jieba.suggest_freq(('第一', '所'), True)
        log("✅ Jieba 設定完成")

        log("載入基底模型 + LoRA（4-bit 量化，約需 30 秒）...")
        from model_utils import load_finetuned_model
        self.model, self.tokenizer = load_finetuned_model()

        log("載入 Sentence-BERT 語意過濾器...")
        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer(SBERT_MODEL)
            log("✅ Sentence-BERT 載入完成")
        except Exception as sem_e:
            self.semantic_model = None
            log(f"⚠️ Sentence-BERT 載入失敗，改用回退相似度：{sem_e}")

        self.ready = True
        log("✅ 模型載入完成，管線就緒！")
        return self

    # ---------- 單句推論 ----------
    def _lora_generate(self, sent, level):
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

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"""{instruction}
原文：{sent}
請只輸出改寫結果。"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        raw = self.tokenizer.decode(
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
        raw = similarity_filter(sent, raw, self.semantic_model)
        if not raw or not is_valid_chinese_output(raw):
            return sent

        return raw

    def _postprocess_sentence(self, sent, level, max_retry=3):
        if is_heading_like(sent):
            return apply_text_corrections(sent)

        char_limit = LEVEL_LIMITS[level]
        result = sent
        for attempt in range(max_retry):
            result = hsk_replace(result, self.hsk_dict, level)
            clean = "".join(c for c in result if c not in "。，、！？：；「」")
            if len(clean) <= char_limit:
                break
            if attempt < max_retry - 1 and self.model is not None:
                new_result = self._lora_generate(sent, level)
                if is_valid_chinese_output(new_result):
                    result = new_result
        result = apply_text_corrections(result)
        return result

    # ---------- 對外主介面 ----------
    def simplify_text(self, text):
        """將整段文字做三級鏈式簡化，回傳 {"l1","l2","l3"}。"""
        if not self.ready:
            raise RuntimeError("管線尚未就緒，請先呼叫 load()")

        sentences = split_sentences(text)
        results = {1: [], 2: [], 3: []}

        for sent in sentences:
            l1 = self._postprocess_sentence(sent, 1)
            l2 = self._postprocess_sentence(l1, 2)
            l3 = self._postprocess_sentence(l2, 3)
            results[1].append(l1)
            results[2].append(l2)
            results[3].append(l3)

        return {
            "l1": "".join(results[1]),
            "l2": "".join(results[2]),
            "l3": "".join(results[3]),
        }
