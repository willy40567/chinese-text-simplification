# few_shot_inference.py
import json, re, torch

from config import BASE_MODEL, TRAINING_DATA as JSONL_PATH
from model_utils import load_base_model

OUTPUT_PATH = "few_shot_results.jsonl"

# ── 3-shot 範例（各級各一組） ──
FEW_SHOT = {
    1: [
        ("蔡倫是東漢時代的官員，他利用樹皮、破布等材料，發明了造紙術。",
         "蔡倫是東漢的官員。他用樹皮和破布做出了紙。"),
        ("南丁格爾從小就喜歡幫助別人，長大後成為一位護士。",
         "南丁格爾小時候喜歡幫助人。她長大後當了護士。"),
        ("鄭成功率領軍隊驅逐荷蘭人，收復台灣。",
         "鄭成功帶兵打走荷蘭人。他收回了台灣。"),
    ],
    2: [
        ("蔡倫是東漢的官員。他用樹皮和破布做出了紙。",
         "蔡倫是官員。他做出了紙。"),
        ("南丁格爾小時候喜歡幫助人。她長大後當了護士。",
         "南丁格爾喜歡幫助人。她當了護士。"),
        ("鄭成功帶兵打走荷蘭人。他收回了台灣。",
         "鄭成功打走了敵人。他收回台灣。"),
    ],
    3: [
        ("蔡倫是官員。他做出了紙。",
         "蔡倫做出了紙。"),
        ("南丁格爾喜歡幫助人。她當了護士。",
         "她當了護士。"),
        ("鄭成功打走了敵人。他收回台灣。",
         "他收回台灣。"),
    ],
}

from config import LEVEL_INSTRUCTION

def build_few_shot_prompt(sent, level):
    instruction = LEVEL_INSTRUCTION[level]
    examples = FEW_SHOT[level]
    msgs = [{"role": "system", "content": "你是一個專業的中文語言學家，負責將複雜的中文句子改寫成簡單的初級中文。"}]
    for src, tgt in examples:
        msgs.append({"role": "user",      "content": f"{instruction}\n原文：{src}\n請只輸出改寫結果。"})
        msgs.append({"role": "assistant", "content": tgt})
    msgs.append({"role": "user", "content": f"{instruction}\n原文：{sent}\n請只輸出改寫結果。"})
    return msgs

def clean_output(raw, sent):
    noise = ["簡化", "答案", "原句", "規則", "注意", "→", "```", "# ", "原文"]
    for n in noise:
        if n in raw:
            raw = raw[:raw.index(n)].strip()
    raw = re.sub(r'[。]{2,}', '。', raw)
    raw = raw.strip()
    if '。' in raw:
        raw = raw.split('。')[0] + '。'
    raw = raw[:60]
    if raw and raw[-1] not in '。！？':
        raw += '。'
    chinese = len(re.findall(r'[\u4e00-\u9fff]', raw))
    total   = len(raw.replace(" ", ""))
    if total == 0 or chinese / total < 0.85:
        return sent  # fallback
    return raw

def main():
    print("載入 base model（無 LoRA）...")
    model, tokenizer = load_base_model()
    print("✅ Base model 載入完成（無 LoRA）")

    results = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    for i, s in enumerate(samples):
        lv  = s["metadata"]["level"]
        inp = s["input"].strip()
        ref = s["output"].strip()

        msgs   = build_few_shot_prompt(inp, lv)
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=120,
                repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        raw = tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        pred = clean_output(raw, inp)

        results.append({
            "level":      lv,
            "input":      inp,
            "reference":  ref,
            "prediction": pred,
        })
        print(f"[{i+1}/{len(samples)}] L{lv} | {pred[:30]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成，輸出至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()