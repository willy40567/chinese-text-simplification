# inference_test_v2.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH    = r"F:\download\專題\peft_results\fold_4\best_lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    device_map          = "auto",
    max_memory          = {0: "11GiB", "cpu": "30GiB"},
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

def simplify(instruction, input_text, max_new_tokens=300):
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}\n\n{input_text}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = 0.3,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
    # 只截取 assistant 回應
    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in full:
        response = full.split(marker)[-1]
        response = response.replace("<|eot_id|>", "").strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ============================================================
# 測試案例：使用訓練集外的新段落
# ============================================================
INSTRUCTION_L1 = "請將以下課文簡化為適合閱讀障礙學生閱讀的版本，每句不超過20個字，詞彙使用國小4年級程度。"

# 使用一段訓練集外的真實課文（可替換）
TEST_TEXT = """
二 謹言慎行
人類是群居的動物，人與人相處很容易因為說錯話、做錯事，造成彼此的誤會。自古以來，老師總是叮嚀學生，要學習運用智慧，過謹言慎行的生活。
某一天，哲學家蘇格拉底的學生匆匆忙忙的跑來，想要告訴老師一件事。不過，蘇格拉底先問他：「你要告訴我的事情是真實的嗎？」學生答：「這是我從街上聽來的，我不知道是不是真的。」再問：「那麼，這件事情是善意的嗎？」學生答：「不是。」又問：「既然如此，這件事情是重要的嗎？」學生感到羞愧的說：「不是很重要。」蘇格拉底藉此斷定，那僅是無聊的傳言罷了。因此,如果我們能先認清現況,再決定該如何說話與行動,就不容易被錯誤牽著鼻子走。 每個人都知道謹言慎行的重要性,卻未必能確實做到。至聖先師孔子曾說過,有仁德的人懂得謹言慎行。學生司馬牛追問,這樣做就可稱作「仁」嗎?孔子回答說:「實際上這是很難做到的,因此須時常提醒自己,在生活裡實踐。」
俗話說得好：「謹慎是智慧的泉源」，一個行為正直、有智慧的人，必定能了解事情全貌後，再做出判斷，不會隨便說話、衝動行事。我們若能謹言慎行，必能創造順遂的人生。
"""

print("=" * 50)
print("【原文】")
print(TEST_TEXT.strip())
print()
print("【L1 簡化結果（≤20字/句）】")
result = simplify(INSTRUCTION_L1, TEST_TEXT.strip())
print(result)
print("=" * 50)