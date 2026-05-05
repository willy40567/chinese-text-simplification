import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "F:/download/專題/training_data.jsonl"
OUTPUT_DIR = "F:/download/專題/lora_weights"

print("🚀 載入模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 改用BF16
    low_cpu_mem_usage=True
)
model = model.to("cuda")

print("🔧 配置LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("📊 載入資料...")
dataset = load_dataset('json', data_files=DATA_PATH, split='train')

def preprocess(examples):
    texts = [f"{inst}\n\n{resp}" for inst, resp in zip(examples['instruction'], examples['response'])]
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

print("🏃 開始訓練...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,  # 使用BF16
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()

print("💾 儲存模型...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 完成！模型已儲存至：{OUTPUT_DIR}")