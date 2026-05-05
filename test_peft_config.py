import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import sys

def test_peft_environment():
    """驗證PEFT環境與LoRA配置"""

    print("=" * 70)
    print("🔍 PEFT環境驗證開始")
    print("=" * 70)

    # 1. GPU狀態檢查
    print(f"\n【1/5】GPU硬體檢測")
    print("-" * 70)
    if torch.cuda.is_available():
        print(f"✅ GPU可用")
        print(f"   - 裝置名稱：{torch.cuda.get_device_name(0)}")
        print(f"   - 顯存總量：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   - CUDA版本：{torch.version.cuda}")

        # 清空顯存
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"   - 可用顯存：{free_mem:.2f} GB")
    else:
        print("⚠️ 未檢測到GPU，將使用CPU訓練（不建議，速度極慢）")
        response = input("\n是否繼續？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # 2. 套件版本檢查
    print(f"\n【2/5】關鍵套件版本")
    print("-" * 70)
    import transformers
    import peft
    import datasets

    print(f"✅ transformers: {transformers.__version__}")
    print(f"✅ peft: {peft.__version__}")
    print(f"✅ datasets: {datasets.__version__}")
    print(f"✅ torch: {torch.__version__}")

    # 3. 模型載入測試
    print(f"\n【3/5】Llama3.1-8B模型載入")
    print("-" * 70)
    MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        print("載入Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer載入成功")

        print("載入模型（FP16格式）...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✅ 模型載入成功")

        # 顯示模型大小
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   - 模型參數量：{param_count:,} ({param_count/1e9:.2f}B)")

    except Exception as e:
        print(f"❌ 模型載入失敗：{e}")
        print(f"\n💡 請確認模型路徑：{MODEL_PATH}")
        sys.exit(1)

    # 4. LoRA配置測試
    print(f"\n【4/5】LoRA參數配置")
    print("-" * 70)

    lora_config = LoraConfig(
        r=16,                          # LoRA秩（影響可訓練參數量）
        lora_alpha=32,                 # 縮放因子（alpha/r = 學習率縮放）
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 目標注意力層
        lora_dropout=0.05,             # Dropout率（防過擬合）
        bias="none",                   # 不訓練bias
        task_type="CAUSAL_LM"          # 因果語言模型
    )

    print("LoRA配置參數：")
    print(f"   - 秩 (r)：{lora_config.r}")
    print(f"   - Alpha：{lora_config.lora_alpha}")
    print(f"   - 目標層：{lora_config.target_modules}")
    print(f"   - Dropout：{lora_config.lora_dropout}")

    try:
        print("\n應用LoRA到模型...")
        peft_model = get_peft_model(model, lora_config)
        print("✅ LoRA配置成功")

        # 顯示可訓練參數統計
        print("\n參數統計：")
        peft_model.print_trainable_parameters()

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())

        print(f"   - 可訓練參數：{trainable_params:,} 個")
        print(f"   - 總參數量：{total_params:,} 個")
        print(f"   - 訓練比例：{trainable_params/total_params*100:.2f}%")
        print(f"   - 節省記憶體：約 {(1-trainable_params/total_params)*100:.1f}%")

    except Exception as e:
        print(f"❌ LoRA配置失敗：{e}")
        sys.exit(1)

    # 5. 訓練資料檢查
    print(f"\n【5/5】訓練資料驗證")
    print("-" * 70)

    from datasets import load_dataset

    try:
        dataset = load_dataset(
            'json',
            data_files='F:/download/專題/training_data.jsonl',
            split='train'
        )
        print(f"✅ 訓練資料載入成功")
        print(f"   - 樣本數量：{len(dataset)} 組")
        print(f"   - 欄位名稱：{dataset.column_names}")

        # 顯示第一筆樣本
        print(f"\n樣本範例（第1組）：")
        example = dataset[0]
        print(f"   - 指令長度：{len(example['instruction'])} 字元")
        print(f"   - 回應長度：{len(example['response'])} 字元")
        print(f"   - 類別：{example['metadata']['category']}")
        print(f"   - 難度級別：{example['metadata']['level']}")

    except Exception as e:
        print(f"❌ 資料載入失敗：{e}")
        sys.exit(1)

    # 最終總結
    print("\n" + "=" * 70)
    print("✅ 環境驗證完成！所有檢查通過")
    print("=" * 70)
    print("\n📌 系統配置摘要：")
    print(f"   - GPU：RTX 4070 12GB（可用顯存 {free_mem:.1f} GB）")
    print(f"   - 模型：Llama3.1-8B（{param_count/1e9:.2f}B參數）")
    print(f"   - LoRA可訓練參數：{trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   - 訓練資料：{len(dataset)} 組樣本")
    print(f"\n🎯 下一步：執行 train_peft.py 開始微調訓練")
    print("=" * 70)

if __name__ == "__main__":
    test_peft_environment()