from datasets import load_dataset

# 這裡使用您電腦上的絕對路徑，請確認路徑是否正確（注意斜線方向）
base_path = "F:/download/專題/mcts-main"

print("正在讀取資料集...")

# 1. 讀取 69 萬筆的偽資料訓練集 (Pseudo Data)
# zh_selected.ori 是複雜句 (原始句)，zh_selected.sim 是對應的簡化句
train_complex_ds = load_dataset("text", data_files=f"{base_path}/pseudo_data/zh_selected.ori", split="train")
train_simple_ds = load_dataset("text", data_files=f"{base_path}/pseudo_data/zh_selected.sim", split="train")

print("\n--- 訓練集 (複雜句) ---")
print(train_complex_ds)
print("第一筆資料預覽:", train_complex_ds[0])

print("\n--- 訓練集 (簡化句) ---")
print("第一筆資料預覽:", train_simple_ds[0])

# --------------------------------------------------

# 2. 讀取測試集 (Test Data)
# mcts.test.orig 是原始複雜句
test_orig_ds = load_dataset("text", data_files=f"{base_path}/dataset/mcts.test.orig", split="train")

# mcts.test.simp.1 到 .5 是五個人工撰寫的參考簡化句，這裡以第 1 個為例
test_simp1_ds = load_dataset("text", data_files=f"{base_path}/dataset/mcts.test.simp.1", split="train")

print("\n--- 測試集 (原始複雜句) ---")
print(test_orig_ds)
print("第一筆資料預覽:", test_orig_ds[0])

print("\n--- 測試集 (參考簡化句 1) ---")
print("第一筆資料預覽:", test_simp1_ds[0])