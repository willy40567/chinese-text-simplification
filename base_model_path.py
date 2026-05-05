import os

# 如果你是用 Transformers 庫
from transformers import AutoModel
try:
    # 這裡填入你程式碼中使用的模型變數名稱
    print(model.config._name_or_path)
except:
    print("無法從變數取得路徑，請檢查程式碼中的 model_id 或 path 字串")