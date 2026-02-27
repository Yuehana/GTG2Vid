import os
import json
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

"""
Generate text features for ChatGPT-4o-mini error descriptions.

Input:
  chatgpt4omini_error.txt, each line:
    <name> <description_with_or_without_underscores>
  e.g.
    action_1_type_1_1 Skipping_step
    action_2_type_1_1 Pushing_blade_up_instead_of_retracting

Output:
  For each line, save a 768-dim BERT feature to:
    vc_chatgpt4omini_error_features/<name>.npy
"""

# -------------------------------------------------------------------------
# 路径设置（根据你现在的项目结构）
# -------------------------------------------------------------------------
ROOT_DATA_DIR = "/mnt/22TB_IndEgo//GTG2Vid/data/IndEgo"
DATASET_NAME = "IndEgo_vc_T05"

ERROR_TXT_NAME = "chatgpt4omini_error.txt"   # 文件名（不带路径）
ERROR_TXT_PATH = os.path.join(ROOT_DATA_DIR, DATASET_NAME, ERROR_TXT_NAME)

SAVE_DIR = os.path.join(ROOT_DATA_DIR, DATASET_NAME, "vc_chatgpt4omini_error_features")
os.makedirs(SAVE_DIR, exist_ok=True)

# 使用 BERT-base-uncased，768 维输出
MODEL_NAME = "bert-base-uncased"

# -------------------------------------------------------------------------
# 加载文本编码模型
# -------------------------------------------------------------------------
print(f"Loading text encoder: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device
print("Model device:", device)

# -------------------------------------------------------------------------
# 读取错误描述列表
# -------------------------------------------------------------------------
if not os.path.exists(ERROR_TXT_PATH):
    raise FileNotFoundError(f"Error list file not found: {ERROR_TXT_PATH}")

print(f"Reading error descriptions from: {ERROR_TXT_PATH}")
with open(ERROR_TXT_PATH, "r") as fp:
    lines = [l.strip() for l in fp.readlines() if l.strip()]

print(f"Found {len(lines)} error descriptions.\n")

# -------------------------------------------------------------------------
# 编码每条错误描述并保存为 .npy
# -------------------------------------------------------------------------
def encode_text(text: str) -> np.ndarray:
    """Encode text to a 768-dim feature using BERT (mean pooling)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # last_hidden_state: [1, seq_len, 768]
        last_hidden = outputs.last_hidden_state  # [1, L, 768]
        feature = last_hidden.mean(dim=1).squeeze(0)  # [768]
        return feature.cpu().numpy()

print("Generating error features...")
for line in tqdm(lines):
    # 只按第一个空格切分：name + 描述
    if " " not in line:
        print(f"[WARN] invalid line (no space): '{line}'")
        continue

    name, desc = line.split(" ", 1)
    # 把下划线变成空格，变成自然语言一点
    desc = desc.replace("_", " ").strip()

    save_path = os.path.join(SAVE_DIR, name + ".npy")
    if os.path.exists(save_path):
        # 已经存在就跳过（避免重复计算）
        # 如果你想覆盖，就删掉这个 if
        # print(f"[SKIP] {save_path} already exists.")
        pass

    feat = encode_text(desc)  # shape: (768,)
    np.save(save_path, feat)

    print(f"[SAVE] {name}: dim={feat.shape[0]}, path={save_path}")

print("\nDone.")
print("ChatGPT-4o-mini error features saved to:", SAVE_DIR)
