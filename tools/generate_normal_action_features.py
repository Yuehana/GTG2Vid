import os
import json
import numpy as np
from tqdm import tqdm

"""
Generate normal-action prototypes for IndEgo_vc_T04.

For each action (1..N, excluding BG=0), this script:
1. Reads frame-wise labels from refined_label_v3/A_*.txt (format: "action_name|state")
2. Loads corresponding VideoCLIP features from vc_v_features_10fps/*.npy
3. Collects all frames whose label == action_idx
4. Averages them to get a prototype feature vector
5. Saves to vc_normal_action_features/Action_X.npy
"""

# -----------------------------------------------------------------------------
# PATH SETTINGS
# -----------------------------------------------------------------------------
ROOT_DATA_DIR = "/mnt/22TB_IndEgo/GTG2Vid/data/IndEgo"
DATASET_NAME = "IndEgo_vc_T05"

LABEL_DIR = os.path.join(ROOT_DATA_DIR, DATASET_NAME, "refined_label_v3")
VFEAT_DIR = os.path.join(ROOT_DATA_DIR, DATASET_NAME, "features_10fps")

# Your feature files are like: 04_u01_r06_c_01_480.npy
FEATURE_SUFFIX = "_480"

SAVE_DIR = os.path.join(ROOT_DATA_DIR, DATASET_NAME, "vc_normal_action_features")
os.makedirs(SAVE_DIR, exist_ok=True)

ACTION2IDX_PATH = os.path.join(ROOT_DATA_DIR, "action2idx.json")

# -----------------------------------------------------------------------------
# Load action2idx mapping
# -----------------------------------------------------------------------------
print("Loading action2idx:", ACTION2IDX_PATH)
with open(ACTION2IDX_PATH, "r") as fp:
    action2idx_all = json.load(fp)

# 兼容两种情况：有顶层键 "IndEgo_vc_T04"，或者直接就是字典
if "IndEgo_vc_T05" in action2idx_all:
    action2idx = action2idx_all["IndEgo_vc_T05"]
else:
    action2idx = action2idx_all

print("action2idx:", action2idx)

# Build reverse mapping for real actions (exclude BG=0)
normal_action_name_map = {
    idx: f"Action_{idx}"
    for name, idx in action2idx.items()
    if idx != 0
}
print("normal_action_name_map:", normal_action_name_map)

# -----------------------------------------------------------------------------
# Collect features for each action
# -----------------------------------------------------------------------------
all_label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith("_labels.txt")]
print(f"Found {len(all_label_files)} label files.\n")

# Initialize container: action_idx -> list of feature arrays
action_features = {idx: [] for idx in normal_action_name_map.keys()}

print("Collecting normal frame features...")
for lf in tqdm(all_label_files):
    # Example lf: A_04_u01_r06_c_01_labels.txt
    label_path = os.path.join(LABEL_DIR, lf)

    video_id = lf.replace("_labels.txt", "")  # A_04_u01_r06_c_01
    feat_id = video_id.replace("A_", "")      # 04_u01_r06_c_01

    feat_path = os.path.join(VFEAT_DIR, feat_id + FEATURE_SUFFIX + ".npy")

    if not os.path.exists(feat_path):
        print(f"[WARN] feature file not found for {video_id}: {feat_path}")
        continue

    # Load feature array: shape (T, D)
    feats = np.load(feat_path)  # e.g. (T, 768) or (T, 256)
    T_feat = feats.shape[0]

    # Load frame-wise labels (format: "action_name|state" or "action_name|state|something")
    with open(label_path, "r") as fp:
        lines = fp.readlines()

    frame_labels = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        action_name = parts[0].strip()  # e.g. "BG" or "put on gloves"

        if action_name not in action2idx:
            print(f"[WARN] Unknown action name in label file {lf}: '{action_name}'")
            # 可以选择跳过，也可以继续，这里我们选择跳过这个 frame
            continue

        label_idx = action2idx[action_name]
        frame_labels.append(label_idx)

    frame_labels = np.array(frame_labels, dtype=int)
    T_label = frame_labels.shape[0]

    if T_label == 0:
        print(f"[WARN] No valid labels parsed for {lf}, skip.")
        continue

    # Align length
    T = min(T_feat, T_label)
    if T_feat != T_label:
        print(f"[INFO] {feat_id}: feature frames = {T_feat}, label frames = {T_label} -> use {T}")
    feats = feats[:T]
    frame_labels = frame_labels[:T]

    # For each action, collect normal frames (这里只看动作类别，不看 Normal / Error 状态)
    for action_idx in normal_action_name_map.keys():
        mask = (frame_labels == action_idx)
        if mask.any():
            action_features[action_idx].append(feats[mask])

# -----------------------------------------------------------------------------
# Save per-action prototype
# -----------------------------------------------------------------------------
print("\nSaving normal-action prototypes...")
for action_idx, feat_list in action_features.items():
    action_name = normal_action_name_map[action_idx]

    if len(feat_list) == 0:
        print(f"[WARN] No normal frames collected for action_idx={action_idx} ({action_name})")
        continue

    # Concatenate all frames of this action across all videos
    all_feats = np.concatenate(feat_list, axis=0)  # (N_frames, D)
    proto = all_feats.mean(axis=0)                 # (D,)

    save_path = os.path.join(SAVE_DIR, f"{action_name}.npy")
    np.save(save_path, proto)

    print(f"[SAVE] {action_name}: feature_dim={proto.shape[0]}, path={save_path}")

print("\nDone.")
print("Normal action feature prototypes saved to:", SAVE_DIR)
