import os
import json
import numpy as np
from collections import Counter
from glob import glob

########################################
# Settings
########################################
JSON_DIR   = "/mnt/logicNAS/Exchange/yuehan/IndEgoMD/ego_480/task_05/A_Task_05"
OUT_DIR    = "/mnt/22TB_IndEgo/GTG2Vid/data/IndEgo/IndEgo_vc_T05/refined_label_v3"

FPS        = 10

# 输出模式：
# "frame" = 每帧一行（推荐：与你示例一致）
# "clip"  = 每个滑窗clip一行（chunk/stride控制）
MODE       = "frame"

# clip 模式参数（仅 MODE="clip" 时用）
CHUNK      = 8
STRIDE     = 4
LABEL_MODE = "center"     # "center" or "majority"
MIN_PURITY = 0.6          # only majority uses
BG_TOKEN   = "BG"         # 背景动作名
STATE      = "Normal"     # 你要的第二列固定值

# ✅ 新增：文件名格式
OUT_PREFIX = "A_"
OUT_SUFFIX = "_labels"
########################################


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_action_name_map(data):
    attr = data.get("attribute", {})
    if len(attr) == 0:
        return {}
    first_key = sorted(attr.keys())[0]
    options = attr[first_key].get("options", {})
    return {int(k): v for k, v in options.items()}


def load_segments(data):
    segs = []
    for _, v in data.get("metadata", {}).items():
        start, end = v["z"]
        action_id = int(v["av"]["1"])
        segs.append((float(start), float(end), action_id))
    segs.sort(key=lambda x: x[0])
    return segs


def label_time(t, segments):
    for (s, e, a) in segments:
        if s <= t < e:
            return a
    return -1


def infer_duration_sec(data):
    segs = load_segments(data)
    if not segs:
        return 0.0
    return max(e for (_, e, _) in segs)


def frame_labels(data, fps=10):
    segs = load_segments(data)
    dur = infer_duration_sec(data)
    n_frames = int(np.ceil(dur * fps))
    labels = np.full(n_frames, -1, dtype=int)
    for i in range(n_frames):
        t = i / fps
        labels[i] = label_time(t, segs)
    return labels


def clip_labels_from_frame_labels(frame_lab, fps=10, chunk=8, stride=4,
                                  label_mode="center", min_purity=0.6):
    T = len(frame_lab)
    out = []
    for s in range(0, T - chunk + 1, stride):
        e = s + chunk
        lab_clip = frame_lab[s:e]

        if label_mode == "center":
            out.append(int(lab_clip[chunk // 2]))
        else:
            valid = lab_clip[lab_clip != -1]
            if len(valid) == 0:
                out.append(-1)
            else:
                top, cnt = Counter(valid.tolist()).most_common(1)[0]
                purity = cnt / len(valid)
                out.append(int(top) if purity >= min_purity else -1)

    return np.array(out, dtype=int)


def write_txt(lines, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def make_out_path(base, mode, chunk=None, stride=None):
    """
    base: 05_u01_r02_c_01
    输出:
      frame: A_05_u01_r02_c_01_labels.txt
      clip : A_05_u01_r02_c_01_chunk8_stride4_labels.txt  (可选保留参数，方便区分)
    """
    if mode == "frame":
        fname = f"{OUT_PREFIX}{base}{OUT_SUFFIX}.txt"
    else:
        # 你如果不想把 chunk/stride 放进文件名，把下面这一行改成跟 frame 一样即可
        fname = f"{OUT_PREFIX}{base}_chunk{chunk}_stride{stride}{OUT_SUFFIX}.txt"
    return os.path.join(OUT_DIR, fname)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    json_files = sorted(glob(os.path.join(JSON_DIR, "*.json")))
    print("Found json:", len(json_files))

    for jp in json_files:
        data = load_json(jp)
        name_map = get_action_name_map(data)
        base = os.path.splitext(os.path.basename(jp))[0]  # e.g. 05_u01_r02_c_01

        if MODE == "frame":
            flab = frame_labels(data, fps=FPS)
            lines = []
            for a in flab:
                act = BG_TOKEN if a == -1 else name_map.get(int(a), str(int(a)))
                lines.append(f"{act}|{STATE}")

            out_path = make_out_path(base, mode="frame")
            write_txt(lines, out_path)

        elif MODE == "clip":
            flab = frame_labels(data, fps=FPS)
            clab = clip_labels_from_frame_labels(
                flab, fps=FPS, chunk=CHUNK, stride=STRIDE,
                label_mode=LABEL_MODE, min_purity=MIN_PURITY
            )
            lines = []
            for a in clab:
                act = BG_TOKEN if a == -1 else name_map.get(int(a), str(int(a)))
                lines.append(f"{act}|{STATE}")

            out_path = make_out_path(base, mode="clip", chunk=CHUNK, stride=STRIDE)
            write_txt(lines, out_path)

        else:
            raise ValueError("MODE must be 'frame' or 'clip'")

    print("Done. Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
