import numpy as np

path = "/home/yuehanzhang/GTG2Vid/data/IndEgo/IndEgo_vc_T04/vc_v_features_10fps/04_u01_r06_c_01_480.npy"

arr = np.load(path)
print("Shape:", arr.shape)

if arr.ndim == 2:
    T, D = arr.shape
    print(f"Frames (time steps): {T}")
    print(f"Feature dimension: {D}")
else:
    print("Unexpected .npy shape:", arr.shape)
    raise ValueError("Expected a 2D array in the .npy file.")