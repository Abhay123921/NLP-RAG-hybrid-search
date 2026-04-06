import numpy as np

def normalize(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def log(msg):
    print(f"[INFO] {msg}")