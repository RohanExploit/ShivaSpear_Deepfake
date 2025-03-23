import numpy as np

try:
    X, y = np.load("features.npy", allow_pickle=True)
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
except Exception as e:
    print("Error loading features.npy:", e)
