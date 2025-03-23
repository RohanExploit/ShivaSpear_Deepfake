# import os
# import librosa
# import numpy as np

# DATASET_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "dataset"))
# OUTPUT_FILE = "features.npy"
# FIXED_FEATURE_SIZE = 40  # Ensure all features are of the same size

# def extract_features(file_path, fixed_size=FIXED_FEATURE_SIZE):
#     try:
#         audio, sr = librosa.load(file_path, sr=16000)
#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=fixed_size)

#         # Ensure uniform shape
#         if mfcc.shape[1] < fixed_size:  
#             pad_width = fixed_size - mfcc.shape[1]
#             mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
#         elif mfcc.shape[1] > fixed_size:  
#             mfcc = mfcc[:, :fixed_size]

#         return np.mean(mfcc, axis=1)  # Return a fixed-size feature vector
#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")
#         return None

# X, y = [], []

# for label, folder in enumerate(["real", "fake"]):
#     folder_path = os.path.join(DATASET_PATH, folder)

#     if not os.path.exists(folder_path):
#         print(f"Warning: Folder not found -> {folder_path}")
#         continue

#     for file in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file)
#         features = extract_features(file_path)
#         if features is not None:
#             X.append(features)
#             y.append(label)

# # Convert X and y into proper NumPy arrays
# X = np.array(X, dtype=np.float32)  
# y = np.array(y, dtype=np.int32)  # Ensure y is an integer array

# # ✅ Debugging output to check shape before saving
# print(f"✅ Extracted features shape: {X.shape}")
# print(f"✅ Labels shape: {y.shape}")

# # Save the arrays
# np.save(OUTPUT_FILE, (X, y))
# print("Feature extraction completed and saved!")

import os
import librosa
import numpy as np

# Define dataset and output paths
DATASET_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "dataset"))
FEATURES_FILE = "features.npy"
LABELS_FILE = "labels.npy"
FIXED_FEATURE_SIZE = 40  # Number of MFCC features

def extract_features(file_path, fixed_size=FIXED_FEATURE_SIZE):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=fixed_size)

        # Ensure uniform shape
        if mfcc.shape[1] < fixed_size:  # Pad shorter ones
            pad_width = fixed_size - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > fixed_size:  # Truncate longer ones
            mfcc = mfcc[:, :fixed_size]

        return np.mean(mfcc, axis=1)  # Take mean across time
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

X, y = [], []

# Ensure dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path not found -> {DATASET_PATH}")
    exit(1)

# Process each class folder (real and fake)
for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(DATASET_PATH, folder)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found -> {folder_path}")
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X, dtype=np.float32)  # Ensure correct dtype
y = np.array(y)

# Save features and labels separately
np.save(FEATURES_FILE, X)
np.save(LABELS_FILE, y)

print(f"Feature extraction completed and saved!")
print(f"Extracted features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
