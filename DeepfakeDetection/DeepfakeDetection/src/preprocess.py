import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_images(folder, label):
    images = []
    labels = []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def get_dataset():
    real_path = "../dataset/real/"
    fake_path = "../dataset/fake/"

    real_images, real_labels = load_images(real_path, 0)  # 0 for real
    fake_images, fake_labels = load_images(fake_path, 1)  # 1 for fake

    # Combine data
    X = np.vstack((real_images, fake_images))
    y = np.hstack((real_labels, fake_labels))

    # Normalize images
    X = X / 255.0  

    # Split dataset
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataset()
    print("Dataset Loaded Successfully!")
