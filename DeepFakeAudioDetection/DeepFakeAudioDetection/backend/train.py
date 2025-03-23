import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load extracted features and labels
FEATURES_FILE = "features.npy"
LABELS_FILE = "labels.npy"

X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)

# Normalize features
X = X / np.max(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification (real vs fake)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save("deepfake_audio_model.h5")

print("Training completed! Model saved as deepfake_audio_model.h5")
