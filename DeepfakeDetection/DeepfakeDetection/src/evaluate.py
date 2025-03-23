# import tensorflow as tf
# from preprocess import get_dataset
# from sklearn.metrics import accuracy_score

# # Load dataset
# X_train, X_test, y_train, y_test = get_dataset()

# # Load trained model
# model = tf.keras.models.load_model("../models/deepfake_detector.h5")

# # Predictions
# y_pred = (model.predict(X_test) > 0.5).astype("int32")

# # Print Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

import tensorflow as tf
from preprocess import get_dataset
from sklearn.metrics import accuracy_score
import sys  # Import sys for flushing output

# Load dataset
print(" Loading dataset...", flush=True)
X_train, X_test, y_train, y_test = get_dataset()
print(f" Dataset loaded: {X_test.shape}, {y_test.shape}", flush=True)

# Load trained model
print(" Loading model...", flush=True)
try:
    model = tf.keras.models.load_model("../models/deepfake_detector.h5")
    print("Model loaded successfully!", flush=True)
except Exception as e:
    print(f" Error loading model: {e}", flush=True)
    sys.exit()

# Predictions
print(" Making predictions...", flush=True)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f" First 10 Predictions: {y_pred[:10]}", flush=True)

# Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy: {accuracy * 100:.2f}%", flush=True)
