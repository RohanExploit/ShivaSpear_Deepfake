import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

# Load the trained model
MODEL_PATH = "deepfake_audio_model.h5"  # Update if needed
AUDIO_FILE = "test_audio.wav"   # Change to the actual audio file

FIXED_FEATURE_SIZE = 40  # Should match training size

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f" Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(file_path, fixed_size=FIXED_FEATURE_SIZE):
    """Extracts MFCC features from an audio file."""
    try:
        audio, sr = sf.read(file_path)  # Using soundfile instead of librosa
        if len(audio.shape) > 1:  # Convert stereo to mono if necessary
            audio = np.mean(audio, axis=1)

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)  # Resample
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=fixed_size)

        # Ensure uniform shape
        if mfcc.shape[1] < fixed_size:
            pad_width = fixed_size - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > fixed_size:
            mfcc = mfcc[:, :fixed_size]

        return np.mean(mfcc, axis=1).reshape(1, -1)  # Reshape for model input
    except Exception as e:
        print(f"Error extracting features from: {file_path} | Error: {e}")
        return None

# Check if the file exists
if not os.path.exists(AUDIO_FILE):
    raise FileNotFoundError(f" Audio file not found: {AUDIO_FILE}")

# Extract features
features = extract_features(AUDIO_FILE)

if features is not None:
    # Make a prediction
    prediction = model.predict(features)
    label = "Real" if prediction[0] > 0.5 else "Fake"  # Adjust threshold if needed
    confidence = float(prediction[0])  # Convert to float for better readability

    print(f" Prediction Result: {label} (Confidence: {confidence:.2f})")
else:
    print(" Failed to process the audio file.")
