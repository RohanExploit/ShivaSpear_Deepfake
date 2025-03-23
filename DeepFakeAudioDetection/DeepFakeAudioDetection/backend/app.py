import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# Load the trained model
MODEL_PATH = "deepfake_audio_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please train the model first.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# Function to extract features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs = np.mean(mfccs.T, axis=0)  # Convert to fixed shape
        return mfccs
    except Exception as e:
        st.error(f" Error extracting features: {e}")
        return None

# Streamlit UI
st.title(" DeepFake Audio Detection")
st.write("Upload an audio file to check if it is **real or fake**.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save uploaded file temporarily
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features and make prediction
    features = extract_features(temp_audio_path)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Reshape for model input
        prediction = model.predict(features)[0][0]

        # Display results
        if prediction > 0.5:
            st.error("Fake Audio Detected!")
        else:
            st.success(" This is a Real Audio!")
