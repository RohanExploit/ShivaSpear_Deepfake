
#<------ this is working project------->
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image

# # Load trained model
# model = tf.keras.models.load_model("../models/deepfake_detector.h5")

# # Streamlit UI
# st.title("Deepfake Image Detector")
# st.write("Upload an image to check whether it is Real or Fake.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# # Function to preprocess image
# def preprocess_image(image):
#     image = np.array(image)
#     image = cv2.resize(image, (128, 128))  # Resize to match training data
#     image = image / 255.0  # Normalize
#     return np.expand_dims(image, axis=0)  # Add batch dimension

# # Prediction
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image
#     processed_image = preprocess_image(image)

#     # Get prediction
#     prediction = model.predict(processed_image)[0][0]

#     # Display result
#     if prediction > 0.5:
#         st.error("This image is likely FAKE! ")
#     else:
#         st.success("This image appears to be REAL ")




#<---working---->
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("../models/deepfake_detector.h5")

# Streamlit UI
st.set_page_config(page_title="Deepfake Image Detector", layout="centered")

st.title(" Deepfake Image Detector")
st.write("Upload an image to check whether it is **Real or Fake**.")

# File uploader
uploaded_file = st.file_uploader(" Upload an image...", type=["jpg", "png", "jpeg"])

# # Function to preprocess image
# # def preprocess_image(image):
# #     image = np.array(image)
# #     image = cv2.resize(image, (128, 128))  # Resize to match training data
# #     image = image / 255.0  # Normalize
# #     return np.expand_dims(image, axis=0)  # Add batch dimension

# #<----   working code  ------>
# def preprocess_image(image):
#     image = np.array(image)

#     # Ensure the image is in RGB format
#     if len(image.shape) == 2:  # If grayscale, convert to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     elif image.shape[-1] == 4:  # If RGBA, convert to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

#     # Resize to match model input
#     image = cv2.resize(image, (128, 128))  
#     image = image / 255.0  # Normalize
    
#     return np.expand_dims(image, axis=0)  # Add batch dimension

def preprocess_image(image):
    image = np.array(image)

    # Ensure image has 3 channels
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:  # If RGBA, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[-1] != 3:  # If image is not RGB, force conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to match model input
    image = cv2.resize(image, (128, 128))  
    image = image / 255.0  # Normalize
    
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Get prediction
    prediction = model.predict(processed_image)[0][0]

    # Display result
    st.markdown("---")
    if prediction > 0.5:
        st.error(" **This image is likely FAKE!**")
    else:
        st.success(" **This image appears to be REAL.**")
    st.markdown("---")

