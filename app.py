# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os
from PIL import Image

# --------------------------
# 1. Download Model from Google Drive
# --------------------------
MODEL_PATH = "pneumonia_model.h5"
GDRIVE_LINK = "https://drive.google.com/uc?id=1t5UlwpWWeIniy_WazLj6XnWtZ8cdbnlk"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    gdown.download(GDRIVE_LINK, MODEL_PATH, quiet=False)

# --------------------------
# 2. Load Model
# --------------------------
@st.cache_resource
def load_pneumonia_model():
    model = load_model(MODEL_PATH)
    return model

model = load_pneumonia_model()

# --------------------------
# 3. Streamlit App Layout
# --------------------------
st.title("🧠 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and get a prediction: Normal or Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    IMG_SIZE = (150, 150)  # adjust according to your trained model
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if your model was trained on normalized images

    # Predict
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        st.error(f"Pneumonia Detected! ({prediction[0][0]*100:.2f}% confidence)")
    else:
        st.success(f"Normal Lung X-ray! ({(1-prediction[0][0])*100:.2f}% confidence)")
    
   
       
    
