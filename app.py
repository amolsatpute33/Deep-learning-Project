import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import gdown
import os
from PIL import Image

# Google Drive file ID
file_id = "1t5UlwpWWeIniy_WazLj6XnWtZ8cdbnlk"
url = f"https://drive.google.com/uc?id={file_id}"

model_path = "pneumonia_detection_model.h5"

# Download model if not exists
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

st.title("Chest X-ray Pneumonia Detection")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((224,224))
    
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,224,224,1)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Pneumonia Detected")
    else:
        st.success("Normal Chest X-ray")
 
