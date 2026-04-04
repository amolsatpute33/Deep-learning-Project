import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import gdown
import os
from PIL import Image

st.set_page_config(page_title="🧠 Pneumonia Detection", layout="centered")
st.title("Chest X‑ray Pneumonia Detection")

# -----------------------------
# 1️⃣ Model Download
# -----------------------------
MODEL_FILE = "pneumonia_detection_model.h5"
DRIVE_ID = "1t5UlwpWWeIniy_WazLj6XnWtZ8cdbnlk"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

if not os.path.exists(MODEL_FILE):
    st.info("Downloading model from Google Drive…")
    gdown.download(DRIVE_URL, MODEL_FILE, quiet=False)
    st.success("Model downloaded successfully!")

# -----------------------------
# 2️⃣ Load Model
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_FILE)

model = load_cnn_model()

# -----------------------------
# 3️⃣ Upload Input
# -----------------------------
uploaded_file = st.file_uploader("Upload Chest X‑ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load & display image
        img = Image.open(uploaded_file).convert("L")  # grayscale
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 1)

        # Predict
        pred = model.predict(img_array)[0][0]

        st.write("### Prediction Result")
        if pred > 0.5:
            st.error(f"🚨 Pneumonia Detected ({pred*100:.2f}% confidence)")
        else:
            st.success(f"✅ Normal Chest X‑ray ({(1‑pred)*100:.2f}% confidence)")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
else:
    st.write("Upload an X‑ray image to get a prediction.")
