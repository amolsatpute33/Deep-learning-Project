# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown
import pickle

# Page configuration
st.set_page_config(page_title="Pneumonia Detection", layout="wide")

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and the model will predict whether it is Normal or Pneumonia.")

# ------------------ Model Download ------------------ #
MODEL_PATH = "pneumonia_detection_model.h5"

# Google Drive direct download link for your model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1t5UlwpWWeIniy_WazLj6XnWtZ8cdbnlk"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ------------------ Sidebar Navigation ------------------ #
menu = ["Predict X-ray", "Dataset Overview", "Training Metrics"]
choice = st.sidebar.selectbox("Menu", menu)

# ------------------ Predict X-ray ------------------ #
if choice == "Predict X-ray":
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png","webp"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')  # convert to grayscale
        st.image(img, caption="Uploaded X-ray", use_column_width=True)
        
        # Preprocess the image
        img = img.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = img_array.reshape(1,224,224,1)
        
        # Predict
        prediction = model.predict(img_array)
        if prediction > 0.5:
            st.error("⚠️ Pneumonia Detected")
        else:
            st.success("✅ Normal Chest X-ray")

# ------------------ Dataset Overview ------------------ #
elif choice == "Dataset Overview":
    st.header("Dataset Sample Images and Class Distribution")
    
    # Modify this path if dataset exists
    dataset_path = "Chest-Xray-2/chest_xray/train"
    classes = ['NORMAL','PNEUMONIA']
    counts = {}
    
    st.write("**Class Distribution:**")
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        if os.path.exists(folder):
            counts[cls] = len(os.listdir(folder))
        else:
            counts[cls] = 0
    st.bar_chart(counts)
    
    # Display sample images
    st.write("**Sample Images:**")
    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    
    for i, cls in enumerate(classes):
        folder = os.path.join(dataset_path, cls)
        if os.path.exists(folder):
            images = os.listdir(folder)[:4]
            for j, img_name in enumerate(images):
                img = Image.open(os.path.join(folder, img_name)).convert('L')
                axes[i,j].imshow(img, cmap='gray')
                axes[i,j].set_title(cls)
                axes[i,j].axis('off')
        else:
            for j in range(4):
                axes[i,j].axis('off')
    
    st.pyplot(fig)

# ------------------ Training Metrics ------------------ #
elif choice == "Training Metrics":
    st.header("Training Accuracy & Loss Plots")
    
    history_file = "history.pkl"
    if os.path.exists(history_file):
        with open(history_file, "rb") as f:
            history = pickle.load(f)
        
        # Accuracy plot
        st.subheader("Accuracy")
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        st.pyplot(plt)
        plt.clf()
        
        # Loss plot
        st.subheader("Loss")
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend()
        st.pyplot(plt)
        plt.clf()
    else:
        st.warning("Training history not found. Save it as 'history.pkl' after training.")