# 🫁 Pneumonia Detection from Chest X-Ray Images using Deep Learning

## 📌 Project Overview
This project builds a Deep Learning model using Convolutional Neural Networks (CNN) to detect Pneumonia from chest X-ray images.

The model classifies images into two categories:
- Normal
- Pneumonia

Early detection helps doctors provide faster treatment and reduce health risks.

---

## 💼 Business Problem
Manual diagnosis of pneumonia takes time and requires expert radiologists.  
This project automates detection using AI to help doctors make faster and more accurate decisions.

---

## 📊 Dataset
The dataset contains grayscale chest X-ray images divided into:
- Train dataset  
- Validation dataset  
- Test dataset  

Classes:
- Normal  
- Pneumonia  

### 📂 Folder Structure
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/

---

## ⚙️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  


---

## 🔄 Project Workflow
1. Data Loading  
2. Data Preprocessing (Resizing, Normalization, Augmentation)  
3. Model Building (CNN)  
4. Model Training  
5. Model Evaluation  
6. Prediction  

---

## 🧠 Model Architecture
- Convolution Layer  
- MaxPooling Layer  
- Flatten Layer  
- Dense Layer  
- Dropout  

---

## 🏋️ Model Training
- EarlyStopping  
- ModelCheckpoint  
- ReduceLROnPlateau  

---

## 📈 Model Evaluation
- Accuracy  
- Confusion Matrix  
- Classification Report  

---

## 🎯 Model Output
The model predicts whether an X-ray image is:
- Normal  
- Pneumonia  

---

## 📦 Installation & Setup
git clone https://github.com/amolsatpute33/pneumonia-detection.git  
cd pneumonia-detection  
pip install -r requirements.txt  
python app.py  

---

## 📚 Libraries Used
import numpy as np  
import matplotlib.pyplot as plt  
from tensorflow import keras  
from tensorflow.keras import layers  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from sklearn.metrics import classification_report  

---

## 🚀 Future Improvements
- Use Transfer Learning (ResNet, VGG16, EfficientNet)  
- Improve accuracy with larger datasets  
- Deploy as a web application  
- Integrate into hospital systems  

---

## 👨‍💻 Author
Amol Satpute  
LinkedIn: https://linkedin.com/in/amol-satpute-35b73a372  
GitHub: https://github.com/amolsatpute33  

---

## ⭐ Support
If you like this project, please give it a star on GitHub!
