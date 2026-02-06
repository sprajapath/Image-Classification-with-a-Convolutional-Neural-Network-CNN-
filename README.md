# Image-Classification-with-a-Convolutional-Neural-Network-CNN-

# CIFAR-10 Image Classification with CNN + Streamlit App

This capstone project demonstrates a complete deep learning workflow using **TensorFlow/Keras** to train a **Convolutional Neural Network (CNN)** on the **CIFAR-10 dataset**.  
After training and evaluation, the model is deployed through a **Streamlit web application** that allows users to upload an image and receive a predicted class label.

---

## 📌 Project Overview

This project includes:

- Loading and preprocessing the CIFAR-10 dataset
- Building a CNN model using TensorFlow/Keras
- Training the model and tracking learning curves
- Evaluating performance using:
  - Accuracy
  - Classification report
  - Confusion matrix
- Saving the trained model
- Deploying the model using a Streamlit web app
- Running Streamlit inside Google Colab using **ngrok**

---

## 🎯 Objectives

- Load and preprocess CIFAR-10 images
- Design a CNN architecture suitable for multi-class image classification
- Train and validate the model
- Evaluate model performance using multiple metrics
- Save the trained model as an `.h5` file
- Deploy the model in a Streamlit app with an image upload interface

---

## 📊 Dataset: CIFAR-10

CIFAR-10 is a well-known benchmark dataset for image classification.

- **60,000 RGB images**
- Image size: **32 × 32**
- **50,000 training images**
- **10,000 test images**
- **10 classes**

### Classes
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## 🧠 Model Architecture (CNN)

The CNN model is built using Keras Sequential API.

### Layers Used
- Conv2D + ReLU
- MaxPooling2D
- Flatten
- Dense + ReLU
- Dropout
- Dense + Softmax (10 classes)

---

## 🧪 Training & Evaluation

### Preprocessing
- Pixel values normalized to range **0–1**
- Labels kept as integer encoded (sparse categorical)

### Evaluation Metrics
- Test Accuracy
- Classification Report (precision, recall, F1-score)
- Confusion Matrix heatmap

Typical accuracy for this architecture is usually around **65%–75%**, depending on training settings.

---

## 💾 Model Saving

After training, the model is saved as:


This file is later loaded inside the Streamlit app for prediction.

---

## 🌐 Streamlit Deployment (Colab + ngrok)

A Streamlit web app is created (`app.py`) to:

- Upload an image (jpg/png)
- Resize it to 32×32
- Normalize it
- Predict the class using the trained CNN model
- Display the predicted label

Since Google Colab does not expose local ports directly, **ngrok** is used to create a public URL for the Streamlit app.

---

## 🛠️ Tools & Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- pyngrok
- PIL (Pillow)

---

## 🚀 How to Run

### Option 1: Run Notebook (Recommended)
Open the notebook in:

- Google Colab
- Jupyter Notebook / JupyterLab

Then run all cells.

---

### Option 2: Run Streamlit Locally (Optional)

#### 1) Install dependencies
```bash
pip install -r requirements.txt


streamlit run app.py
