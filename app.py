# app.py
import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🖊️ Handwritten Digit Recognition (Logistic Regression)")

uploaded_file = st.file_uploader("Upload a digit image (white on black, 28x28 preferred)", type=["png", "jpg", "jpeg"])

def preprocess_image(image: Image.Image):
    # Convert to grayscale
    gray_image = image.convert("L")
    
    # Invert: make background black and digit white (MNIST style)
    inverted = ImageOps.invert(gray_image)

    # Resize to 28x28
    resized = inverted.resize((28, 28))

    # Convert to numpy array and flatten
    img_array = np.array(resized).reshape(1, -1)

    # Scale using saved scaler
    img_scaled = scaler.transform(img_array)

    return img_scaled

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    st.success(f"Predicted Digit: {prediction[0]}")

