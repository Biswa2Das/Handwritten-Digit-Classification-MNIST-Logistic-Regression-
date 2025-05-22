import streamlit as st
import joblib
import numpy as np
from PIL import Image

model = joblib.load("logistic_mnist_model.joblib")
scaler = joblib.load("scaler_mnist.joblib")

def preprocess_image(image):
    img = image.convert('L')  
    img = img.resize((28, 28))
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_flattened = img_array.reshape(1, -1)
    img_scaled = scaler.transform(img_flattened)
    return img_scaled

def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    confidence = model.predict_proba(processed).max()
    return prediction, confidence
st.title("Handwritten Digit Recognition")
st.write("Upload a 28x28 grayscale image of a digit (0-9).")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction, confidence = predict_digit(image)
    st.write(f"### Predicted Digit: {prediction}")
    st.write(f"Confidence: {confidence:.2f}")
