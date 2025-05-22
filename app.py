import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

model = joblib.load("logistic_mnist_model.joblib")
scaler = joblib.load("scaler_mnist.joblib")

def preprocess_image(img):
    img = img.convert('L') 
    if np.mean(np.array(img)) > 127:
        img = ImageOps.invert(img)
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    upper_left = ((28 - img.width) // 2, (28 - img.height) // 2)
    canvas.paste(img, upper_left)
    img_array = np.array(canvas).reshape(1, -1)
    img_scaled = scaler.transform(img_array)
    return img_scaled, canvas

st.title("MNIST Digit Recognizer")
st.write("Upload an image of a digit (ideally handwritten).")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_input, processed_img = preprocess_image(image)
    prediction = model.predict(processed_input)[0]
    confidence = model.predict_proba(processed_input).max()
    st.image(processed_img, caption="Processed 28x28 Image", width=150)
    st.success(f"Predicted Digit: {prediction}")
    st.info(f"Confidence: {confidence:.2f}")

