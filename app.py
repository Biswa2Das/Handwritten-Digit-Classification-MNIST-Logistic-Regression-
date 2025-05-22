import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the saved model
model = load_model("mnist_digit_classifier.h5")

st.title("Handwritten Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert image (black background to white)
    image = image.resize((28, 28))  # Resize to 28x28

    st.image(image, caption='Uploaded Image', width=150)
    
    # Preprocess the image
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"**Predicted Digit:** {predicted_class}")
