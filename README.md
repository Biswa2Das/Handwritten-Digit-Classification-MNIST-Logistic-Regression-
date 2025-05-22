#  Handwritten Digit Recognition (MNIST + Logistic Regression)

This is a web application built with [Streamlit](https://streamlit.io) that allows you to upload an image of a handwritten digit and predicts the digit using a logistic regression model trained on the MNIST dataset.

##  Features

- Upload digit images in PNG, JPG, or JPEG formats.
- Automatically preprocesses images to match MNIST input standards.
- Displays the predicted digit using a trained logistic regression model.
- Clean and simple web interface.


##  Model Details

- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images (28x28) of handwritten digits (0 to 9).
- **Preprocessing**:
  - Standardized using `StandardScaler`.
  - Inversion: white digit on black background assumed.
  - Resizing to 28x28 pixels.
- **Model**: `LogisticRegression` from Scikit-learn with multinomial classification and the `saga` solver.

---

## How to Run

1. **Install dependencies**:

```bash
pip install streamlit scikit-learn joblib pillow numpy
````

2. **Ensure the following files are in your working directory**:

   * `streamlit_app.py` (your Streamlit app script)
   * `logistic_model.joblib` (trained model)
   * `scaler.joblib` (standard scaler)


4. **Upload a digit image** and see the prediction in your browser.


## Image Format Guidelines

To ensure the best prediction accuracy:

* **Format**: PNG, JPG, JPEG
* **Size**: Preferably 28x28 pixels (other sizes are auto-resized)
* **Color**: White digit on black background (handled by inversion)
* **Content**: Centered and clearly visible handwritten digit

## Author

Developed by **Biswadas E J**
Feel free to contribute or raise issues!


