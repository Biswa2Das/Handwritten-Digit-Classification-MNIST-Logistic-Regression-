#  Handwritten Digit Recognition with Logistic Regression

This project demonstrates a simple yet effective machine learning pipeline for classifying handwritten digits (0–9) using the **MNIST dataset** and **Logistic Regression**. It includes a Jupyter Notebook for training and evaluation, as well as a lightweight **Streamlit web app** for real-time predictions using uploaded digit images.

---

## Project Structure

- `Handwritten_Digit_Classification_(MNIST_+_Logistic_Regression).ipynb`: Jupyter notebook used to load, preprocess, train, and evaluate the logistic regression model on the MNIST dataset.
- `app.py`: A Streamlit app that allows users to upload an image and get a digit prediction.
- `logistic_model.joblib`: Trained logistic regression model serialized using `joblib`.
- `scaler.joblib`: StandardScaler used for feature normalization during preprocessing.

---

##  How It Works

### 1. Training the Model
The notebook performs the following steps:
- Loads the MNIST dataset from OpenML.
- Splits the dataset into training and test sets.
- Scales the pixel values using `StandardScaler`.
- Trains a `LogisticRegression` classifier from `sklearn`.
- Saves the trained model and scaler using `joblib`.

### 2. Running the Streamlit App
The web app lets users upload an image of a digit and returns a prediction.

Steps performed:
- Converts the image to grayscale.
- Inverts the image (background black, digit white to match MNIST).
- Resizes to 28x28 pixels.
- Scales the input using the saved scaler.
- Uses the trained model to predict the digit.

---

##  Sample Usage

To launch the Streamlit app locally:

```bash
pip install streamlit scikit-learn pillow joblib numpy
streamlit run app.py
