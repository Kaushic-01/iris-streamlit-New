
import streamlit as st
import joblib
import numpy as np

# 1. Load the saved machine learning model
@st.cache_resource
def load_model():
    return joblib.load("iris_model.pkl")

model = load_model()

# 2. Define the flower names
class_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

# 3. Create the webpage title
st.title("Iris Flower Classifier")

st.write("Enter the flower measurements:")

# 4. Take input from the user
sepal_length = st.number_input(
    "Sepal length (cm)",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal width (cm)",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal length (cm)",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal width (cm)",
    min_value=0.0,
    value=0.2
)

# 5. Predict when the button is clicked
if st.button("Predict"):

    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(features)

    result = class_names[prediction[0]]

    st.success(f"Prediction: {result}")
