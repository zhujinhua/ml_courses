import os

import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into setosa, versicolor, virginica')
st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
    st.text('Pepal characteristics')
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)
predict_button = st.button('Predict type of Iris', key="predict_button")

iris_model = joblib.load(os.path.join(BASE_DIR, 'machine_learning', 'iris.joblib'))
if predict_button:
    result = iris_model.predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])
