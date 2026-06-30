!pip install streamlit

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('log_model.pkl')

st.title("Diabetes Prediction App")

# User inputs
preg = st.number_input("Pregnancies", 0, 20, 1)
glu = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 20.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 33)

# Create dataframe
user_data = pd.DataFrame({
    'Pregnancies':[preg],
    'Glucose':[glu],
    'BloodPressure':[bp],
    'SkinThickness':[skin],
    'Insulin':[insulin],
    'BMI':[bmi],
    'DiabetesPedigreeFunction':[dpf],
    'Age':[age]
})

# Predict button
if st.button("Predict"):
    pred = model.predict(user_data)[0]
    if pred == 1:
        st.error("High chance of Diabetes ⚠️")
    else:
        st.success("Low chance of Diabetes ✅")