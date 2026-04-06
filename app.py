import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved pipeline
model = joblib.load('full_pipeline.sav')

st.title("Credit Card Fraud Detection System")
st.write("Enter transaction details to check if it is Fraudulent or Normal.")

# Creating input fields for the features
# Note: Since there are 30 features, for simplicity we use a few or provide a way to upload CSV
input_data = st.text_input("Enter feature values separated by commas (30 values):")

if st.button("Predict"):
    if input_data:
        try:
            values = [float(i) for i in input_data.split(',')]
            if len(values) == 30:
                prediction = model.predict([values])
                if prediction[0] == 0:
                    st.success("Result: This is a NORMAL Transaction.")
                else:
                    st.error("Result: This is a FRAUDULENT Transaction!")
            else:
                st.warning("Please enter exactly 30 values.")
        except Exception as e:
            st.error(f"Error: {e}")
