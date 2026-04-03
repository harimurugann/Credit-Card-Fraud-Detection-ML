import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('credit_card_model.sav')

st.title("Credit Card Fraud Detection System")

st.write("Enter transaction features to check for fraud:")

# Creating inputs for all features (Simplified for demonstration)
input_df = st.text_input('Input all feature values separated by commas')

if st.button('Predict'):
    try:
        # Convert input string to numpy array
        features = np.array([float(x) for x in input_df.split(',')]).reshape(1, -1)
        prediction = model.predict(features)
        
        if prediction[0] == 0:
            st.success("Legitimate Transaction")
        else:
            st.error("Fraudulent Transaction Detected!")
    except:
        st.warning("Please enter valid comma-separated values (30 features required).")
      
