import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('credit_card_model.sav')

st.title("Credit Card Fraud Detection System")

st.write("Enter transaction features to check for fraud:")

# Input field for features
input_df = st.text_input('Input all feature values separated by commas')

if st.button('Predict'):
    try:
        # Convert input string to numpy array
        input_list = [float(x) for x in input_df.split(',')]
        features = np.array(input_list).reshape(1, -1)
        
        # Making prediction
        prediction = model.predict(features)
        
        if prediction[0] == 0:
            st.success("Legitimate Transaction")
        else:
            st.error("Fraudulent Transaction Detected!")
            
    except Exception as e:
        st.warning("Please enter all 30 feature values separated by commas correctly.")
        
