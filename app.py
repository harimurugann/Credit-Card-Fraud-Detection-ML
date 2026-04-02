import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Health & Finance Predictor", layout="wide")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Go to", ["Credit Card Fraud Detection", "Chronic Disease Prediction"])

# --- 1. Credit Card Fraud Detection Page ---
if selection == "Credit Card Fraud Detection":
    st.title("🚨 Credit Card Fraud Detection")
    st.write("Enter transaction details to check for fraud.")

    # Loading the model
    fraud_model = pickle.load(open('credit_card_fraud_model.sav', 'rb'))

    # Input fields (Example: V1, V2 and Amount)
    # Inga namma sample-ku 3 inputs vaikalam (Real app-la ellathaiyum add pannalam)
    v1 = st.number_input("Feature V1")
    v2 = st.number_input("Feature V2")
    amount = st.number_input("Transaction Amount")

    if st.button("Detect Fraud"):
        # Real dataset-la 30 features irukum, sample-ku mathadhula 0 nu vaikalam
        features = np.zeros(30)
        features[1] = v1
        features[2] = v2
        features[29] = amount
        
        prediction = fraud_model.predict([features])
        
        if prediction[0] == 1:
            st.error("🚨 Warning: This is a Fraudulent Transaction!")
        else:
            st.success("✅ This is a Normal Transaction.")

# --- 2. Chronic Disease Prediction Page ---
elif selection == "Chronic Disease Prediction":
    st.title("🏥 Chronic Disease Prediction")
    st.write("Provide health metrics to predict disease risk.")

    # Loading the model
    disease_model = pickle.load(open('chronic_disease_model.sav', 'rb'))

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120)
        bp = st.number_input("Blood Pressure")
    with col2:
        glucose = st.number_input("Glucose Level")
        bmi = st.number_input("BMI")

    if st.button("Predict Health Status"):
        # Input-ah model format-ku mathanum
        health_features = np.array([age, bp, glucose, bmi]).reshape(1, -1)
        
        # Note: Inga unga model-la evlo columns irukko adhu ellathaiyum 
        # input-ah kudukanum. Idhu oru sample logic dhan.
        prediction = disease_model.predict(health_features)
        
        if prediction[0] == 1:
            st.warning("⚠️ High Risk: Chronic Disease Detected.")
        else:
            st.success("🎉 Low Risk: Patient is Healthy.")

