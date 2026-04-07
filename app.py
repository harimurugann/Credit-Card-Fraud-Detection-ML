import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline
model = joblib.load('full_pipeline.sav')

st.title("🛡️ Smart Fraud Investigation Dashboard")

# Option 1: File Upload for Bulk Data (100+ rows)
uploaded_file = st.file_uploader("Upload Transaction CSV File for Analysis", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Make predictions for the entire file
    # Note: Assuming the CSV has the same 30 features as training
    predictions = model.predict(data)
    
    # Add predictions back to the dataframe
    data['Status'] = predictions
    data['Status'] = data['Status'].map({0: 'Normal', 1: '🚨 FRAUD'})
    
    # --- HERE IS YOUR IDEA IMPLEMENTED ---
    
    # 1. Total Summary
    total_fraud = (predictions == 1).sum()
    st.subheader(f"Analysis Summary: Found {total_fraud} Fraudulent Transactions")
    
    # 2. Filter and Show ONLY Fraud Transactions
    fraud_only = data[data['Status'] == '🚨 FRAUD']
    
    if not fraud_only.empty:
        st.error("### 🚩 List of Fraudulent Transactions Identified:")
        # Highlighted Table
        st.dataframe(fraud_only.style.applymap(lambda x: 'background-color: #ffcccc', subset=['Status']))
        
        # Download button for the Fraud Report
        csv = fraud_only.to_csv(index=False).encode('utf-8')
        st.download_button("Download Fraud Report", csv, "fraud_report.csv", "text/csv")
    else:
        st.success("✅ No Fraudulent Transactions detected in this batch!")

# Option 2: Individual Input (Keep your existing single input logic here...)
