# ============================================================
# app.py — Streamlit Fraud Detection Inference App
# Production-ready | Mobile & PC compatible
# ============================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="FraudShield — ML Detection", page_icon="🛡️", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the fully fitted model. Checks both pipeline and model paths."""
    paths = ["outputs/best_fraud_model.sav", "outputs/fraud_pipeline.sav"]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

def get_v_feature_names(n=28):
    return [f"V{i}" for i in range(1, n + 1)]

# ─────────────────────────────────────────────────────────────
# HEADER & SIDEBAR
# ─────────────────────────────────────────────────────────────
st.markdown("<h1>🛡️ FRAUDSHIELD AI</h1>", unsafe_allow_html=True)
st.markdown("Production-grade Credit Card Fraud Detection")

active_model, loaded_path = load_model()

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    if active_model:
        st.success(f"✅ Model loaded: {loaded_path.split('/')[-1]}")
    else:
        st.warning("⚠️ No model found! Run fraud_pipeline.py first.")

# ──────────────────────────────────────────────────────────────
# MAIN UI: TRANSACTION ANALYSIS
# ──────────────────────────────────────────────────────────────
st.markdown("### Transaction Input")
col1, col2 = st.columns([3, 2])

with col1:
    v_features = get_v_feature_names(28)
    v_values = {}
    
    # Render inputs in rows of 4 for mobile friendliness
    for i in range(0, 28, 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < 28:
                fname = v_features[i + j]
                v_values[fname] = col.number_input(fname, value=0.0, format="%.4f")

    st.markdown("#### Scaled Features")
    sc1, sc2 = st.columns(2)
    scaled_amount = sc1.number_input("scaled_amount", value=0.0, format="%.4f")
    scaled_time = sc2.number_input("scaled_time", value=0.0, format="%.4f")

with col2:
    if st.button("🔍 ANALYZE TRANSACTION", use_container_width=True):
        if not active_model:
            st.error("Prediction error: Model is not loaded.")
        else:
            # Build exact feature matching dataframe
            input_data = {**v_values, "scaled_amount": scaled_amount, "scaled_time": scaled_time}
            input_df = pd.DataFrame([input_data])
            
            try:
                if hasattr(active_model, "predict_proba"):
                    probability = float(active_model.predict_proba(input_df)[:, 1][0])
                else:
                    probability = float(active_model.predict(input_df)[0])
                
                prediction = int(probability >= threshold)
                
                if prediction == 1:
                    st.error(f"⚠️ FRAUD DETECTED! (Probability: {probability*100:.2f}%)")
                else:
                    st.success(f"✅ TRANSACTION LEGITIMATE (Probability: {probability*100:.2f}%)")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")