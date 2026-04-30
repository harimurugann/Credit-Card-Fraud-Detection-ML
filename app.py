# ============================================================
# app.py — Streamlit Fraud Detection Inference App
# UI/UX Enhanced Version | Fixed Batch Processing
# ============================================================
import os
import time
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import RobustScaler

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS (Dark Theme & Responsive UI)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58A6FF !important;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Alert Boxes for Results */
    .fraud-alert {
        padding: 20px;
        background-color: rgba(255, 71, 87, 0.1);
        border-left: 5px solid #FF4757;
        color: #FF4757;
        border-radius: 5px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .safe-alert {
        padding: 20px;
        background-color: rgba(46, 213, 115, 0.1);
        border-left: 5px solid #2ED573;
        color: #2ED573;
        border-radius: 5px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #1F6FEB 0%, #3B82F6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 15px rgba(59, 130, 246, 0.5);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the fully fitted model/pipeline."""
    paths = ["outputs/best_fraud_model.sav", "outputs/fraud_pipeline.sav"]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

def get_v_features():
    return [f"V{i}" for i in range(1, 29)]

# ─────────────────────────────────────────────────────────────
# SIDEBAR (Settings & Status)
# ─────────────────────────────────────────────────────────────
active_model = load_model()

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    if active_model:
        st.success("🟢 System Online\nModel loaded successfully.")
    else:
        st.error("🔴 System Offline\nModel file not found.")
        
    st.markdown("---")
    threshold = st.slider("Risk Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                          help="Lower threshold makes the system more sensitive to fraud.")
    
    st.markdown("---")
    st.caption("FraudShield AI | Built by AI Data Engineer")

# ─────────────────────────────────────────────────────────────
# MAIN DASHBOARD HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("<h1>🛡️ Credit Guard ML</h1>", unsafe_allow_html=True)
st.markdown("Real-time AI monitoring for fraudulent transactions.")

# Show high-level metrics
col1, col2, col3 = st.columns(3)
col1.metric("Status", "Active Monitoring")
col2.metric("Detection Engine", "Hybrid RF + SMOTE")
col3.metric("Current Threshold", f"{threshold*100:.0f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS FOR BETTER UX
# ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Real-time Analysis", "📁 Batch Processing"])

# ─── TAB 1: REAL-TIME ANALYSIS ──────────────────────────────
with tab1:
    st.markdown("### Transaction Input")
    
    # UX Feature: Quick Presets
    preset = st.selectbox("Quick Testing Presets (Auto-fill)", 
                          ["-- Select Preset --", "Legitimate Transaction (Safe)", "Suspicious Pattern (Fraud)"])
    
    # Initialize default values
    v_values = {v: 0.0 for v in get_v_features()}
    amount_val = 150.0
    time_val = 3600.0
    
    if preset == "Suspicious Pattern (Fraud)":
        v_values["V3"] = -5.0
        v_values["V14"] = -8.5
        v_values["V17"] = -6.2
        amount_val = 999.99
    elif preset == "Legitimate Transaction (Safe)":
        v_values["V3"] = 1.2
        v_values["V14"] = 0.5
        v_values["V17"] = 0.2
        amount_val = 45.00

    # Primary Inputs
    c1, c2 = st.columns(2)
    with c1:
        scaled_amount = st.number_input("Transaction Amount ($) / scaled_amount", value=amount_val, format="%.2f")
    with c2:
        scaled_time = st.number_input("Time / scaled_time", value=time_val, format="%.2f")

    # Advanced Inputs hidden in an expander for Mobile friendliness
    with st.expander("⚙️ Advanced PCA Features (V1 - V28)"):
        st.caption("Modify these only if you have raw PCA transformed data.")
        v_cols = st.columns(4) # 4 columns for compact view
        for i, v_name in enumerate(get_v_features()):
            with v_cols[i % 4]:
                v_values[v_name] = st.number_input(v_name, value=v_values[v_name], format="%.4f")

    # Analysis Action
    if st.button("🚀 Analyze Transaction", use_container_width=True):
        if not active_model:
            st.error("Cannot analyze. Model is missing.")
        else:
            with st.spinner("Analyzing patterns..."):
                time.sleep(1) # Simulated network delay for better UX feel
                
                # Prepare data
                input_data = {**v_values, "scaled_amount": scaled_amount, "scaled_time": scaled_time}
                input_df = pd.DataFrame([input_data])
                
                # Predict
                try:
                    if hasattr(active_model, "predict_proba"):
                        probability = float(active_model.predict_proba(input_df)[:, 1][0])
                    else:
                        probability = float(active_model.predict(input_df)[0])
                        
                    prediction = int(probability >= threshold)
                    
                    # Display Results beautifully
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            🚨 HIGH RISK ALERT 🚨<br>
                            <span style="font-size: 16px; color: #fff;">Fraud Probability: {probability*100:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            ✅ TRANSACTION VERIFIED ✅<br>
                            <span style="font-size: 16px; color: #fff;">Fraud Probability: {probability*100:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

# ─── TAB 2: BATCH PROCESSING (Fixed Error) ──────────────────
with tab2:
    st.markdown("### Upload Multiple Transactions")
    st.caption("Upload a CSV file to analyze bulk records.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and active_model:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} records.")
        
        if st.button("Analyze Batch"):
            with st.spinner("Processing batch..."):
                try:
                    # 1. Handle Raw Data (Scale Time & Amount if present)
                    if "Amount" in df.columns and "Time" in df.columns:
                        scaler = RobustScaler()
                        df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
                        df["scaled_time"] = scaler.fit_transform(df[["Time"]])
                        
                    # 2. Extract strictly the required 30 features
                    expected_features = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
                    
                    # Check if all expected features are present
                    missing_cols = [col for col in expected_features if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"Missing columns in CSV: {missing_cols}")
                        st.info("Ensure your CSV has V1-V28 and either (Amount, Time) or (scaled_amount, scaled_time).")
                    else:
                        # Select ONLY the features the model knows (Ignore Class, Time, Amount, etc.)
                        X_batch = df[expected_features]
                        
                        # 3. Predict
                        if hasattr(active_model, "predict_proba"):
                            probs = active_model.predict_proba(X_batch)[:, 1]
                        else:
                            probs = active_model.predict(X_batch)
                            
                        # 4. Prepare Results
                        result_df = df.copy() # Keep original data for user reference
                        result_df["Fraud_Probability"] = probs
                        result_df["Risk_Status"] = ["Fraud" if p >= threshold else "Safe" for p in probs]
                        
                        # Show output beautifully
                        st.dataframe(result_df[["Risk_Status", "Fraud_Probability"]].head(100), use_container_width=True)
                        
                        # Download option
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ Download Results", csv, "fraud_analysis_results.csv", "text/csv")
                        
                except Exception as e:
                    st.error(f"Error during batch processing: {e}")
