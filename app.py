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
import seaborn as sns
from io import BytesIO

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — ML Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark cyberpunk theme, mobile-first responsive
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:    #0a0e1a;
        --bg-secondary:  #111827;
        --bg-card:       #1a2035;
        --accent-blue:   #00d4ff;
        --accent-red:    #ff4757;
        --accent-green:  #2ed573;
        --accent-orange: #ff6b35;
        --text-primary:  #e8eaf0;
        --text-muted:    #8892a4;
        --border:        rgba(0,212,255,0.15);
    }

    /* ── Global ── */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Rajdhani', sans-serif;
        color: var(--text-primary);
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1rem 1.5rem 2rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--accent-blue);
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
    }

    /* ── Page header ── */
    .page-header {
        background: linear-gradient(135deg, #0d1b3e 0%, #0a2240 50%, #0d1b3e 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .page-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-red), var(--accent-blue));
    }
    .page-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: clamp(1.4rem, 3vw, 2.2rem);
        color: var(--accent-blue);
        margin: 0 0 0.3rem 0;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
        letter-spacing: 2px;
    }
    .page-header p {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent-blue); }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-blue);
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── Alert boxes ── */
    .alert-fraud {
        background: rgba(255,71,87,0.12);
        border: 1px solid var(--accent-red);
        border-left: 4px solid var(--accent-red);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Space Mono', monospace;
    }
    .alert-legit {
        background: rgba(46,213,115,0.1);
        border: 1px solid var(--accent-green);
        border-left: 4px solid var(--accent-green);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Space Mono', monospace;
    }

    /* ── Upload zone ── */
    .uploadedFile {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 8px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: #000 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), #0077aa);
        color: #000;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        letter-spacing: 1px;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,212,255,0.4);
    }

    /* ── Sliders and inputs ── */
    .stSlider > div > div > div { background: var(--accent-blue) !important; }
    .stNumberInput input, .stTextInput input {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem 0.8rem 1rem; }
        .page-header { padding: 1.2rem; }
        .page-header h1 { font-size: 1.3rem; }
        .metric-card .value { font-size: 1.3rem; }
    }

    /* ── DataFrames ── */
    .dataframe { font-size: 0.82rem !important; }

    /* ── Section dividers ── */
    .section-header {
        font-family: 'Space Mono', monospace;
        color: var(--accent-blue);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(path: str):
    """Load the serialized pipeline with caching."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def get_v_feature_names(n=28):
    """Return PCA feature names V1-V28."""
    return [f"V{i}" for i in range(1, n + 1)]


def make_gauge_chart(probability: float) -> plt.Figure:
    """
    Render a simple gauge/meter chart for fraud probability.
    Returns a Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0a0e1a",
                           subplot_kw={"polar": True})
    fig.patch.set_facecolor("#0a0e1a")

    theta = np.linspace(0, np.pi, 200)
    # Background arc
    ax.fill_between(theta, 0.7, 1.0, color="#1a2035", zorder=1)
    # Foreground arc (probability)
    fill_theta = np.linspace(0, probability * np.pi, 200)
    color = "#2ed573" if probability < 0.4 else \
            "#ffa502" if probability < 0.7 else "#ff4757"
    ax.fill_between(fill_theta, 0.7, 1.0, color=color, alpha=0.9, zorder=2)
    # Needle
    needle_theta = probability * np.pi
    ax.plot([needle_theta, needle_theta], [0, 0.9],
            color="white", linewidth=2.5, zorder=3)
    ax.plot(needle_theta, 0, "o", color="white", markersize=8, zorder=4)

    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#0a0e1a")

    ax.text(0, 0.35, f"{probability*100:.1f}%",
            ha="center", va="center",
            color="white", fontsize=18, fontweight="bold",
            fontfamily="monospace")
    ax.text(0, 0.15, "FRAUD PROBABILITY",
            ha="center", va="center",
            color="#8892a4", fontsize=7, fontfamily="monospace")
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("---")

    model_path = st.text_input(
        "Model path",
        value="outputs/best_fraud_model.sav",
        help="Path to your saved .sav model file"
    )
    pipeline_path = st.text_input(
        "Pipeline path",
        value="outputs/fraud_pipeline.sav",
        help="Path to saved pipeline (preferred)"
    )

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Decision threshold", 0.1, 0.9, 0.5, 0.05,
        help="Adjust the probability cutoff for fraud classification"
    )

    show_shap = st.checkbox("Show feature breakdown", value=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#8892a4;font-size:0.75rem;font-family:monospace'>"
        "v1.0.0 | FraudShield ML<br>"
        "Built with ❤️ for production</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🛡️ FRAUDSHIELD</h1>
    <p>Production-grade Credit Card Fraud Detection · Powered by Hybrid ML Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
pipeline = load_pipeline(pipeline_path)
model    = load_pipeline(model_path)
active   = pipeline or model

if active is None:
    st.warning(
        "⚠️ No model found. Run `fraud_pipeline.py` first to generate "
        "`outputs/best_fraud_model.sav`, then refresh this page."
    )
    use_demo = True
else:
    use_demo = False
    st.success(f"✅ Model loaded from: `{pipeline_path if pipeline else model_path}`")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🔍 Single Transaction", "📁 Batch Upload", "📊 Model Info"]
)


# ──────────────────────────────────────────────────────────────
# TAB 1: SINGLE TRANSACTION ANALYSIS
# ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Transaction Input</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.caption("Enter the PCA-transformed V1–V28 features and scaled time/amount.")

        # Layout V features in a responsive grid
        v_features = get_v_feature_names(28)
        v_values = {}

        # Display in groups of 4
        for row_start in range(0, 28, 4):
            row_cols = st.columns(4)
            for i, col in enumerate(row_cols):
                fidx = row_start + i
                if fidx < 28:
                    fname = v_features[fidx]
                    with col:
                        v_values[fname] = st.number_input(
                            fname, value=0.0, format="%.4f",
                            step=0.1, label_visibility="visible",
                            key=f"v_{fidx}"
                        )

        st.markdown('<div class="section-header">Scaled Features</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            scaled_amount = st.number_input(
                "scaled_amount", value=0.0, format="%.4f"
            )
        with c2:
            scaled_time = st.number_input(
                "scaled_time", value=0.0, format="%.4f"
            )

    with col_right:
        st.markdown('<div class="section-header">Quick Presets</div>',
                    unsafe_allow_html=True)
        preset = st.selectbox(
            "Load a preset transaction",
            ["— Select —", "Normal purchase", "Suspicious pattern",
             "High-risk transaction"]
        )

        # Preset injection hint
        if preset == "Suspicious pattern":
            st.info("💡 Set V4=4.0, V11=-3.5, V14=-6.2, Amount=1500 "
                    "for a suspicious pattern.")
        elif preset == "High-risk transaction":
            st.info("💡 Set V14=-10.0, V17=-8.5, V12=-5.0, Amount=8999 "
                    "for a high-risk pattern.")

        st.markdown('<div class="section-header">Analysis</div>',
                    unsafe_allow_html=True)

        if st.button("🔍 ANALYZE TRANSACTION", key="analyze"):
            # Build feature vector
            input_data = {**v_values,
                          "scaled_amount": scaled_amount,
                          "scaled_time":   scaled_time}
            input_df = pd.DataFrame([input_data])

            if use_demo:
                # Demo mode: random probability
                probability = float(np.random.beta(1, 8))
            else:
                try:
                    m = pipeline if pipeline else model
                    if hasattr(m, "predict_proba"):
                        probability = float(m.predict_proba(input_df)[:, 1][0])
                    else:
                        probability = float(m.predict(input_df)[0])
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    probability = 0.0

            prediction = int(probability >= threshold)

            # ── Gauge ─────────────────────────────────────────
            gauge_fig = make_gauge_chart(probability)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close()

            # ── Verdict ───────────────────────────────────────
            if prediction == 1:
                st.markdown(f"""
                <div class="alert-fraud">
                    ⚠️ <strong>FRAUD DETECTED</strong><br>
                    Probability: {probability:.4f} &nbsp;|&nbsp;
                    Threshold: {threshold:.2f}<br>
                    <small>Recommend: Block transaction & trigger review</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-legit">
                    ✅ <strong>TRANSACTION LEGITIMATE</strong><br>
                    Probability: {probability:.4f} &nbsp;|&nbsp;
                    Threshold: {threshold:.2f}<br>
                    <small>Transaction approved</small>
                </div>""", unsafe_allow_html=True)

            # ── Feature breakdown ─────────────────────────────
            if show_shap:
                st.markdown('<div class="section-header">Feature Contribution</div>',
                            unsafe_allow_html=True)
                # Approximate contribution using input values
                vals = np.array(list(input_data.values()))
                labels = list(input_data.keys())
                sorted_idx = np.argsort(np.abs(vals))[-15:]

                fig, ax = plt.subplots(figsize=(6, 5),
                                       facecolor="#0a0e1a")
                ax.set_facecolor("#1a2035")
                colors_ = ["#ff4757" if v > 0 else "#00d4ff"
                           for v in vals[sorted_idx]]
                ax.barh(np.array(labels)[sorted_idx],
                        vals[sorted_idx], color=colors_,
                        edgecolor="none")
                ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
                ax.set_title("Top Features by Magnitude",
                             color="white", fontsize=10,
                             fontfamily="monospace")
                ax.tick_params(colors="white", labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                for sp in ax.spines.values():
                    sp.set_color("#333")
                st.pyplot(fig, use_container_width=True)
                plt.close()


# ──────────────────────────────────────────────────────────────
# TAB 2: BATCH UPLOAD
# ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Batch Transaction Scoring</div>',
                unsafe_allow_html=True)
    st.caption(
        "Upload a CSV with V1–V28, scaled_amount, scaled_time columns. "
        "The app will score each row and flag fraud."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Max recommended: 50,000 rows for in-browser performance"
    )

    if uploaded_file:
        with st.spinner("Processing batch..."):
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_upload):,} transactions")

                # Check required columns
                required = get_v_feature_names(28) + \
                           ["scaled_amount", "scaled_time"]
                missing = [c for c in required if c not in df_upload.columns]

                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    X_batch = df_upload[required]

                    if use_demo:
                        probs = np.random.beta(1, 10, len(df_upload))
                    else:
                        m = pipeline if pipeline else model
                        if hasattr(m, "predict_proba"):
                            probs = m.predict_proba(X_batch)[:, 1]
                        else:
                            probs = m.predict(X_batch).astype(float)

                    df_upload["fraud_probability"] = probs
                    df_upload["prediction"] = (probs >= threshold).astype(int)
                    df_upload["verdict"] = df_upload["prediction"].map(
                        {0: "✅ Legitimate", 1: "⚠️ FRAUD"}
                    )

                    # ── Summary metrics ───────────────────────
                    n_fraud = int(df_upload["prediction"].sum())
                    n_legit = len(df_upload) - n_fraud
                    fraud_rate = n_fraud / len(df_upload) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    for m, lbl, val in [
                        (m1, "Total", f"{len(df_upload):,}"),
                        (m2, "Legitimate", f"{n_legit:,}"),
                        (m3, "Fraud", f"{n_fraud:,}"),
                        (m4, "Fraud Rate", f"{fraud_rate:.2f}%"),
                    ]:
                        m.markdown(
                            f'<div class="metric-card">'
                            f'<div class="value">{val}</div>'
                            f'<div class="label">{lbl}</div></div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("")

                    # ── Distribution plot ─────────────────────
                    fig, ax = plt.subplots(figsize=(10, 4),
                                           facecolor="#0a0e1a")
                    ax.set_facecolor("#1a2035")
                    ax.hist(probs[df_upload["prediction"] == 0],
                            bins=50, alpha=0.7, color="#00d4ff",
                            label="Legitimate", density=True)
                    ax.hist(probs[df_upload["prediction"] == 1],
                            bins=50, alpha=0.7, color="#ff4757",
                            label="Fraud", density=True)
                    ax.axvline(threshold, color="#ffa502", linewidth=1.5,
                               linestyle="--", label=f"Threshold={threshold}")
                    ax.set_title("Fraud Probability Distribution",
                                 color="white", fontfamily="monospace",
                                 fontsize=11)
                    ax.set_xlabel("Probability", color="white")
                    ax.set_ylabel("Density", color="white")
                    ax.tick_params(colors="white")
                    ax.legend(labelcolor="white", facecolor="#1a2035",
                              edgecolor="#444")
                    ax.spines[["top", "right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    # ── Results table ─────────────────────────
                    st.markdown('<div class="section-header">Results Preview</div>',
                                unsafe_allow_html=True)
                    display_cols = ["fraud_probability", "prediction", "verdict"]
                    st.dataframe(
                        df_upload[display_cols].head(200),
                        use_container_width=True
                    )

                    # ── Download ──────────────────────────────
                    csv_out = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Scored CSV",
                        csv_out,
                        "fraud_scored.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")


# ──────────────────────────────────────────────────────────────
# TAB 3: MODEL INFO
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Pipeline Architecture</div>',
                unsafe_allow_html=True)

    arch_data = {
        "Stage": ["1. Data Cleaning", "2. RobustScaler",
                  "3. SMOTETomek", "4. ML Model", "5. Threshold"],
        "Component": [
            "Deduplicate + scale Time/Amount",
            "Handles outliers in Amount/Time",
            "SMOTE oversample + Tomek undersample",
            "RandomForest / XGBoost / LightGBM",
            "Custom threshold (configurable)"
        ],
        "Purpose": [
            "Clean raw data",
            "Normalize skewed features",
            "Balance class distribution",
            "Learn fraud patterns",
            "Tune Precision vs Recall tradeoff"
        ]
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True)

    st.markdown('<div class="section-header">Output Files</div>',
                unsafe_allow_html=True)

    outputs = [
        ("📊", "class_distribution.png", "Class balance chart"),
        ("🔥", "correlation_heatmap.png", "Feature correlations"),
        ("📈", "eda_feature_distributions.png", "Fraud vs Legit KDE"),
        ("🎯", "confusion_matrices.png", "Model evaluation"),
        ("📉", "precision_recall_curves.png", "PR curves"),
        ("🏆", "model_comparison_summary.png", "Model comparison"),
        ("💾", "best_fraud_model.sav", "Saved model"),
        ("🔧", "fraud_pipeline.sav", "Saved pipeline"),
    ]
    for icon, fname, desc in outputs:
        fpath = f"outputs/{fname}"
        exists = "✅" if os.path.exists(fpath) else "⬜"
        st.markdown(
            f"{exists} **{icon} {fname}** — {desc}"
        )
        # Show saved plots if they exist
        if os.path.exists(fpath) and fname.endswith(".png"):
            with st.expander(f"Preview: {fname}"):
                st.image(fpath)

    st.markdown('<div class="section-header">How to Run</div>',
                unsafe_allow_html=True)
    st.code("""
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Place your dataset
# Download creditcard.csv from Kaggle (IEEE-CIS / ULB dataset)

# Step 3: Train the pipeline
python fraud_pipeline.py

# Step 4: Launch the app
streamlit run app.py
    """, language="bash")

    st.markdown('<div class="section-header">CONFIG Quick Reference</div>',
                unsafe_allow_html=True)
    st.code("""
CONFIG = {
    "mode":               "pc",          # "mobile" | "pc"
    "imbalance_strategy": "smote_tomek", # "smote_tomek" | "easy_ensemble" | "smote"
    "training_mode":      "fast",        # "fast" | "tuned"
    "plot_dpi":           150,           # 72 (mobile) | 150+ (pc)
    "interactive_plots":  True,          # Plotly charts (pc only)
}
    """, language="python")
