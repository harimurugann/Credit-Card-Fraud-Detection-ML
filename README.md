# 🛡️ FraudShield — End-to-End Hybrid ML Pipeline
### Credit Card Fraud Detection | Production-Grade

---

## 📁 Project Structure

```
fraud_detection/
│
├── fraud_pipeline.py       # Main ML pipeline (train + evaluate + save)
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Pinned dependencies
│
└── outputs/                # Auto-created on first run
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── eda_feature_distributions.png
    ├── confusion_matrices.png
    ├── precision_recall_curves.png
    ├── model_comparison_summary.png
    ├── best_fraud_model.sav
    └── fraud_pipeline.sav
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
Download `creditcard.csv` from Kaggle:
→ https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 3. Configure & Train
Edit the `CONFIG` dict at the top of `fraud_pipeline.py`:

```python
CONFIG = {
    "mode":               "pc",          # "mobile" for lightweight run
    "imbalance_strategy": "smote_tomek", # see options below
    "training_mode":      "fast",        # "tuned" for GridSearch (slow)
    "plot_dpi":           150,
    "interactive_plots":  True,
}
```

Then run:
```bash
python fraud_pipeline.py
```

### 4. Launch App
```bash
streamlit run app.py
```

Open `http://localhost:8501` in any browser (works on mobile too).

---

## 🔧 Configuration Options

| Setting | Mobile | PC |
|---|---|---|
| `mode` | `"mobile"` | `"pc"` |
| `training_mode` | `"fast"` | `"tuned"` |
| `plot_dpi` | `72` | `150` |
| `interactive_plots` | `False` | `True` |
| `imbalance_strategy` | `"smote"` | `"smote_tomek"` |

### Imbalance Strategies
| Strategy | Description | Speed | Precision |
|---|---|---|---|
| `smote` | Oversample minority only | ⚡ Fast | Good |
| `smote_tomek` | Oversample + clean boundaries | 🔄 Medium | **Best** |
| `easy_ensemble` | Ensemble-level balancing | 🐢 Slow | High Recall |

---

## 📊 Pipeline Sections

| # | Section | Key Operation |
|---|---|---|
| 1 | Dependencies | imblearn, sklearn, joblib |
| 2 | Data Cleaning | Dedup + RobustScaler |
| 3 | Visualization | Class dist + Heatmap |
| 4 | EDA | KDE: Fraud vs Legit |
| 5 | Modeling | RF, XGBoost, LightGBM |
| 6 | Split | X/y separation |
| 7 | Train/Test | Stratified split |
| 8 | Training | Fast / Tuned mode |
| 9 | Evaluation | PR curve, F1, CM |
| 10 | Summary | Model comparison |
| 11 | Model Save | `.sav` + joblib compress=5 |
| 12 | Pipeline | ImbPipeline object |
| 13 | Pipeline Save | `.sav` serialized |

---

## 🚀 Deployment Notes

- The `fraud_pipeline.sav` is a full sklearn-compatible Pipeline
- Load it anywhere: `model = joblib.load("outputs/fraud_pipeline.sav")`
- Feed raw (unscaled) features — the pipeline handles scaling
- Adjust `threshold` in the Streamlit sidebar to tune Precision/Recall

---

## 📱 Mobile Compatibility

The Streamlit app uses:
- Responsive CSS (`clamp()`, mobile breakpoints)
- Lightweight matplotlib figures (no heavy JS dependencies)
- Lazy loading via `@st.cache_resource`

For mobile training, set `CONFIG["mode"] = "mobile"` to reduce:
- `n_estimators`: 100 → 50
- `plot_dpi`: 150 → 72
- Disabled: interactive Plotly, hyperparameter search
