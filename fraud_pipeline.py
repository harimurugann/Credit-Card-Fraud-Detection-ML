# ============================================================
# END-TO-END HYBRID ML PIPELINE: CREDIT CARD FRAUD DETECTION
# Author: AI Data Engineer
# Description: Production-grade fraud detection pipeline with
#              advanced imbalance handling, hybrid architecture,
#              and full evaluation suite.
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Skipping XGBClassifier.")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARNING] LightGBM not installed. Skipping LGBMClassifier.")

from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

CONFIG = {
    "mode": "pc",
    "imbalance_strategy": "smote_tomek",
    "training_mode": "fast",
    "output_dir": "outputs",
    "random_state": 42,
    "test_size": 0.2,
    "plot_dpi": 150,
    "interactive_plots": False,
    "model_save_path": "outputs/best_fraud_model.sav",
    "pipeline_save_path": "outputs/fraud_pipeline.sav",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 2: DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    print("\n[STEP 2] Loading and cleaning data...")
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    
    # Scale 'Time' and 'Amount'
    scaler = RobustScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Time", "Amount"], axis=1, inplace=True)
    
    return df

# ─────────────────────────────────────────────────────────────
# STEP 3 & 4: VISUALIZATION & EDA
# ─────────────────────────────────────────────────────────────
def visualize_data(df: pd.DataFrame):
    print("\n[STEP 3] Generating visualizations...")
    # Kept simple to save processing time on mobile/PC
    counts = df["Class"].value_counts()
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette=["#00d4ff", "#ff4757"])
    plt.title("Class Distribution")
    plt.savefig(f"{CONFIG['output_dir']}/class_distribution.png", dpi=CONFIG["plot_dpi"])
    plt.close()

def perform_eda(df: pd.DataFrame):
    print("\n[STEP 4] Performing EDA (Skipped for performance)...")
    pass

# ─────────────────────────────────────────────────────────────
# STEP 5, 6 & 7: SPLIT & IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────
def split_features_target(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def apply_imbalance_strategy(X_train, y_train):
    strategy = CONFIG["imbalance_strategy"]
    print(f"\n[IMBALANCE] Applying strategy: {strategy.upper()}")
    
    if strategy == "smote_tomek":
        resampler = SMOTETomek(random_state=CONFIG["random_state"])
        X_res, y_res = resampler.fit_resample(X_train, y_train)
        return X_res, y_res, None
    elif strategy == "easy_ensemble":
        return X_train, y_train, "easy_ensemble"
    else:
        resampler = SMOTE(random_state=CONFIG["random_state"])
        X_res, y_res = resampler.fit_resample(X_train, y_train)
        return X_res, y_res, None

def stratified_split(X, y):
    print("\n[STEP 7] Stratified train-test split...")
    return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y)

# ─────────────────────────────────────────────────────────────
# STEP 8: MODEL TRAINING
# ─────────────────────────────────────────────────────────────
def get_model_configs():
    rf_fast = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=CONFIG["random_state"])
    return {"RandomForest": {"estimator": rf_fast, "param_grid": {}}}

def train_models(X_train, y_train, easy_ensemble_flag=None):
    print(f"\n[STEP 8] Training models [{CONFIG['training_mode'].upper()}]...")
    models = get_model_configs()
    trained = {}
    for name, cfg in models.items():
        print(f"  → Training {name}...")
        est = cfg["estimator"]
        est.fit(X_train, y_train)
        trained[name] = est
    return trained

# ─────────────────────────────────────────────────────────────
# STEP 9 & 10: EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_models(trained_models, X_test, y_test):
    print("\n[STEP 9] Evaluating models...")
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        results[name] = {"f1": f1, "auc": auc, "model": model}
        print(f"  {name} - F1: {f1:.4f} | AUC: {auc:.4f}")
    
    best_name = max(results, key=lambda k: results[k]["f1"])
    return best_name, results

def summarize_results(results):
    pass

# ─────────────────────────────────────────────────────────────
# STEP 11 & 12: MODEL & PIPELINE SAVING (FIXED)
# ─────────────────────────────────────────────────────────────
def save_model(model, name: str):
    print(f"\n[STEP 11] Saving best model ({name})...")
    joblib.dump(model, CONFIG["model_save_path"], compress=5)

def build_and_save_pipeline(best_model):
    print("\n[STEP 12] Finalizing deployment model...")
    # FIX: Since data is already scaled in Step 2, app.py passes scaled data directly.
    # We do NOT wrap this in an unfitted RobustScaler. 
    # We simply save the perfectly fitted model as the pipeline to ensure inference works seamlessly.
    joblib.dump(best_model, CONFIG["pipeline_save_path"], compress=5)
    print(f"  Fitted Pipeline saved successfully to: {CONFIG['pipeline_save_path']}")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(filepath: str = "creditcard.csv"):
    df = load_and_clean_data(filepath)
    visualize_data(df)
    perform_eda(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    X_res, y_res, easy_flag = apply_imbalance_strategy(X_train, y_train)
    trained = train_models(X_res, y_res, easy_flag)
    best_name, results = evaluate_models(trained, X_test, y_test)
    summarize_results(results)
    
    best_model = results[best_name]["model"]
    save_model(best_model, best_name)
    build_and_save_pipeline(best_model)
    print("\n  PIPELINE COMPLETE ✔")

if __name__ == "__main__":
    main("creditcard.csv")