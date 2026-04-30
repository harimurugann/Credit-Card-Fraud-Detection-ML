# ============================================================
# END-TO-END HYBRID ML PIPELINE: CREDIT CARD FRAUD DETECTION
# Author: AI Data Engineer
# Description: Production-grade fraud detection pipeline with
#              advanced imbalance handling, hybrid architecture,
#              and full evaluation suite.
# ============================================================

# ─────────────────────────────────────────────────────────────
# SECTION 1: IMPORTING DEPENDENCIES
# ─────────────────────────────────────────────────────────────
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Scikit-learn — preprocessing, modeling, evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)

# Gradient boosting libraries
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

# Imbalanced-learn — advanced resampling
from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional: Plotly for interactive plots (PC only)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION — Switch between Mobile & PC modes here
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Set to "mobile" for lightweight run, "pc" for full power
    "mode": "pc",

    # Imbalance strategy: "smote_tomek" | "easy_ensemble" | "smote"
    "imbalance_strategy": "smote_tomek",

    # Training mode: "fast" | "tuned"
    "training_mode": "fast",

    # Output directory for plots and models
    "output_dir": "outputs",

    # Random seed for reproducibility
    "random_state": 42,

    # Test split ratio
    "test_size": 0.2,

    # Plot DPI (72 for mobile, 150+ for PC)
    "plot_dpi": 150,

    # Use interactive Plotly charts (PC only)
    "interactive_plots": True,

    # Model save paths
    "model_save_path": "outputs/best_fraud_model.sav",
    "pipeline_save_path": "outputs/fraud_pipeline.sav",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"[INFO] Running in [{CONFIG['mode'].upper()}] mode | "
      f"Training: [{CONFIG['training_mode'].upper()}] | "
      f"Imbalance: [{CONFIG['imbalance_strategy'].upper()}]")


# ─────────────────────────────────────────────────────────────
# SECTION 2: DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset, remove duplicates, and scale 'Time' and 'Amount'
    using RobustScaler (resistant to outliers).

    Args:
        filepath: Path to creditcard.csv

    Returns:
        Cleaned DataFrame with scaled features.
    """
    print("\n[STEP 2] Loading and cleaning data...")
    df = pd.read_csv(filepath)
    print(f"  Raw shape       : {df.shape}")
    print(f"  Null values     : {df.isnull().sum().sum()}")

    # Remove exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicates removed: {before - len(df)}")
    print(f"  Clean shape     : {df.shape}")

    # Scale 'Time' and 'Amount' — V1-V28 are already PCA-transformed
    scaler = RobustScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"]   = scaler.fit_transform(df[["Time"]])

    # Drop original columns after scaling
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    # Class balance summary
    fraud_pct = df["Class"].value_counts(normalize=True)[1] * 100
    print(f"  Fraud ratio     : {fraud_pct:.4f}%")

    return df


# ─────────────────────────────────────────────────────────────
# SECTION 3: DATA VISUALIZATION
# ─────────────────────────────────────────────────────────────
def visualize_data(df: pd.DataFrame) -> None:
    """
    Generate class distribution bar chart and correlation heatmap.
    Saves PNG files and optionally shows Plotly interactive versions.
    """
    print("\n[STEP 3] Generating visualizations...")

    # ── 3a. Class Distribution ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             facecolor="#0d0d0d")
    fig.suptitle("Credit Card Fraud — Dataset Overview",
                 color="white", fontsize=15, fontweight="bold")

    counts = df["Class"].value_counts()
    colors = ["#00d4ff", "#ff4757"]

    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(["Legitimate (0)", "Fraudulent (1)"],
                  counts.values, color=colors,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{val:,}", ha="center", color="white", fontsize=10)
    ax.set_title("Class Distribution", color="white")
    ax.tick_params(colors="white")
    ax.set_ylabel("Count", color="white")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#444")

    # ── 3b. Fraud % Pie ──────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=["Legitimate", "Fraudulent"],
        autopct="%1.3f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=0.8)
    )
    for t in texts + autotexts:
        t.set_color("white")
    ax2.set_title("Class Proportion", color="white")

    plt.tight_layout()
    path = f"{CONFIG['output_dir']}/class_distribution.png"
    plt.savefig(path, dpi=CONFIG["plot_dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")

    # ── 3c. Correlation Heatmap ───────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 16), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        annot=False, linewidths=0.3, linecolor="#333",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap",
                 color="white", fontsize=14, pad=15)
    plt.xticks(rotation=45, color="white", fontsize=8)
    plt.yticks(rotation=0, color="white", fontsize=8)

    path = f"{CONFIG['output_dir']}/correlation_heatmap.png"
    plt.savefig(path, dpi=CONFIG["plot_dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")

    # ── 3d. Interactive Plotly (PC only) ──────────────────────
    if CONFIG["interactive_plots"] and PLOTLY_AVAILABLE:
        fig_plotly = px.bar(
            x=["Legitimate", "Fraudulent"],
            y=counts.values,
            color=["Legitimate", "Fraudulent"],
            color_discrete_map={"Legitimate": "#00d4ff",
                                "Fraudulent": "#ff4757"},
            title="Class Distribution (Interactive)",
            labels={"x": "Class", "y": "Count"},
            template="plotly_dark"
        )
        fig_plotly.write_html(
            f"{CONFIG['output_dir']}/class_distribution_interactive.html"
        )
        print(f"  Saved: {CONFIG['output_dir']}/class_distribution_interactive.html")


# ─────────────────────────────────────────────────────────────
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────
def perform_eda(df: pd.DataFrame) -> None:
    """
    Analyze feature distributions for Fraud vs Non-Fraud classes.
    Plots KDE overlays for the most discriminative V-features.
    """
    print("\n[STEP 4] Performing EDA...")

    fraud     = df[df["Class"] == 1]
    legit     = df[df["Class"] == 0]

    # Top discriminative features (by mean absolute difference)
    v_cols = [c for c in df.columns if c.startswith("V")]
    diff   = (fraud[v_cols].mean() - legit[v_cols].mean()).abs()
    top_features = diff.nlargest(12).index.tolist()

    fig, axes = plt.subplots(3, 4, figsize=(20, 12),
                             facecolor="#0d0d0d")
    fig.suptitle("Feature Distributions: Fraud vs Legitimate",
                 color="white", fontsize=14, fontweight="bold")

    for ax, feat in zip(axes.flatten(), top_features):
        ax.set_facecolor("#1a1a2e")
        sns.kdeplot(legit[feat],  ax=ax, label="Legit",
                    color="#00d4ff", fill=True, alpha=0.4, linewidth=1.5)
        sns.kdeplot(fraud[feat],  ax=ax, label="Fraud",
                    color="#ff4757", fill=True, alpha=0.4, linewidth=1.5)
        ax.set_title(feat, color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(fontsize=7, labelcolor="white",
                  facecolor="#222", edgecolor="#444")
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#444")

    plt.tight_layout()
    path = f"{CONFIG['output_dir']}/eda_feature_distributions.png"
    plt.savefig(path, dpi=CONFIG["plot_dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Top discriminative features: {top_features[:6]}")


# ─────────────────────────────────────────────────────────────
# SECTION 5 & 6: FEATURE/TARGET SPLIT + IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────
def split_features_target(df: pd.DataFrame):
    """Split DataFrame into features X and target y."""
    print("\n[STEP 6] Splitting features and target...")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print(f"  Feature shape: {X.shape} | Target shape: {y.shape}")
    return X, y


def apply_imbalance_strategy(X_train, y_train):
    """
    Apply the configured imbalance handling strategy.

    Strategies:
        smote_tomek    — SMOTE + Tomek Links (hybrid over+under sampling)
        easy_ensemble  — EasyEnsembleClassifier (ensemble-level balancing)
        smote          — Standard SMOTE only
    """
    strategy = CONFIG["imbalance_strategy"]
    print(f"\n[IMBALANCE] Applying strategy: {strategy.upper()}")

    if strategy == "smote_tomek":
        resampler = SMOTETomek(random_state=CONFIG["random_state"])
        X_res, y_res = resampler.fit_resample(X_train, y_train)
        print(f"  After SMOTETomek: {np.bincount(y_res)}")
        return X_res, y_res, None  # None = no wrapper needed

    elif strategy == "easy_ensemble":
        # EasyEnsemble wraps the estimator — return placeholder
        print("  EasyEnsembleClassifier will wrap the base estimator.")
        return X_train, y_train, "easy_ensemble"

    else:  # plain SMOTE
        resampler = SMOTE(random_state=CONFIG["random_state"])
        X_res, y_res = resampler.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {np.bincount(y_res)}")
        return X_res, y_res, None


# ─────────────────────────────────────────────────────────────
# SECTION 7: STRATIFIED TRAIN-TEST SPLIT
# ─────────────────────────────────────────────────────────────
def stratified_split(X, y):
    """
    Stratified split to preserve the fraud ratio in both sets.
    Critical for highly imbalanced datasets.
    """
    print("\n[STEP 7] Stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y  # ← preserves fraud ratio
    )
    print(f"  Train: {X_train.shape} | "
          f"Fraud in train: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"  Test : {X_test.shape}  | "
          f"Fraud in test : {y_test.sum()} ({y_test.mean()*100:.3f}%)")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# SECTION 8: MODEL DEFINITIONS & TRAINING
# ─────────────────────────────────────────────────────────────
def get_model_configs():
    """
    Return model definitions with two parameter sets:
        fast  — lightweight defaults (mobile-friendly)
        tuned — GridSearch / RandomizedSearch parameter grids (PC)
    """
    mode = CONFIG["mode"]

    # ── Random Forest ─────────────────────────────────────────
    rf_fast   = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1,
        random_state=CONFIG["random_state"], class_weight="balanced"
    )
    rf_grid   = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    models = {
        "RandomForest": {
            "estimator": rf_fast,
            "param_grid": rf_grid
        }
    }

    # ── XGBoost ───────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        xgb_fast = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=100,   # compensates for imbalance
            use_label_encoder=False, eval_metric="logloss",
            random_state=CONFIG["random_state"], n_jobs=-1
        )
        xgb_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0]
        }
        models["XGBoost"] = {"estimator": xgb_fast, "param_grid": xgb_grid}

    # ── LightGBM ──────────────────────────────────────────────
    if LGBM_AVAILABLE:
        lgbm_fast = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            is_unbalance=True,
            random_state=CONFIG["random_state"], n_jobs=-1,
            verbose=-1
        )
        lgbm_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63]
        }
        models["LightGBM"] = {"estimator": lgbm_fast,
                               "param_grid": lgbm_grid}

    return models


def train_models(X_train, y_train, easy_ensemble_flag=None):
    """
    Train all configured models.

    Modes:
        fast   — use default estimators directly
        tuned  — RandomizedSearchCV with StratifiedKFold (PC)
    """
    print(f"\n[STEP 8] Training models [{CONFIG['training_mode'].upper()}]...")
    model_configs = get_model_configs()
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=CONFIG["random_state"])

    for name, cfg in model_configs.items():
        print(f"  → Training {name}...")
        estimator = cfg["estimator"]

        # Wrap with EasyEnsemble if strategy requires it
        if easy_ensemble_flag == "easy_ensemble":
            estimator = EasyEnsembleClassifier(
                base_estimator=estimator,
                n_estimators=10,
                random_state=CONFIG["random_state"]
            )
            estimator.fit(X_train, y_train)
            trained_models[name] = estimator
            continue

        if CONFIG["training_mode"] == "tuned" and CONFIG["mode"] == "pc":
            # RandomizedSearchCV for speed vs full GridSearchCV
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=cfg["param_grid"],
                n_iter=10,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                random_state=CONFIG["random_state"],
                verbose=0
            )
            search.fit(X_train, y_train)
            trained_models[name] = search.best_estimator_
            print(f"     Best params: {search.best_params_}")
        else:
            # Fast mode — just fit
            estimator.fit(X_train, y_train)
            trained_models[name] = estimator

        print(f"     Done.")

    return trained_models


# ─────────────────────────────────────────────────────────────
# SECTION 9: MODEL EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_models(trained_models: dict, X_test, y_test) -> str:
    """
    Evaluate all models with:
        - Classification report (Precision, Recall, F1)
        - ROC-AUC score
        - Confusion matrix
        - Precision-Recall curve

    Returns:
        Name of the best model (by F1 on fraud class).
    """
    print("\n[STEP 9] Evaluating models...")
    results = {}

    n_models = len(trained_models)
    fig, axes = plt.subplots(2, n_models,
                             figsize=(7 * n_models, 12),
                             facecolor="#0d0d0d")
    if n_models == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Model Evaluation Dashboard",
                 color="white", fontsize=15, fontweight="bold")

    pr_fig, pr_ax = plt.subplots(figsize=(10, 7), facecolor="#0d0d0d")
    pr_ax.set_facecolor("#1a1a2e")
    colors_map = ["#00d4ff", "#ff6b35", "#2ed573", "#ff4757", "#ffa502"]

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred  = model.predict(X_test)
        y_proba = (model.predict_proba(X_test)[:, 1]
                   if hasattr(model, "predict_proba") else y_pred)

        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {"f1": f1, "auc": auc, "model": model}

        print(f"\n  ── {name} ──")
        print(classification_report(y_test, y_pred,
                                    target_names=["Legit", "Fraud"]))
        print(f"  ROC-AUC: {auc:.4f}")

        # ── Confusion Matrix ──────────────────────────────────
        cm  = confusion_matrix(y_test, y_pred)
        ax_cm = axes[0, idx]
        ax_cm.set_facecolor("#1a1a2e")
        disp = ConfusionMatrixDisplay(cm,
                                      display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title(f"{name}\nF1={f1:.3f} | AUC={auc:.3f}",
                        color="white", fontsize=10)
        ax_cm.tick_params(colors="white")
        ax_cm.xaxis.label.set_color("white")
        ax_cm.yaxis.label.set_color("white")
        for text in ax_cm.texts:
            text.set_color("white")

        # ── Precision-Recall Curve ────────────────────────────
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        color = colors_map[idx % len(colors_map)]
        pr_ax.plot(rec, prec, color=color, linewidth=2,
                   label=f"{name} (AP={ap:.3f})")

        # ── Feature Importance ────────────────────────────────
        ax_fi = axes[1, idx]
        ax_fi.set_facecolor("#1a1a2e")
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.Series(
                model.feature_importances_,
                index=X_test.columns
            ).nlargest(15)
            feat_imp.sort_values().plot(
                kind="barh", ax=ax_fi, color=color,
                edgecolor="white", linewidth=0.3
            )
            ax_fi.set_title(f"{name} — Top 15 Features",
                            color="white", fontsize=10)
            ax_fi.tick_params(colors="white", labelsize=7)
            ax_fi.set_xlabel("Importance", color="white")
            ax_fi.spines[["top", "right"]].set_visible(False)
        else:
            ax_fi.text(0.5, 0.5, "Feature importance\nnot available",
                       ha="center", va="center", color="white",
                       transform=ax_fi.transAxes)

    # Finish PR curve plot
    pr_ax.axhline(y=y_test.mean(), color="#ffa502", linestyle="--",
                  alpha=0.7, label="Baseline (random)")
    pr_ax.set_title("Precision-Recall Curve Comparison",
                    color="white", fontsize=13)
    pr_ax.set_xlabel("Recall", color="white")
    pr_ax.set_ylabel("Precision", color="white")
    pr_ax.tick_params(colors="white")
    pr_ax.legend(fontsize=9, labelcolor="white",
                 facecolor="#1a1a2e", edgecolor="#444")
    pr_ax.spines[["top", "right"]].set_visible(False)
    for spine in pr_ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    cm_path = f"{CONFIG['output_dir']}/confusion_matrices.png"
    pr_path = f"{CONFIG['output_dir']}/precision_recall_curves.png"
    fig.savefig(cm_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    pr_fig.savefig(pr_path, dpi=CONFIG["plot_dpi"], bbox_inches="tight",
                   facecolor=pr_fig.get_facecolor())
    plt.close("all")
    print(f"\n  Saved: {cm_path}")
    print(f"  Saved: {pr_path}")

    # Return best model name
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  ✔ Best model: {best_name} "
          f"(F1={results[best_name]['f1']:.4f} | "
          f"AUC={results[best_name]['auc']:.4f})")
    return best_name, results


# ─────────────────────────────────────────────────────────────
# SECTION 10: COMMUNICATION & SUMMARY VISUALIZATION
# ─────────────────────────────────────────────────────────────
def summarize_results(results: dict) -> None:
    """
    Generate a final summary bar chart comparing F1 and AUC
    across all trained models.
    """
    print("\n[STEP 10] Building results summary...")

    names   = list(results.keys())
    f1_vals = [results[n]["f1"]  for n in names]
    auc_vals= [results[n]["auc"] for n in names]
    x       = np.arange(len(names))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#1a1a2e")

    bars1 = ax.bar(x - width/2, f1_vals,  width, label="F1 Score",
                   color="#00d4ff", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, auc_vals, width, label="ROC-AUC",
                   color="#ff6b35", edgecolor="white", linewidth=0.5)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", color="white", fontsize=9)

    ax.set_title("Model Comparison: F1 Score vs ROC-AUC",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, color="white")
    ax.tick_params(colors="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color="white")
    ax.legend(fontsize=10, labelcolor="white",
              facecolor="#222", edgecolor="#444")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#444")

    path = f"{CONFIG['output_dir']}/model_comparison_summary.png"
    plt.savefig(path, dpi=CONFIG["plot_dpi"], bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# SECTION 11: MODEL SAVING
# ─────────────────────────────────────────────────────────────
def save_model(model, name: str) -> None:
    """
    Save the best model as a .sav file using joblib with
    compression level 5 (balance of speed and file size).
    """
    print(f"\n[STEP 11] Saving best model ({name})...")
    path = CONFIG["model_save_path"]
    joblib.dump(model, path, compress=5)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path} ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────
# SECTION 12 & 13: AUTOMATED PIPELINE + PIPELINE SAVING
# ─────────────────────────────────────────────────────────────
def build_and_save_pipeline(best_model) -> ImbPipeline:
    """
    Create an end-to-end sklearn-compatible Pipeline object:
        RobustScaler → SMOTETomek → BestModel

    This pipeline can be used directly for inference in app.py.
    """
    print("\n[STEP 12] Building automated deployment pipeline...")

    pipeline = ImbPipeline(steps=[
        ("scaler",    RobustScaler()),
        ("resampler", SMOTETomek(random_state=CONFIG["random_state"])),
        ("model",     best_model)
    ])

    # Save pipeline
    path = CONFIG["pipeline_save_path"]
    joblib.dump(pipeline, path, compress=5)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Pipeline saved: {path} ({size_kb:.1f} KB)")
    return pipeline


# ─────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────
def main(filepath: str = "creditcard.csv"):
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — ML PIPELINE")
    print("=" * 60)

    # Steps 2–4: Load, clean, visualize, EDA
    df = load_and_clean_data(filepath)
    visualize_data(df)
    perform_eda(df)

    # Steps 5–6: Split
    X, y = split_features_target(df)

    # Step 7: Stratified split
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # Imbalance handling
    X_res, y_res, easy_flag = apply_imbalance_strategy(X_train, y_train)

    # Step 8: Train
    trained_models = train_models(X_res, y_res, easy_flag)

    # Step 9: Evaluate
    best_name, results = evaluate_models(trained_models, X_test, y_test)

    # Step 10: Summarize
    summarize_results(results)

    # Step 11: Save best model
    best_model = results[best_name]["model"]
    save_model(best_model, best_name)

    # Steps 12–13: Pipeline
    build_and_save_pipeline(best_model)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE ✔")
    print(f"  All outputs in: ./{CONFIG['output_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    # ── Change this path to your dataset ──────────────────────
    main("creditcard.csv")
