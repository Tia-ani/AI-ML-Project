import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path("artifacts")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("Customer Churn Predictor")
st.caption("Upload a Telco CSV and get churn predictions + quick insights.")
st.markdown(
    "This model predicts whether a telecom customer is likely to churn based on demographic and service usage data."
)

# -----------------------------
# Load artefacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    label_encoders = joblib.load(ARTIFACTS_DIR / "label_encoders.pkl")  # dict-like: {col: encoder}
    feature_names = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())

    fi_path = ARTIFACTS_DIR / "feature_importance.csv"
    feature_importance = pd.read_csv(fi_path) if fi_path.exists() else None

    return model, label_encoders, feature_names, feature_importance

model, label_encoders, feature_names, feature_importance = load_artifacts()

# -----------------------------
# Utilities
# -----------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning for UI safety (doesn't replace your clean_data.py)."""
    df = df.copy()

    # drop obvious ID column if present
    for col in ["customerID", "CustomerID", "CustomerId"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # TotalCharges in Telco dataset sometimes blank -> coerce to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Some CSVs have extra spaces
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    return df

def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply saved LabelEncoders column-wise.
    Handles unseen categories by mapping them to the most frequent seen class (fallback).
    """
    df = df.copy()
    for col, le in encoders.items():
        if col not in df.columns:
            continue

        # Ensure str categories (common for LabelEncoder)
        values = df[col].astype(str)

        known = set(getattr(le, "classes_", []))
        if len(known) == 0:
            continue

        # fallback: use first known class
        fallback = list(known)[0]

        values_safe = values.apply(lambda x: x if x in known else fallback)
        df[col] = le.transform(values_safe)

    return df

def align_features(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Add missing expected cols as 0, drop extras, reorder."""
    df = df.copy()
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    return df

def plot_bar(counts: pd.Series, title: str, xlabel: str):
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.subheader("Demo file")
    if st.button("Use sample Telco dataset"):
        uploaded = str(Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    st.markdown("---")
    show_probability = st.toggle("Show churn probability (if supported)", value=True)

# -----------------------------
# Read data
# -----------------------------
df = None
if uploaded:
    if isinstance(uploaded, str):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(uploaded)

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")

    # Drop target if someone uploads labelled data
    for target_col in ["Churn", "churn", "target", "label"]:
        if target_col in df.columns:
            st.info(f"Detected target column `{target_col}` — it will be ignored for prediction.")
            df = df.drop(columns=[target_col])

    # -----------------------------
    # Prepare features
    # -----------------------------
    X = basic_clean(df)
    X = apply_label_encoders(X, label_encoders)
    X = align_features(X, feature_names)

    # -----------------------------
    # Predict
    # -----------------------------
    st.divider()
    st.markdown("## Predictions")
    preds = model.predict(X)

    proba = None
    if show_probability and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = None

    out = df.copy()
    out["PredictedChurn"] = preds
    if proba is not None:
        out["ChurnProbability"] = np.round(proba, 4)

    # KPIs
    churn_rate = float(np.mean(preds)) * 100.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted churn rate", f"{churn_rate:.2f}%")
    c2.metric("Customers predicted churn", f"{int(np.sum(preds)):,}")
    c3.metric("Customers predicted stay", f"{int(len(preds) - np.sum(preds)):,}")

    st.dataframe(out.head(50), use_container_width=True)

    # Download
    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Charts
    # -----------------------------
    st.markdown("## Charts")
    left, right = st.columns(2)

    with left:
        counts = pd.Series(preds).value_counts().sort_index()
        plot_bar(counts, "Predicted Class Distribution", "0 = No churn, 1 = Churn")

    with right:
        if proba is not None:
            fig, ax = plt.subplots()
            ax.hist(proba, bins=20)
            ax.set_title("Churn Probability Distribution")
            ax.set_xlabel("Probability of churn")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("Probability chart unavailable (model has no predict_proba).")

    # -----------------------------
    # Feature importance (if present)
    # -----------------------------
    st.markdown("## Model Explainability")
    if feature_importance is not None:
        st.caption("Top features (from your saved feature_importance.csv)")
        fi = feature_importance.copy()

        # Try to standardise column naming
        if "feature" not in fi.columns:
            # pick first text column as feature
            text_cols = [c for c in fi.columns if fi[c].dtype == "object"]
            if text_cols:
                fi = fi.rename(columns={text_cols[0]: "feature"})
        if "importance" not in fi.columns:
            num_cols = [c for c in fi.columns if np.issubdtype(fi[c].dtype, np.number)]
            if num_cols:
                fi = fi.rename(columns={num_cols[0]: "importance"})

        fi = fi.sort_values("importance", ascending=False).head(15)
        st.dataframe(fi, use_container_width=True)

        fig, ax = plt.subplots()
        ax.barh(fi["feature"][::-1], fi["importance"][::-1])
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    else:
        st.info("feature_importance.csv not found — skipping explainability section.")

else:
    st.info("Upload a CSV from the sidebar, or click **Use sample Telco dataset**.")
st.markdown("---")
st.caption("Built by Team Eden | Logistic Regression & Random Forest")