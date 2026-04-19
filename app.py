"""Streamlit application for customer churn prediction and agentic retention recommendations."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd
import streamlit as st

from agents.retention_agent import CHURN_THRESHOLD, get_retention_plan

ARTIFACTS_DIR = Path("artifacts")
DEMO_DATA_PATH = Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
TARGET_COLUMNS = ["Churn", "churn", "target", "label"]
ID_COLUMNS = ["customerID", "CustomerID", "CustomerId"]
NOISY_MISSING_TOKENS = {"", "na", "n/a", "none", "null", "?", "nan", "-"}
NUMERIC_HINT_COLUMNS = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}

st.set_page_config(page_title="Customer Churn Intelligence Suite", page_icon="📉", layout="wide")
st.title("Customer Churn Intelligence Suite")
st.caption("Individual prediction + extension dashboard for batch-level action.")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    fi_path = ARTIFACTS_DIR / "feature_importance.csv"
    feature_importance = pd.read_csv(fi_path) if fi_path.exists() else None
    return model, feature_names, feature_importance


@st.cache_data
def load_demo_data() -> pd.DataFrame:
    if DEMO_DATA_PATH.exists():
        return pd.read_csv(DEMO_DATA_PATH)
    return pd.DataFrame()


def standardize_feature_importance_columns(fi: pd.DataFrame) -> pd.DataFrame:
    fi = fi.copy()
    if fi.empty:
        return fi
    if "feature" not in fi.columns:
        text_cols = [c for c in fi.columns if fi[c].dtype == "object"]
        if text_cols:
            fi = fi.rename(columns={text_cols[0]: "feature"})
    if "importance" not in fi.columns:
        num_cols = [c for c in fi.columns if np.issubdtype(fi[c].dtype, np.number)]
        if num_cols:
            fi = fi.rename(columns={num_cols[0]: "importance"})
    return fi


def robust_preprocess(raw_df: pd.DataFrame, expected_features: list[str]):
    """Prepares noisy input for model scoring and records what was imputed."""
    work = raw_df.copy()
    warnings = []

    dropped_target = [c for c in TARGET_COLUMNS if c in work.columns]
    if dropped_target:
        work = work.drop(columns=dropped_target)
        warnings.append(f"Dropped target columns from input: {', '.join(dropped_target)}")

    # Keep a model-ready view and a human-readable payload for the LLM.
    model_frame = work.copy()

    for col in model_frame.columns:
        if model_frame[col].dtype == object:
            series = model_frame[col].astype(str).str.strip()
            noisy_mask = series.str.lower().isin(NOISY_MISSING_TOKENS)
            if noisy_mask.any():
                warnings.append(f"{col}: replaced {int(noisy_mask.sum())} noisy entries with missing values")
                series = series.mask(noisy_mask)
            model_frame[col] = series

    for col in model_frame.columns:
        if col in NUMERIC_HINT_COLUMNS:
            model_frame[col] = pd.to_numeric(model_frame[col], errors="coerce")

    # Column-wise imputation so data always reaches the model safely.
    for col in model_frame.columns:
        missing_count = int(model_frame[col].isna().sum())
        if missing_count == 0:
            continue

        if np.issubdtype(model_frame[col].dtype, np.number):
            fill_value = float(model_frame[col].median()) if not model_frame[col].dropna().empty else 0.0
            model_frame[col] = model_frame[col].fillna(fill_value)
            warnings.append(f"{col}: imputed {missing_count} missing numeric values with median ({fill_value:.2f})")
        else:
            non_null = model_frame[col].dropna()
            fill_value = str(non_null.mode().iloc[0]) if not non_null.empty else "Unknown"
            model_frame[col] = model_frame[col].fillna(fill_value)
            warnings.append(f"{col}: imputed {missing_count} missing categorical values with mode ('{fill_value}')")

    encoded = pd.get_dummies(model_frame)
    missing_features = [c for c in expected_features if c not in encoded.columns]
    unexpected_features = [c for c in encoded.columns if c not in expected_features]
    aligned = encoded.reindex(columns=expected_features, fill_value=0)

    if missing_features:
        warnings.append(
            f"Aligned schema by adding {len(missing_features)} expected model features not found in input"
        )
    if unexpected_features:
        warnings.append(
            f"Ignored {len(unexpected_features)} extra generated features not used by the trained model"
        )

    # Create LLM payload with imputed values reflected.
    payload_frame = model_frame.copy()
    for col in ID_COLUMNS:
        if col in work.columns and col not in payload_frame.columns:
            payload_frame[col] = work[col]

    return aligned, payload_frame, warnings


def render_risk_gauge(probability: float):
    prob = float(np.clip(probability, 0.0, 1.0))

    if prob < CHURN_THRESHOLD:
        color = "#1f9d55"
        risk_band = "Low"
    elif prob < 0.6:
        color = "#e3a008"
        risk_band = "Moderate"
    else:
        color = "#d64545"
        risk_band = "High"

    fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"aspect": "equal"})
    ax.add_patch(Wedge((0, 0), 1.0, 180, 360, width=0.28, color="#e5e7eb"))
    ax.add_patch(Wedge((0, 0), 1.0, 180, 180 + (180 * prob), width=0.28, color=color))

    ax.text(0, -0.05, f"{prob * 100:.1f}%", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0, -0.28, f"{risk_band} Risk", ha="center", va="center", fontsize=12, color=color)
    ax.text(-1.02, -0.02, "0%", fontsize=9, color="#4b5563")
    ax.text(0.97, -0.02, "100%", fontsize=9, color="#4b5563", ha="right")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 0.8)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)


def display_agent_report(report: dict):
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Risk Summary")
        st.markdown(report.get("risk_summary", "No summary available."))

        with st.expander("Contributing Factors", expanded=True):
            factors = report.get("contributing_factors", [])
            if factors:
                st.markdown("\n".join([f"- {factor}" for factor in factors]))
            else:
                st.markdown("- No material factors identified.")

    with right:
        with st.expander("Recommended Actions", expanded=True):
            actions = report.get("recommended_actions", [])
            if actions:
                st.markdown("\n".join([f"1. {action}" for action in actions]))
            else:
                st.markdown("1. Continue standard retention cadence.")

        st.info(report.get("disclaimer", "AI-generated recommendation. Verify with business policy."))


def get_primary_churn_factor(at_risk_features: pd.DataFrame, feature_importance: pd.DataFrame) -> str:
    if at_risk_features.empty or feature_importance is None or feature_importance.empty:
        return "Unavailable"

    fi = standardize_feature_importance_columns(feature_importance)
    if "feature" not in fi.columns or "importance" not in fi.columns:
        return "Unavailable"

    merged = fi[["feature", "importance"]].copy()
    merged = merged[merged["feature"].isin(at_risk_features.columns)]
    if merged.empty:
        return "Unavailable"

    prevalence = at_risk_features[merged["feature"]].mean(axis=0)
    merged["score"] = merged["feature"].map(prevalence).fillna(0) * merged["importance"].astype(float)
    top_feature = merged.sort_values("score", ascending=False).iloc[0]["feature"]
    return str(top_feature).replace("_", " ")


def run_agent_for_customer(payload: dict) -> dict:
    return get_retention_plan(payload, str(MODEL_PATH))


model, feature_names, feature_importance = load_artifacts()
demo_df = load_demo_data()

tab_individual, tab_batch = st.tabs(["Individual Prediction", "Batch Dashboard (Extension)"])

with tab_individual:
    st.subheader("Single Customer Prediction")
    source = st.radio(
        "Choose customer input method",
        ["Select from demo dropdown", "Upload single-customer CSV"],
        horizontal=True,
    )

    selected_customer_df = None
    if source == "Select from demo dropdown":
        if demo_df.empty:
            st.warning("Demo dataset not found in data/ folder.")
        else:
            id_col = "customerID" if "customerID" in demo_df.columns else None
            index_options = demo_df.index.tolist()

            def label_fn(idx):
                if id_col:
                    return f"{demo_df.loc[idx, id_col]} | tenure={demo_df.loc[idx, 'tenure']}"
                return f"Row {idx}"

            selected_idx = st.selectbox("Pick a customer", options=index_options, format_func=label_fn)
            selected_customer_df = demo_df.loc[[selected_idx]].copy()
    else:
        uploaded_single = st.file_uploader("Upload a CSV (one or more rows)", type=["csv"], key="single_csv")
        if uploaded_single is not None:
            incoming = pd.read_csv(uploaded_single)
            if incoming.empty:
                st.error("Uploaded file has no rows.")
            else:
                row_idx = st.selectbox("Choose row index", options=incoming.index.tolist())
                selected_customer_df = incoming.loc[[row_idx]].copy()

    if selected_customer_df is not None and not selected_customer_df.empty:
        st.markdown("#### Customer Snapshot")
        st.dataframe(selected_customer_df, use_container_width=True)

        if st.button("Predict + Generate Agentic Report", type="primary"):
            X_single, payload_single_df, prep_warnings = robust_preprocess(selected_customer_df, feature_names)

            if prep_warnings:
                st.warning("Input data required cleaning/imputation before scoring.")
                with st.expander("See preprocessing warnings"):
                    for warning in prep_warnings:
                        st.markdown(f"- {warning}")

            pred = int(model.predict(X_single)[0])
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_single)[0][1])
            else:
                proba = float(pred)

            left, right = st.columns([1.1, 1])
            with left:
                st.markdown("#### Mid-Sem ML Prediction")
                render_risk_gauge(proba)
            with right:
                st.metric("Predicted Label", "Churn" if pred == 1 else "Stay")
                st.metric("Risk Probability", f"{proba * 100:.2f}%")
                st.metric("High-Risk Threshold", f"{CHURN_THRESHOLD * 100:.0f}%")

            st.markdown("#### Agentic Retention Plan")
            try:
                payload = payload_single_df.iloc[0].to_dict()
                report = run_agent_for_customer(payload)
                display_agent_report(report)
            except Exception as exc:
                st.error(
                    "Agent report generation failed. Check API keys/vector store setup and try again. "
                    f"Error: {exc}"
                )

with tab_batch:
    st.subheader("Batch-Level Extension Dashboard")
    st.caption("Upload a full cohort, review aggregate churn risk, and generate reports only for selected at-risk customers.")

    uploaded_batch = st.file_uploader("Upload cohort CSV", type=["csv"], key="batch_csv")
    use_demo_batch = st.button("Use sample Telco dataset for batch")

    batch_df = None
    if uploaded_batch is not None:
        batch_df = pd.read_csv(uploaded_batch)
    elif use_demo_batch and not demo_df.empty:
        batch_df = demo_df.copy()

    if batch_df is not None and not batch_df.empty:
        st.markdown("#### Batch Preview")
        st.dataframe(batch_df.head(20), use_container_width=True)

        X_batch, payload_batch_df, prep_warnings = robust_preprocess(batch_df, feature_names)

        if prep_warnings:
            st.warning("Batch input contained missing/noisy fields; values were imputed before scoring.")
            with st.expander("Batch preprocessing details"):
                for warning in prep_warnings:
                    st.markdown(f"- {warning}")

        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

        results = batch_df.copy()
        results["PredictedChurn"] = preds
        results["RiskProbability"] = np.round(probs, 4)

        at_risk_mask = results["RiskProbability"] >= CHURN_THRESHOLD
        at_risk_results = results[at_risk_mask].copy()
        at_risk_features = X_batch.loc[at_risk_mask].copy()

        total = len(results)
        total_at_risk = int(at_risk_mask.sum())
        at_risk_rate = (total_at_risk / total) * 100 if total else 0
        primary_factor = get_primary_churn_factor(at_risk_features, feature_importance)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Customers", f"{total:,}")
        m2.metric("Total at Risk", f"{total_at_risk:,}")
        m3.metric("At-Risk Rate", f"{at_risk_rate:.2f}%")
        st.info(f"Primary Churn Factor Across Cohort: {primary_factor}")

        st.markdown("#### At-Risk Customers")
        if at_risk_results.empty:
            st.success("No customers exceeded the at-risk threshold in this batch.")
        else:
            id_column = next((col for col in ID_COLUMNS if col in at_risk_results.columns), None)
            display_cols = [id_column] if id_column else []
            display_cols += ["RiskProbability"]
            st.dataframe(
                at_risk_results[display_cols + [c for c in at_risk_results.columns if c not in display_cols]].head(200),
                use_container_width=True,
            )

            selectable_indices = at_risk_results.index.tolist()

            def risk_label(idx):
                cid = str(at_risk_results.loc[idx, id_column]) if id_column else f"row {idx}"
                rp = float(at_risk_results.loc[idx, "RiskProbability"])
                return f"{cid} | risk={rp:.2%}"

            selected_idx = st.selectbox(
                "Select one at-risk customer for agentic report",
                options=selectable_indices,
                format_func=risk_label,
            )

            if st.button("Generate Agentic Report for Selected Customer", type="primary"):
                try:
                    selected_payload = payload_batch_df.loc[selected_idx].to_dict()
                    selected_report = run_agent_for_customer(selected_payload)
                    st.markdown("#### Selected Customer Retention Report")
                    display_agent_report(selected_report)
                except Exception as exc:
                    st.error(
                        "Failed to generate selected customer report. "
                        f"Error: {exc}"
                    )

        st.download_button(
            "Download batch predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="batch_churn_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a cohort CSV or click 'Use sample Telco dataset for batch'.")

st.markdown("---")
st.caption("Built by Team Eden | Individual + Extension Dashboard")