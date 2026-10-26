"""Model loading, inference and the Prediction page."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

from modules.data_processing import (
    MODEL_FEATURES, prepare_features, queue_category,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.joblib"
INFO_PATH = BASE_DIR / "models" / "model_info.json"
CASES_PATH = BASE_DIR / "models" / "test_cases.csv"

HIGH_RISK_DEFAULT = 0.70
MEDIUM_RISK_DEFAULT = 0.40

LIMITATIONS = [
    "The model is a small Decision Tree trained on completed requests only; "
    "it may not generalize to new offices, services or semesters.",
    "It uses only information known at submission time (service, requester, "
    "channel, priority, office, document status, weekday, queue length, SLA days).",
    "Probabilities from a shallow tree are coarse and are not calibrated "
    "confidence scores.",
    "Predictions are decision-support only. A staff member must review the "
    "request and make the final decision; the model never approves, rejects "
    "or denies any service.",
]


# ---------------------------------------------------------------------------
# Loading (cached so the model is NOT retrained or reloaded on every refresh)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_info() -> dict:
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


def model_files_exist() -> bool:
    return MODEL_PATH.exists() and INFO_PATH.exists()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def positive_index(model) -> int:
    return list(model.classes_).index(1)


def predict_dataframe(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (labels, breach_probabilities) using the saved pipeline."""
    X = prepare_features(df)
    labels = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, positive_index(model)]
    return labels, proba


def risk_level(prob: float, high: float = HIGH_RISK_DEFAULT,
               medium: float = MEDIUM_RISK_DEFAULT) -> str:
    if prob >= high:
        return "High"
    if prob >= medium:
        return "Medium"
    return "Low"


def build_input_row(values: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact training feature names."""
    row = dict(values)
    row["Submission_Weekday"] = pd.Timestamp(row.pop("Submission_Date")).day_name()
    return pd.DataFrame([row])[MODEL_FEATURES]


# ---------------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------------
def _model_banner(info: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model version", info["model_version"])
    c2.metric("Trained at", info["trained_at"])
    c3.metric("Training rows", info["n_train"])
    c4.metric("scikit-learn", info["sklearn_version"])
    st.caption(f"Data source: {info['data_source']}")
    if info["data_source"].lower().startswith("synthetic"):
        st.warning("This model was trained on **synthetic sample data**. Run "
                   "`python train_model.py <your_midterm_csv>` and commit the new "
                   "`models/` files to use your own midterm results.")
    if info["sklearn_version"] != sklearn.__version__:
        st.warning(f"The model was saved with scikit-learn {info['sklearn_version']} "
                   f"but this app runs {sklearn.__version__}. Pin the same version "
                   "in requirements.txt.")


def _single_prediction_tab(model, info: dict) -> None:
    options = info["category_options"]
    st.subheader("Predict one request")
    st.caption("Enter the details that are known when the request is submitted.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        service = c1.selectbox("Service type", options["Service_Type"])
        requester = c2.selectbox("Requester type", options["Requester_Type"])
        channel = c3.selectbox("Requested channel", options["Requested_Channel"])

        c4, c5, c6 = st.columns(3)
        priority = c4.selectbox("Priority", options["Priority"])
        office = c5.selectbox("Assigned office", options["Assigned_Office"])
        docs = c6.selectbox("Required documents complete?",
                            options["Required_Documents_Complete"])

        c7, c8, c9 = st.columns(3)
        submitted = c7.date_input("Submission date", value=datetime.today())
        queue_len = c8.number_input("Queue length", min_value=0, max_value=500,
                                    value=10, step=1)
        sla_days = c9.number_input("SLA days", min_value=1, max_value=60,
                                   value=5, step=1)
        go = st.form_submit_button("Predict SLA breach")

    if go:
        row = build_input_row({
            "Service_Type": service, "Requester_Type": requester,
            "Requested_Channel": channel, "Priority": priority,
            "Assigned_Office": office, "Required_Documents_Complete": docs,
            "Submission_Date": submitted, "Queue_Length": float(queue_len),
            "SLA_Days": float(sla_days),
        })
        label, proba = predict_dataframe(model, row)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        breach = int(label[0]) == 1
        level = risk_level(float(proba[0])) if proba is not None else "n/a"

        st.subheader("Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted class", "SLA breach" if breach else "No breach")
        if proba is not None:
            r2.metric("Breach probability", f"{proba[0]:.1%}")
        r3.metric("Risk level", level)
        r4.metric("Queue category", queue_category(queue_len))
        if breach:
            st.error("Likely SLA breach - recommend staff review and follow-up.")
        else:
            st.success("SLA breach not expected - routine handling can continue.")
        st.caption(f"Model version {info['model_version']} | Predicted at {stamp}")
        st.warning("Human review required: this prediction supports a decision, "
                   "it does not make one.")

        log = st.session_state.setdefault("prediction_log", [])
        log.append({
            "Predicted_At": stamp, "Model_Version": info["model_version"],
            "Service_Type": service, "Priority": priority, "Office": office,
            "Documents_Complete": docs, "Queue_Length": int(queue_len),
            "SLA_Days": int(sla_days),
            "Predicted_Class": "Breach" if breach else "No breach",
            "Breach_Probability": round(float(proba[0]), 3) if proba is not None else None,
            "Risk_Level": level,
        })

    with st.expander("Limitations"):
        for item in LIMITATIONS:
            st.write("- " + item)

    log = st.session_state.get("prediction_log", [])
    if log:
        st.subheader("Prediction history (this session)")
        hist = pd.DataFrame(log)
        st.dataframe(hist)
        d1, d2 = st.columns([1, 5])
        d1.download_button("Download history", hist.to_csv(index=False).encode("utf-8"),
                           "prediction_history.csv", "text/csv")
        if d2.button("Clear history"):
            st.session_state["prediction_log"] = []
            st.rerun()


def _metrics_tab(info: dict) -> None:
    st.subheader("Model metrics (held-out test set)")
    st.caption(f"{info['n_train']} training rows, {info['n_test']} test rows. "
               "Positive class = SLA breach.")
    comp = pd.DataFrame({"Decision Tree": info["metrics"],
                         "Majority baseline": info["baseline_metrics"]}).round(3)
    st.dataframe(comp)

    left, right = st.columns(2)
    with left:
        cm = np.array(info["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["No Breach (0)", "Breach (1)"], y=["No Breach (0)", "Breach (1)"],
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title="Confusion matrix")
        st.plotly_chart(fig)
    with right:
        imp = (pd.Series(info["feature_importance"]).sort_values()
               .rename("Importance").reset_index().rename(columns={"index": "Predictor"}))
        fig2 = px.bar(imp, x="Importance", y="Predictor", orientation="h",
                      title="Feature importance (grouped by predictor)")
        st.plotly_chart(fig2)

    report = pd.DataFrame(info["classification_report"]).T.round(3)
    st.markdown("**Classification report**")
    st.dataframe(report)
    st.markdown("**Features used (no leakage):** " + ", ".join(info["features"]))
    st.caption("Excluded because they are only known after handling: "
               "Processing_Days, Status, Reopened, Satisfaction_Score.")


def _consistency_tab(model, info: dict) -> None:
    st.subheader("Consistency check against the training run")
    st.write("The held-out test cases saved at training time are scored again by the "
             "loaded model. The predictions must match exactly.")
    if not CASES_PATH.exists():
        st.error("models/test_cases.csv was not found. Re-run train_model.py.")
        return
    cases = pd.read_csv(CASES_PATH)
    labels, proba = predict_dataframe(model, cases)
    same_label = labels == cases["Saved_Prediction"].to_numpy()
    same_proba = (np.isclose(proba, cases["Saved_Probability"].to_numpy(), atol=1e-5)
                  if proba is not None else np.ones(len(cases), dtype=bool))
    ok = same_label & same_proba

    c1, c2, c3 = st.columns(3)
    c1.metric("Test cases", len(cases))
    c2.metric("Matching predictions", int(ok.sum()))
    acc = float((labels == cases["Actual_SLA_Breached"].to_numpy()).mean())
    c3.metric("Accuracy on these cases", f"{acc:.3f}")
    if ok.all():
        st.success("All predictions match the saved training results.")
    else:
        st.error(f"{int((~ok).sum())} predictions differ. Check the scikit-learn version.")

    show = cases.copy()
    show["Reloaded_Prediction"] = labels
    show["Reloaded_Probability"] = np.round(proba, 6) if proba is not None else None
    show["Match"] = ok
    st.dataframe(show)


def render_prediction_page() -> None:
    st.header("Predictive Modeling")
    if not model_files_exist():
        st.error("The saved model was not found. Run `python train_model.py` and "
                 "commit models/trained_model.joblib and models/model_info.json.")
        return

    model = load_model()
    info = load_model_info()
    _model_banner(info)

    tab1, tab2, tab3 = st.tabs(["Predict", "Model metrics", "Consistency check"])
    with tab1:
        _single_prediction_tab(model, info)
    with tab2:
        _metrics_tab(info)
    with tab3:
        _consistency_tab(model, info)
