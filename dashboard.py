"""Decision-support dashboard: KPIs, risk/review queue and human decisions."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import CLOSED_STATUSES, TARGET, require_dataset
from modules.prediction import (
    HIGH_RISK_DEFAULT, MEDIUM_RISK_DEFAULT, load_model, load_model_info,
    model_files_exist, predict_dataframe, risk_level,
)

RISK_ORDER = ["High", "Medium", "Low"]
RISK_COLORS = {"High": "indianred", "Medium": "orange", "Low": "seagreen"}
DECISIONS = [
    "Escalate to office head",
    "Reassign / add staff to the queue",
    "Contact requester for missing documents",
    "Monitor only",
    "Override model - no action needed",
]


def recommended_follow_up(row: pd.Series) -> str:
    """Rule-based suggestion that a staff member can accept or change."""
    level = row["Risk_Level"]
    if level == "Low":
        return "Routine processing; no extra action."
    steps = []
    if str(row["Required_Documents_Complete"]) == "No":
        steps.append("Contact the requester to complete the documents")
    if pd.notna(row["Queue_Length"]) and row["Queue_Length"] > 15:
        steps.append(f"Check workload at {row['Assigned_Office']} and consider reassigning")
    if level == "High":
        steps.append("Escalate to the office head today")
    else:
        steps.append("Monitor daily until completed")
    return "; ".join(steps) + "."


def score_requests(df: pd.DataFrame, model, high: float, medium: float) -> pd.DataFrame:
    labels, proba = predict_dataframe(model, df)
    out = df[["Request_ID", "Service_Type", "Priority", "Assigned_Office",
              "Queue_Length", "SLA_Days", "Required_Documents_Complete", "Status"]].copy()
    out["Predicted_Breach"] = labels
    out["Breach_Probability"] = proba
    out["Risk_Level"] = [risk_level(p, high, medium) for p in proba]
    out["Recommended_Follow_Up"] = out.apply(recommended_follow_up, axis=1)
    return out


def _interpretations(clean, model_df, queue, open_note) -> list[str]:
    notes = []
    if not model_df.empty:
        rate = model_df[TARGET].astype(float).mean() * 100
        notes.append(f"Observed SLA breach rate for completed requests is **{rate:.1f}%**.")
        by_service = model_df.groupby("Service_Type")[TARGET].apply(lambda s: s.astype(float).mean())
        notes.append(f"**{by_service.idxmax()}** has the highest observed breach rate "
                     f"({by_service.max() * 100:.1f}%).")
    high = queue[queue["Risk_Level"] == "High"]
    notes.append(f"**{len(high)}** of **{len(queue)}** {open_note} are High risk "
                 "and should be reviewed first.")
    if not high.empty:
        top_office = high["Assigned_Office"].value_counts()
        notes.append(f"**{top_office.index[0]}** holds the most High-risk requests "
                     f"({int(top_office.iloc[0])}).")
        no_docs = int((high["Required_Documents_Complete"] == "No").sum())
        if no_docs:
            notes.append(f"{no_docs} High-risk requests have incomplete documents - "
                         "contacting the requester may prevent a breach.")
    notes.append("Risk levels come from the model's breach probability; staff judgment "
                 "decides the final action.")
    return notes


def _decision_section(queue: pd.DataFrame, info: dict) -> None:
    st.subheader("Record a human review decision")
    st.caption("The model only recommends. A staff member records the final decision. "
               "Decisions are kept for this browser session only (a production system "
               "would store them in a database).")
    candidates = queue.sort_values("Breach_Probability", ascending=False)
    if candidates.empty:
        st.info("No requests available for review.")
        return

    with st.form("decision_form", clear_on_submit=True):
        req_id = st.selectbox("Request", candidates["Request_ID"].tolist())
        reviewer = st.text_input("Reviewer name")
        decision = st.selectbox("Final decision", DECISIONS)
        notes = st.text_area("Notes (optional)")
        saved = st.form_submit_button("Save decision")

    if saved:
        if not reviewer.strip():
            st.error("Please enter the reviewer name before saving a decision.")
        else:
            rec = candidates.loc[candidates["Request_ID"] == req_id].iloc[0]
            st.session_state.setdefault("decision_log", []).append({
                "Recorded_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Request_ID": req_id,
                "Reviewer": reviewer.strip(),
                "Model_Probability": round(float(rec["Breach_Probability"]), 3),
                "Model_Risk_Level": rec["Risk_Level"],
                "Model_Version": info["model_version"],
                "Final_Decision": decision,
                "Notes": notes.strip(),
            })
            st.success(f"Decision saved for {req_id}.")

    log = st.session_state.get("decision_log", [])
    if log:
        st.markdown("**Decisions recorded this session**")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df)
        st.download_button("Download decision log", log_df.to_csv(index=False).encode("utf-8"),
                           "decision_log.csv", "text/csv")


def render_dashboard_page() -> None:
    st.header("Decision-Support Dashboard")
    data = require_dataset()
    if data is None:
        return
    if not model_files_exist():
        st.error("The saved model was not found. Run `python train_model.py` first.")
        return

    model, info = load_model(), load_model_info()
    clean, model_df = data["clean"], data["model_df"]

    with st.expander("Risk thresholds"):
        high = st.slider("High risk at or above", 0.50, 0.95, HIGH_RISK_DEFAULT, 0.05)
        medium = st.slider("Medium risk at or above", 0.10, float(high), MEDIUM_RISK_DEFAULT, 0.05)

    open_df = clean[~clean["Status"].isin(CLOSED_STATUSES) & clean["Status"].notna()]
    open_note = "open requests"
    if open_df.empty:
        open_df = clean
        open_note = "requests (no open requests found, so all were scored)"
        st.info("The dataset has no open requests, so every request was scored for demonstration.")

    queue = score_requests(open_df, model, high, medium)

    # KPIs
    breach_rate = model_df[TARGET].astype(float).mean() * 100 if not model_df.empty else 0.0
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total requests", f"{len(clean):,}")
    k2.metric("Completed", f"{int((clean['Status'] == 'Completed').sum()):,}")
    k3.metric("Observed breach rate", f"{breach_rate:.1f}%")
    k4.metric("Open requests", f"{len(open_df):,}")
    k5.metric("High-risk open", int((queue["Risk_Level"] == "High").sum()))
    k6.metric("Model accuracy", f"{info['metrics']['Accuracy']:.2f}")
    st.caption(f"Model version {info['model_version']} | Scored at "
               f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: {data['name']}")

    st.subheader("Interpretation")
    for note in _interpretations(clean, model_df, queue, open_note):
        st.write("- " + note)

    # Charts
    st.subheader("Workload and risk")
    c1, c2 = st.columns(2)
    risk_counts = (queue["Risk_Level"].value_counts().reindex(RISK_ORDER, fill_value=0)
                   .rename_axis("Risk_Level").reset_index(name="Requests"))
    c1.plotly_chart(px.bar(risk_counts, x="Risk_Level", y="Requests", color="Risk_Level",
                           color_discrete_map=RISK_COLORS, text="Requests",
                           title="Review queue by risk level"))
    by_office = (queue.groupby(["Assigned_Office", "Risk_Level"]).size()
                 .reset_index(name="Requests"))
    c2.plotly_chart(px.bar(by_office, x="Assigned_Office", y="Requests", color="Risk_Level",
                           color_discrete_map=RISK_COLORS,
                           category_orders={"Risk_Level": RISK_ORDER},
                           title="Open workload by office and risk"))

    # Review queue
    st.subheader("Risk / review queue")
    f1, f2 = st.columns(2)
    sel_risk = f1.multiselect("Risk level", RISK_ORDER, default=["High", "Medium"])
    offices = sorted(queue["Assigned_Office"].dropna().unique())
    sel_office = f2.multiselect("Assigned office", offices, default=offices)
    shown = queue[queue["Risk_Level"].isin(sel_risk) & queue["Assigned_Office"].isin(sel_office)]
    shown = shown.sort_values("Breach_Probability", ascending=False)
    st.dataframe(
        shown,
        column_config={
            "Breach_Probability": st.column_config.ProgressColumn(
                "Breach probability", min_value=0.0, max_value=1.0, format="%.2f"),
        },
        hide_index=True,
    )
    st.download_button("Download review queue", shown.to_csv(index=False).encode("utf-8"),
                       "review_queue.csv", "text/csv")
    st.warning("Human review required: these results are decision support only and must "
               "not be used to deny or reject a student service automatically.")

    _decision_section(queue, info)
