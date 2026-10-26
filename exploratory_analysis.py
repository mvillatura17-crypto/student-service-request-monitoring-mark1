"""Exploratory data analysis page with interactive Plotly charts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import NUMERIC_COLUMNS, TARGET, require_dataset

QUEUE_ORDER = ["Low", "Medium", "High"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LABELS = {0: "No breach", 1: "Breach"}


def _with_label(model_df: pd.DataFrame) -> pd.DataFrame:
    out = model_df.copy()
    out[TARGET] = out[TARGET].astype(int)
    out["Outcome"] = out[TARGET].map(LABELS)
    out["Days_Over_SLA"] = out["Processing_Days"] - out["SLA_Days"]
    return out


def breach_rate_table(model_df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = (model_df.groupby(col, observed=True)[TARGET]
         .agg(Total_Requests="size", Breaches="sum", Breach_Rate="mean")
         .reset_index())
    g["Breach_Rate_Percent"] = (g["Breach_Rate"] * 100).round(2)
    return g


# ---- chart builders (each returns a Plotly figure) ------------------------
def chart_volume_by_service(df: pd.DataFrame):
    counts = df["Service_Type"].value_counts().rename_axis("Service_Type").reset_index(name="Requests")
    fig = px.bar(counts.sort_values("Requests"), x="Requests", y="Service_Type",
                 orientation="h", text="Requests",
                 title="Completed requests by service type")
    fig.update_traces(marker_color="steelblue")
    return fig


def chart_breach_by_service(df: pd.DataFrame):
    g = breach_rate_table(df, "Service_Type").sort_values("Breach_Rate_Percent")
    fig = px.bar(g, x="Breach_Rate_Percent", y="Service_Type", orientation="h",
                 text="Breach_Rate_Percent", hover_data=["Total_Requests", "Breaches"],
                 title="SLA breach rate by service type (%)")
    fig.update_traces(marker_color="indianred")
    return fig


def chart_breach_by_queue(df: pd.DataFrame):
    g = breach_rate_table(df, "Queue_Category")
    fig = px.bar(g, x="Queue_Category", y="Breach_Rate_Percent",
                 text="Breach_Rate_Percent", hover_data=["Total_Requests", "Breaches"],
                 category_orders={"Queue_Category": QUEUE_ORDER},
                 title="SLA breach rate by queue category (%)")
    fig.update_traces(marker_color="darkorange")
    return fig


def chart_breach_by_weekday(df: pd.DataFrame):
    g = breach_rate_table(df, "Submission_Weekday")
    fig = px.bar(g, x="Submission_Weekday", y="Breach_Rate_Percent",
                 text="Breach_Rate_Percent", hover_data=["Total_Requests"],
                 category_orders={"Submission_Weekday": WEEKDAY_ORDER},
                 title="SLA breach rate by submission weekday (%)")
    return fig


def chart_days_over_sla(df: pd.DataFrame):
    return px.histogram(df, x="Days_Over_SLA", color="Outcome", barmode="overlay",
                        opacity=0.7, nbins=20,
                        color_discrete_map={"Breach": "indianred", "No breach": "seagreen"},
                        title="Processing days minus SLA days (positive = breach)")


def chart_queue_distribution(df: pd.DataFrame):
    return px.histogram(df, x="Queue_Length", color="Outcome", barmode="overlay",
                        opacity=0.7, nbins=20,
                        color_discrete_map={"Breach": "indianred", "No breach": "seagreen"},
                        title="Queue length distribution by outcome")


def chart_class_distribution(df: pd.DataFrame):
    counts = df["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Requests")
    return px.pie(counts, names="Outcome", values="Requests", hole=0.4,
                  color="Outcome",
                  color_discrete_map={"Breach": "indianred", "No breach": "seagreen"},
                  title="Target class distribution")


def _filters(model_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)
    services = sorted(model_df["Service_Type"].dropna().unique())
    offices = sorted(model_df["Assigned_Office"].dropna().unique())
    priorities = sorted(model_df["Priority"].dropna().unique())
    sel_s = c1.multiselect("Service type", services, default=services)
    sel_o = c2.multiselect("Assigned office", offices, default=offices)
    sel_p = c3.multiselect("Priority", priorities, default=priorities)
    mask = (model_df["Service_Type"].isin(sel_s)
            & model_df["Assigned_Office"].isin(sel_o)
            & model_df["Priority"].isin(sel_p))
    return model_df.loc[mask]


def render_eda_page() -> None:
    st.header("Exploratory Data Analysis")
    data = require_dataset()
    if data is None:
        return

    clean, model_df = data["clean"], data["model_df"]
    if model_df.empty:
        st.warning("The dataset has no completed requests with SLA information, "
                   "so the target cannot be analyzed.")
        return

    st.subheader("Descriptive statistics")
    t1, t2 = st.tabs(["Numeric columns", "Categorical columns"])
    with t1:
        st.dataframe(clean[NUMERIC_COLUMNS].describe().T.round(2))
    with t2:
        cat_cols = ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
                    "Assigned_Office", "Required_Documents_Complete", "Status"]
        st.dataframe(clean[cat_cols].astype("object").describe().T)

    filtered = _filters(model_df)
    if filtered.empty:
        st.warning("No requests match the selected filters.")
        return
    df = _with_label(filtered)
    st.caption(f"{len(df)} completed requests match the filters.")

    st.subheader("Class distribution (SLA_Breached)")
    c1, c2 = st.columns([1, 2])
    counts = df["Outcome"].value_counts()
    c1.dataframe(pd.DataFrame({"Count": counts,
                               "Percent": (counts / counts.sum() * 100).round(2)}))
    c2.plotly_chart(chart_class_distribution(df))

    st.subheader("Interactive charts")
    tabs = st.tabs(["Volume by service", "Breach by service", "Breach by queue",
                    "Breach by weekday", "Days over SLA", "Queue length"])
    charts = [chart_volume_by_service, chart_breach_by_service, chart_breach_by_queue,
              chart_breach_by_weekday, chart_days_over_sla, chart_queue_distribution]
    for tab, builder in zip(tabs, charts):
        with tab:
            st.plotly_chart(builder(df))

    st.subheader("Key findings")
    by_service = breach_rate_table(df, "Service_Type").sort_values("Breach_Rate", ascending=False)
    by_queue = breach_rate_table(df, "Queue_Category").sort_values("Breach_Rate", ascending=False)
    top_volume = df["Service_Type"].value_counts()
    st.write(f"- **{top_volume.index[0]}** has the most completed requests "
             f"({int(top_volume.iloc[0])}).")
    st.write(f"- **{by_service.iloc[0]['Service_Type']}** has the highest breach rate "
             f"({by_service.iloc[0]['Breach_Rate_Percent']:.2f}%).")
    st.write(f"- The **{by_queue.iloc[0]['Queue_Category']}** queue category has the highest "
             f"breach rate ({by_queue.iloc[0]['Breach_Rate_Percent']:.2f}%).")
