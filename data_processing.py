"""Data loading, schema validation, cleaning and feature engineering.

Reuses the cleaning and feature-engineering steps from the midterm notebook.
The pure functions work without Streamlit (so train_model.py can import them);
the page renderer at the bottom needs Streamlit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # Streamlit is optional for the pure functions
    import streamlit as st

    _cache = st.cache_data(show_spinner=False)
except ImportError:  # e.g. when running train_model.py without Streamlit
    st = None

    def _cache(func):
        return func


BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_requests.csv"

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score",
]

CATEGORY_COLUMNS = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Status", "Reopened",
]
NUMERIC_COLUMNS = ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]

# Predictors that are known at submission time only (no leakage).
# Processing_Days, Status, Reopened and Satisfaction_Score are NOT used
# because they are only known after the request is handled.
MODEL_CATEGORICAL = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Required_Documents_Complete", "Submission_Weekday",
]
MODEL_NUMERICAL = ["Queue_Length", "SLA_Days"]
MODEL_FEATURES = MODEL_CATEGORICAL + MODEL_NUMERICAL
TARGET = "SLA_Breached"

CLOSED_STATUSES = {"Completed", "Cancelled"}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def read_csv_safely(file) -> tuple[pd.DataFrame | None, str | None]:
    """Read a CSV and return (dataframe, error_message)."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return None, "The file is empty. Please upload a CSV that contains a header row and data."
    except UnicodeDecodeError:
        return None, "The file could not be decoded. Please save it as a UTF-8 CSV."
    except pd.errors.ParserError as exc:
        return None, f"The file is not a valid CSV: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"The file could not be read: {exc}"

    if df.empty:
        return None, "The CSV has column headers but no data rows."
    return df, None


def validate_schema(df: pd.DataFrame) -> dict:
    """Compare the uploaded columns with the required schema."""
    cols = [str(c).strip() for c in df.columns]
    return {
        "missing": [c for c in REQUIRED_COLUMNS if c not in cols],
        "extra": [c for c in cols if c not in REQUIRED_COLUMNS],
    }


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Null summary per column."""
    out = pd.DataFrame({
        "Data_Type": df.dtypes.astype(str),
        "Missing_Count": df.isnull().sum(),
        "Missing_Percent": (df.isnull().mean() * 100).round(2),
        "Unique_Values": df.nunique(dropna=True),
    })
    return out


def duplicate_summary(df: pd.DataFrame) -> dict:
    id_col = "Request_ID" if "Request_ID" in df.columns else None
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_col].astype("string").str.strip().duplicated().sum())
        if id_col else 0,
    }


# ---------------------------------------------------------------------------
# Cleaning and feature engineering (from the midterm notebook)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame):
    """Return (clean_df, model_df, quality_log)."""
    clean = df.copy()
    log = [f"Rows before cleaning: {len(clean)}"]

    clean.columns = clean.columns.str.strip()

    clean["Request_ID"] = clean["Request_ID"].astype("string").str.strip()
    dup = int(clean["Request_ID"].duplicated(keep="first").sum())
    log.append(f"Duplicate Request_ID rows removed: {dup}")
    clean = clean.loc[~clean["Request_ID"].duplicated(keep="first")].copy()

    for col in CATEGORY_COLUMNS:
        if col in clean.columns:
            old = clean[col].astype("string")
            clean[col] = (
                old.str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                .str.title()
            )
            changed = int((old.fillna("") != clean[col].fillna("")).sum())
            log.append(f"{col}: cells changed during text standardization = {changed}")

    clean["Submission_Date"] = pd.to_datetime(clean["Submission_Date"], errors="coerce")
    log.append(f"Invalid or missing submission dates: {int(clean['Submission_Date'].isna().sum())}")

    for col in NUMERIC_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        neg = int((clean[col] < 0).sum())
        if neg > 0:
            clean.loc[clean[col] < 0, col] = np.nan
        log.append(f"{col}: negative values changed to missing = {neg}")

    bad_sat = clean["Satisfaction_Score"].notna() & ~clean["Satisfaction_Score"].between(1, 5)
    log.append(f"Invalid satisfaction ratings: {int(bad_sat.sum())}")
    clean.loc[bad_sat, "Satisfaction_Score"] = np.nan

    log.append(f"Rows after cleaning: {len(clean)}")

    # Feature engineering
    clean["Submission_Weekday"] = clean["Submission_Date"].dt.day_name()

    clean["Queue_Category"] = pd.Series(pd.NA, index=clean.index, dtype="string")
    clean.loc[clean["Queue_Length"].between(0, 5), "Queue_Category"] = "Low"
    clean.loc[clean["Queue_Length"].between(6, 15), "Queue_Category"] = "Medium"
    clean.loc[clean["Queue_Length"] > 15, "Queue_Category"] = "High"

    eligible = (
        clean["Status"].eq("Completed")
        & clean["Processing_Days"].notna()
        & clean["SLA_Days"].notna()
    )
    clean[TARGET] = pd.Series(pd.NA, index=clean.index, dtype="Int64")
    clean.loc[eligible, TARGET] = (
        clean.loc[eligible, "Processing_Days"] > clean.loc[eligible, "SLA_Days"]
    ).astype(int)

    model_df = clean.loc[eligible & clean[TARGET].notna()].copy()
    return clean, model_df, log


clean_data_cached = _cache(clean_data)


def queue_category(queue_length: float) -> str:
    """Same Low / Medium / High rule used in the midterm."""
    if queue_length <= 5:
        return "Low"
    if queue_length <= 15:
        return "Medium"
    return "High"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact feature frame the saved pipeline was trained on.

    Categorical columns become plain objects (NaN for missing) and numeric
    columns become floats, so training and inference see identical types.
    """
    X = pd.DataFrame(index=df.index)
    for col in MODEL_CATEGORICAL:
        s = df[col].astype("object")
        X[col] = s.where(s.notna(), np.nan)
    for col in MODEL_NUMERICAL:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return X[MODEL_FEATURES]


# ---------------------------------------------------------------------------
# Session helpers (shared by all pages)
# ---------------------------------------------------------------------------
def set_dataset(df: pd.DataFrame, name: str) -> None:
    clean, model_df, log = clean_data_cached(df)
    st.session_state["dataset"] = {
        "name": name,
        "raw": df,
        "clean": clean,
        "model_df": model_df,
        "log": log,
    }


def get_dataset() -> dict | None:
    return st.session_state.get("dataset")


def require_dataset() -> dict | None:
    """Return the loaded dataset or show a friendly message."""
    data = get_dataset()
    if data is None:
        st.info("No dataset loaded yet. Go to the **Dataset** page and upload a CSV "
                "or load the sample data.")
    return data


# ---------------------------------------------------------------------------
# Dataset page
# ---------------------------------------------------------------------------
def render_dataset_page() -> None:
    st.header("Dataset Upload and Validation")
    st.caption("Upload the student service request CSV. Use synthetic or properly "
               "authorized data only - never real student records on a public app.")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    col_a, col_b = st.columns([1, 3])
    use_sample = col_a.button("Load sample data")
    col_b.caption("The sample file is synthetic and follows the same schema.")

    if uploaded is not None:
        df, error = read_csv_safely(uploaded)
        if error:
            st.error(error)
        else:
            problems = validate_schema(df)
            if problems["missing"]:
                st.error("Missing required columns: " + ", ".join(problems["missing"]))
                st.write("Required schema:", ", ".join(REQUIRED_COLUMNS))
            else:
                if problems["extra"]:
                    st.warning("Extra columns will be ignored: " + ", ".join(problems["extra"]))
                if st.session_state.get("dataset", {}).get("name") != uploaded.name:
                    set_dataset(df, uploaded.name)
    elif use_sample:
        if not SAMPLE_DATA_PATH.exists():
            st.error("The sample file data/sample_requests.csv was not found.")
        else:
            df, error = read_csv_safely(SAMPLE_DATA_PATH)
            if error:
                st.error(error)
            else:
                set_dataset(df, "sample_requests.csv (synthetic)")

    data = get_dataset()
    if data is None:
        with st.expander("Required columns"):
            st.write(", ".join(REQUIRED_COLUMNS))
        return

    raw, clean, model_df = data["raw"], data["clean"], data["model_df"]
    st.success(f"Dataset loaded: **{data['name']}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw):,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Eligible completed requests", f"{len(model_df):,}")
    dups = duplicate_summary(raw)
    c4.metric("Duplicate Request_IDs", dups["duplicate_ids"])

    st.subheader("Data preview")
    st.dataframe(raw.head(20))

    left, right = st.columns(2)
    with left:
        st.subheader("Null summary")
        st.dataframe(quality_summary(raw))
    with right:
        st.subheader("Duplicate summary")
        st.write(f"Fully duplicated rows: **{dups['duplicate_rows']}**")
        st.write(f"Duplicate Request_ID values: **{dups['duplicate_ids']}**")
        st.subheader("Cleaning log")
        st.code("\n".join(data["log"]), language="text")

    st.download_button(
        "Download cleaned dataset",
        clean.to_csv(index=False).encode("utf-8"),
        "student_service_requests_cleaned.csv",
        "text/csv",
    )
