import os
import json
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------
# Config
# -----------------------------------
st.set_page_config(
    page_title="USC Builds Termination Risk Simulator",
    layout="centered"
)

ARTIFACT_DIR = "artifacts"
MODEL_PATH  = os.path.join(ARTIFACT_DIR, "termination_model.joblib")
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, "feature_schema.json")

# -----------------------------------
# Load model + schema
# -----------------------------------
@st.cache_resource
def load_model_and_schema():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}. Train/save it first.")
        st.stop()
    if not os.path.exists(SCHEMA_PATH):
        st.error(f"Schema not found: {SCHEMA_PATH}. Save feature_schema.json first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    with open(SCEMA_PATH, "r") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_model_and_schema()
ALL_COLS = schema["all_cols"]

# -----------------------------------
# Header
# -----------------------------------
st.title("USC Builds Termination Risk What-If Simulator")
st.write(
    """
Adjust the inputs below to represent an employee’s **last 180-day behavior**.
Then click **Predict Risk** to estimate whether the employee is likely to stay or leave.
"""
)

st.info(
    "These inputs are the strongest predictors: attendance drop, fewer weekly hours, more zero-hour weeks, and long work gaps."
)

# -----------------------------------
# Main Inputs
# -----------------------------------
st.subheader("1) Attendance Behavior (Last 180 Days)")

attendance_ratio = st.slider(
    "Attendance Ratio (hours worked ÷ expected hours)",
    min_value=0.0, max_value=1.2, value=0.80, step=0.01,
)

avg_weekly_hours = st.slider(
    "Average Weekly Hours Worked",
    min_value=0.0, max_value=60.0, value=35.0, step=0.5,
)

zero_weeks = st.slider(
    "Weeks with 0 Recorded Hours",
    min_value=0, max_value=26, value=2, step=1,
)

gap_days = st.slider(
    "Days Since Last Worked",
    min_value=0, max_value=180, value=15, step=1,
)

st.subheader("2) Formal Absence")

nonpaid_abs = st.slider(
    "Formal Absence Hours",
    min_value=0.0, max_value=80.0, value=5.0, step=0.5,
)

# -----------------------------------
# Build feature row (role columns defaulted)
# -----------------------------------
row = {c: 0 for c in ALL_COLS}

row.update({
    "attendance_ratio_lkbk": attendance_ratio,
    "avg_weekly_hours_lkbk": avg_weekly_hours,
    "zero_weeks_lkbk": zero_weeks,
    "gap_days_since_work": gap_days,
    "nonpaid_abs_hours_lkbk": nonpaid_abs,
})

# If model expects Trade/Dept columns, assign safe values
if "Trade" in row:
    row["Trade"] = "UNKNOWN"
if "Dept" in row:
    row["Dept"] = "UNKNOWN"

X_one = pd.DataFrame([row])

# -----------------------------------
# Prediction Output
# -----------------------------------
st.markdown("---")
if st.button("Predict Risk", type="primary"):
    risk = model.predict_proba(X_one)[0, 1]
    prediction = "Likely to Leave" if risk >= 0.50 else "Likely to Stay"

    st.metric("Estimated Termination Risk", f"{risk:.2%}")

    if prediction == "Likely to Leave":
        st.error(f"Prediction: **{prediction}**")
    else:
        st.success(f"Prediction: **{prediction}**")

with st.expander("What do these inputs mean?"):
    st.write(
        """
- **Attendance Ratio:** % of expected hours worked  
- **Avg Weekly Hours:** workload consistency  
- **Zero-Hour Weeks:** repeated no-shows  
- **Gap Days:** time since last activity  
- **Absence Hours:** formal unpaid leave  
"""
    )
