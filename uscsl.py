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
    with open(SCHEMA_PATH, "r") as f:
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
    "These inputs are the **strongest predictors** from our analysis: "
    "attendance drop, fewer weekly hours, more zero-hour weeks, and long work gaps."
)

# -----------------------------------
# Main Inputs (high impact)
# -----------------------------------
st.subheader("1) Attendance Behavior (Last 180 Days)")

attendance_ratio = st.slider(
    "Attendance Ratio (hours worked ÷ expected hours)",
    min_value=0.0, max_value=1.2, value=0.80, step=0.01,
    help="1.0 = expected hours. Near 0.0 = low attendance."
)

avg_weekly_hours = st.slider(
    "Average Weekly Hours Worked",
    min_value=0.0, max_value=60.0, value=35.0, step=0.5,
    help="Typical stable full-time workers are near full-time hours."
)

zero_weeks = st.slider(
    "Weeks with 0 Recorded Hours",
    min_value=0, max_value=26, value=2, step=1,
    help="Higher values mean repeated no-show weeks."
)

gap_days = st.slider(
    "Days Since Last Worked",
    min_value=0, max_value=180, value=15, step=1,
    help="How long since this employee last clocked any hours."
)

st.subheader("2) Formal Absence")

nonpaid_abs = st.slider(
    "Formal Absence Hours",
    min_value=0.0, max_value=80.0, value=5.0, step=0.5,
    help="Unpaid leave time (rare among leavers)."
)

st.subheader("3) Role Information")

trade = st.selectbox(
    "Trade",
    ["CARP", "DW", "FIN", "INSUL", "PM", "FRM", "OTHER"],
    help="Select the employee’s trade category."
)

# -----------------------------------
# Build feature row
# -----------------------------------
row = {c: 0 for c in ALL_COLS}

row.update({
    "attendance_ratio_lkbk": attendance_ratio,
    "avg_weekly_hours_lkbk": avg_weekly_hours,
    "zero_weeks_lkbk": zero_weeks,
    "gap_days_since_work": gap_days,
    "nonpaid_abs_hours_lkbk": nonpaid_abs,
    "Trade": trade
})

# If model expects Dept column, fill it safely with a placeholder:
if "Dept" in row:
    row["Dept"] = "N/A"

X_one = pd.DataFrame([row])

# -----------------------------------
# Prediction Output
# -----------------------------------
st.markdown("---")
if st.button("Predict Risk", type="primary"):
    risk = model.predict_proba(X_one)[0, 1]
    predicted_leave = int(risk >= 0.50)

    st.metric("Estimated Termination Risk", f"{risk:.2%}")

    if predicted_leave == 1:
        st.error("Prediction: **Likely to Leave**")
        st.write(
            "This pattern resembles employees showing disengagement before exiting."
        )
    else:
        st.success("Prediction: **Likely to Stay**")
        st.write("Attendance and engagement patterns align with retained employees.")

    st.caption("Model based on 180-day employee behavioral patterns.")

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
