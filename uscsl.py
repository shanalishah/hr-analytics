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
MODEL_PATH = os.path.join(ARTIFACT_DIR, "termination_model.joblib")
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

# Choose a typical default role context for the model
DEFAULT_TRADE = "CARP"   # e.g., Carpenter
DEFAULT_DEPT  = "FIELD"  # Field staff (majority group)

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
    "This tool focuses on the **strongest behavioral predictors** we found: "
    "attendance ratio, weekly hours, zero-hour weeks, recent work gaps, "
    "and unpaid absence."
)

# -----------------------------------
# Optional: Quick Profiles for Demo
# -----------------------------------
st.subheader("0) Quick Profiles (Optional)")

profile_options = {
    "Custom": {
        "attendance_ratio": 0.90,
        "avg_weekly_hours": 40.0,
        "zero_weeks": 0,
        "gap_days": 0,
        "nonpaid_abs": 0.0,
    },
    "Stable Full-Time Employee": {
        "attendance_ratio": 0.98,
        "avg_weekly_hours": 40.0,
        "zero_weeks": 0,
        "gap_days": 0,
        "nonpaid_abs": 4.0,
    },
    "Emerging Risk (Inconsistent Attendance)": {
        "attendance_ratio": 0.75,
        "avg_weekly_hours": 32.0,
        "zero_weeks": 2,
        "gap_days": 7,
        "nonpaid_abs": 2.0,
    },
    "High Risk (Disengaged Pattern)": {
        "attendance_ratio": 0.55,
        "avg_weekly_hours": 25.0,
        "zero_weeks": 4,
        "gap_days": 21,
        "nonpaid_abs": 0.0,
    },
}

selected_profile = st.selectbox(
    "Choose a sample profile or keep 'Custom' to set values manually:",
    list(profile_options.keys()),
)

defaults = profile_options[selected_profile]

# -----------------------------------
# Main Inputs (behavior only, with guardrails)
# -----------------------------------
st.subheader("1) Attendance Behavior (Last 180 Days)")

attendance_ratio = st.slider(
    "Attendance Ratio (hours worked ÷ expected hours)",
    min_value=0.30,
    max_value=1.00,
    value=defaults["attendance_ratio"],
    step=0.01,
    help="1.0 = fully meeting expected hours. Values above 1.0 are not allowed."
)

avg_weekly_hours = st.slider(
    "Average Weekly Hours Worked",
    min_value=20.0,
    max_value=50.0,
    value=defaults["avg_weekly_hours"],
    step=1.0,
    help="Typical full-time employees fall between 35–50 hours per week."
)

zero_weeks = st.slider(
    "Weeks with 0 Recorded Hours",
    min_value=0,
    max_value=6,
    value=defaults["zero_weeks"],
    step=1,
    help="Higher values indicate repeated no-show weeks within the last 180 days."
)

gap_days = st.slider(
    "Days Since Last Worked",
    min_value=0,
    max_value=45,
    value=defaults["gap_days"],
    step=1,
    help="Gaps over ~45 days often indicate an exit or extended leave."
)

st.subheader("2) Formal Absence")

nonpaid_abs = st.slider(
    "Formal Absence Hours (Unpaid)",
    min_value=0.0,
    max_value=40.0,
    value=defaults["nonpaid_abs"],
    step=1.0,
    help="Hours of formal unpaid leave. Values above 40 are rare and often disciplinary."
)

# -----------------------------------
# Soft Outlier Warnings (explanatory only)
# -----------------------------------
outlier_msgs = []

if attendance_ratio < 0.60:
    outlier_msgs.append("Very low attendance ratio compared to expected hours.")
if zero_weeks >= 3:
    outlier_msgs.append("Multiple zero-hour weeks, which often precede termination.")
if gap_days > 14:
    outlier_msgs.append("Extended gap since last worked; may indicate disengagement.")
if nonpaid_abs > 20:
    outlier_msgs.append("High unpaid absence, which may reflect instability or issues.")
if avg_weekly_hours < 30:
    outlier_msgs.append("Low average weekly hours for a full-time role.")
if avg_weekly_hours > 48:
    outlier_msgs.append("Very high weekly hours, which may be associated with burnout.")

if outlier_msgs:
    st.warning("⚠️ **Risk pattern signals detected based on behavior:**\n\n- " +
               "\n- ".join(outlier_msgs))

# -----------------------------------
# Build a single-row feature vector
# -----------------------------------
row = {c: 0 for c in ALL_COLS}

row.update({
    "attendance_ratio_lkbk": attendance_ratio,
    "avg_weekly_hours_lkbk": avg_weekly_hours,
    "zero_weeks_lkbk": zero_weeks,
    "gap_days_since_work": gap_days,
    "nonpaid_abs_hours_lkbk": nonpaid_abs,
    # Hidden but important role context for the model
    "Trade": DEFAULT_TRADE,
    "Dept": DEFAULT_DEPT,
})

X_one = pd.DataFrame([row])

# -----------------------------------
# Helper: Confidence Label & Driver Explanation
# -----------------------------------
def get_confidence_label(prob: float) -> str:
    """
    Simple heuristic based on distance from 0.5.
    """
    if 0.45 <= prob <= 0.55:
        return "Low"
    elif 0.35 <= prob <= 0.65:
        return "Medium"
    else:
        return "High"


def get_driver_messages(
    attendance_ratio: float,
    avg_weekly_hours: float,
    zero_weeks: int,
    gap_days: int,
    nonpaid_abs: float,
) -> list[str]:
    """
    Rule-based explanation of key behavioral drivers.
    Not model coefficients, but aligned with the patterns in the data.
    """
    drivers = []

    # Attendance ratio
    if attendance_ratio < 0.70:
        drivers.append("Low attendance ratio compared to expected hours.")
    elif attendance_ratio > 0.95:
        drivers.append("Consistently meeting expected hours (strong attendance).")

    # Weekly hours
    if avg_weekly_hours < 32:
        drivers.append("Working substantially fewer hours than a typical full-time worker.")
    elif 35 <= avg_weekly_hours <= 45:
        drivers.append("Weekly hours are in a healthy full-time range.")

    # Zero-hour weeks
    if zero_weeks == 0:
        drivers.append("No zero-hour weeks recorded, indicating steady engagement.")
    elif zero_weeks >= 3:
        drivers.append("Several zero-hour weeks, which often precede termination.")
    else:
        drivers.append("Some gaps (zero-hour weeks), but not extreme.")

    # Gap days
    if gap_days == 0:
        drivers.append("No current gap since last worked.")
    elif gap_days > 21:
        drivers.append("Long gap since last worked, a strong signal of disengagement.")

    # Unpaid absence
    if nonpaid_abs == 0:
        drivers.append("No unpaid absence recorded.")
    elif nonpaid_abs > 20:
        drivers.append("High unpaid absence hours, which may reflect instability or issues.")

    return drivers

# -----------------------------------
# Predict button + output
# -----------------------------------
st.markdown("---")
if st.button("Predict Risk", type="primary"):
    risk = model.predict_proba(X_one)[0, 1]
    predicted_leave = int(risk >= 0.50)
    confidence = get_confidence_label(risk)
    drivers = get_driver_messages(
        attendance_ratio,
        avg_weekly_hours,
        zero_weeks,
        gap_days,
        nonpaid_abs,
    )

    st.metric("Estimated Termination Risk", f"{risk:.2%}")
    st.write(f"**Model Confidence:** {confidence}")

    if predicted_leave == 1:
        st.error("Prediction: **Likely to Leave / Terminate**")
        st.write(
            "This behavioral pattern is similar to employees who eventually left "
            "USC Builds in the historical data."
        )
    else:
        st.success("Prediction: **Likely to Stay**")
        st.write(
            "This behavioral pattern is similar to stable employees who remained "
            "with USC Builds in the historical data."
        )

    st.subheader("Key Behavioral Drivers Behind This Estimate")
    for msg in drivers:
        st.markdown(f"- {msg}")

    st.caption(
        "Note: This simulator is based on historical patterns and should be used as a "
        "decision support tool, not a standalone decision-maker."
    )

# -----------------------------------
# Help / Definitions
# -----------------------------------
with st.expander("What do these inputs mean?"):
    st.write(
        """
- **Attendance Ratio:** Hours actually worked ÷ hours expected in the last 180 days.  
- **Avg Weekly Hours:** Average hours worked per week in the last 6 months.  
- **Zero-Hour Weeks:** Number of weeks with *no* recorded hours.  
- **Days Since Last Worked:** How long it has been since the employee last worked any hours.  
- **Unpaid Absence Hours:** Formal unpaid leave recorded in HR/TimeOff systems.  

These variables were selected from USC Builds data as the **most predictive early-warning signals**
for potential turnover.
"""
    )
