import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="UTS Decoupled App", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"

st.title("UTS Model Deployment - Decoupled Architecture")
st.write("Frontend Streamlit sebagai client, terhubung ke backend FastAPI.")

# =========================
# Sidebar
# =========================
task = st.sidebar.radio(
    "Choose Prediction Task",
    ["Classification", "Regression"]
)

st.sidebar.info(f"FastAPI Base URL: {API_BASE_URL}")

# =========================
# Input form
# =========================
st.subheader(f"{task} Form")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
    hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
    degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
    entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=70.0)
    technical_skill_score = st.number_input("Technical Skill Score", min_value=0.0, max_value=100.0, value=75.0)
    soft_skill_score = st.number_input("Soft Skill Score", min_value=0.0, max_value=100.0, value=75.0)
    internship_count = st.number_input("Internship Count", min_value=0, value=1)
    live_projects = st.number_input("Live Projects", min_value=0, value=1)
    work_experience_months = st.number_input("Work Experience (Months)", min_value=0, value=6)
    certifications = st.number_input("Certifications", min_value=0, value=2)
    attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0)
    backlogs = st.number_input("Backlogs", min_value=0, value=0)
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    submitted = st.form_submit_button(
        "Predict Placement Status" if task == "Classification" else "Predict Salary Package"
    )

if submitted:
    payload = {
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "cgpa": cgpa,
        "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score,
        "soft_skill_score": soft_skill_score,
        "internship_count": internship_count,
        "live_projects": live_projects,
        "work_experience_months": work_experience_months,
        "certifications": certifications,
        "attendance_percentage": attendance_percentage,
        "backlogs": backlogs,
        "extracurricular_activities": extracurricular_activities
    }

    st.subheader("Input Data")
    st.dataframe(pd.DataFrame([payload]), use_container_width=True)

    try:
        if task == "Classification":
            response = requests.post(f"{API_BASE_URL}/predict_classification", json=payload)
        else:
            response = requests.post(f"{API_BASE_URL}/predict_regression", json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Prediction Result")

            if task == "Classification":
                st.success(f"Placement Status: {result['label']}")
                st.json(result)
            else:
                st.success(f"Predicted Salary Package: {result['predicted_salary_lpa']} LPA")
                st.json(result)
        else:
            st.error(f"Request failed with status code {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("Failed to connect to FastAPI backend.")
        st.exception(e)