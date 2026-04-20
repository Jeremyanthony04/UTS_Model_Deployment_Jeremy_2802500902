
import os
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="UTS Model Deployment", layout="wide")

# =========================
# Load saved models
# =========================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"

classification_model_path = ARTIFACTS_DIR / "classification_pipeline.pkl"
regression_model_path = ARTIFACTS_DIR / "regression_pipeline.pkl"

classification_model = joblib.load(classification_model_path)
regression_model = joblib.load(regression_model_path)

# =========================
# Header
# =========================
st.title("UTS Model Deployment - Monolithic Streamlit App")
st.write("Monolithic deployment using Streamlit with classification and regression prediction.")

# =========================
# Sidebar
# =========================
st.sidebar.header("Navigation")
menu = st.sidebar.radio(
    "Choose Prediction Task",
    ["Classification", "Regression"]
)
st.sidebar.success("Models loaded successfully.")

# =========================
# Classification Section
# =========================
if menu == "Classification":
    st.subheader("Classification Prediction - Placement Status")
    st.write("Fill in the form to predict whether a student will be placed or not.")

    left_col, right_col = st.columns([2, 1])

    with left_col:
        with st.form("classification_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
            hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=70.0)
            degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
            entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=70.0)
            technical_skill_score = st.number_input("Technical Skill Score", min_value=0.0, max_value=100.0, value=70.0)
            soft_skill_score = st.number_input("Soft Skill Score", min_value=0.0, max_value=100.0, value=70.0)
            internship_count = st.number_input("Internship Count", min_value=0, max_value=10, value=1)
            live_projects = st.number_input("Live Projects", min_value=0, max_value=10, value=1)
            work_experience_months = st.number_input("Work Experience (Months)", min_value=0, max_value=60, value=6)
            certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2)
            attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0)
            backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0)
            extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

            submitted_classification = st.form_submit_button("Predict Placement Status")

    with right_col:
        st.info("Prediction result and input summary will appear here after submission.")

    if submitted_classification:
        input_df = pd.DataFrame([{
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
        }])

        prediction = classification_model.predict(input_df)[0]

        left_result, right_result = st.columns([2, 1])

        with left_result:
            st.write("### Input Data")
            st.dataframe(input_df, use_container_width=True)

        with right_result:
            st.write("### Prediction Result")
            if prediction == 1:
                st.success("Placed")
                st.metric("Placement Status", "Placed")
            else:
                st.error("Not Placed")
                st.metric("Placement Status", "Not Placed")

# =========================
# Regression Section
# =========================
elif menu == "Regression":
    st.subheader("Regression Prediction - Salary Package")
    st.write("Fill in the form to predict the student's salary package.")

    left_col, right_col = st.columns([2, 1])

    with left_col:
        with st.form("regression_form"):
            gender = st.selectbox("Gender", ["Male", "Female"], key="reg_gender")
            ssc_percentage = st.number_input("SSC Percentage", min_value=0.0, max_value=100.0, value=70.0, key="reg_ssc")
            hsc_percentage = st.number_input("HSC Percentage", min_value=0.0, max_value=100.0, value=70.0, key="reg_hsc")
            degree_percentage = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=70.0, key="reg_degree")
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, key="reg_cgpa")
            entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0.0, max_value=100.0, value=70.0, key="reg_entrance")
            technical_skill_score = st.number_input("Technical Skill Score", min_value=0.0, max_value=100.0, value=70.0, key="reg_tech")
            soft_skill_score = st.number_input("Soft Skill Score", min_value=0.0, max_value=100.0, value=70.0, key="reg_soft")
            internship_count = st.number_input("Internship Count", min_value=0, max_value=10, value=1, key="reg_intern")
            live_projects = st.number_input("Live Projects", min_value=0, max_value=10, value=1, key="reg_projects")
            work_experience_months = st.number_input("Work Experience (Months)", min_value=0, max_value=60, value=6, key="reg_exp")
            certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2, key="reg_cert")
            attendance_percentage = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0, key="reg_att")
            backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0, key="reg_backlogs")
            extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"], key="reg_extra")

            submitted_regression = st.form_submit_button("Predict Salary Package")

    with right_col:
        st.info("Prediction result and input summary will appear here after submission.")

    if submitted_regression:
        input_df = pd.DataFrame([{
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
        }])

        prediction = regression_model.predict(input_df)[0]

        left_result, right_result = st.columns([2, 1])

        with left_result:
            st.write("### Input Data")
            st.dataframe(input_df, use_container_width=True)

        with right_result:
            st.write("### Prediction Result")
            st.success(f"Predicted Salary Package: {prediction:.2f} LPA")
            st.metric("Salary Package (LPA)", f"{prediction:.2f}")