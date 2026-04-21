from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

app = FastAPI(title="UTS Model Deployment API", version="1.0")

# =========================
# Load models safely
# =========================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"

classification_model_path = ARTIFACTS_DIR / "classification_pipeline.pkl"
regression_model_path = ARTIFACTS_DIR / "regression_pipeline.pkl"

classification_model = joblib.load(classification_model_path)
regression_model = joblib.load(regression_model_path)

# =========================
# Request schema
# =========================
class StudentInput(BaseModel):
    gender: str
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    entrance_exam_score: float
    technical_skill_score: float
    soft_skill_score: float
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: float
    backlogs: int
    extracurricular_activities: str


# =========================
# Root endpoint
# =========================
@app.get("/")
def home():
    return {"message": "UTS Model Deployment FastAPI is running"}


# =========================
# Classification endpoint
# =========================
@app.post("/predict_classification")
def predict_classification(data: StudentInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = classification_model.predict(input_df)[0]

    label = "Placed" if prediction == 1 else "Not Placed"

    return {
        "task": "classification",
        "prediction": int(prediction),
        "label": label
    }


# =========================
# Regression endpoint
# =========================
@app.post("/predict_regression")
def predict_regression(data: StudentInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = regression_model.predict(input_df)[0]

    return {
        "task": "regression",
        "predicted_salary_lpa": round(float(prediction), 2)
    }