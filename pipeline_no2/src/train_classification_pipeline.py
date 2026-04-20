import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_ingestion import load_data, get_classification_data


def build_classification_pipeline():
    """
    Build end-to-end classification pipeline:
    preprocessing + model
    """
    numeric_features = [
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "cgpa",
        "entrance_exam_score",
        "technical_skill_score",
        "soft_skill_score",
        "internship_count",
        "live_projects",
        "work_experience_months",
        "certifications",
        "attendance_percentage",
        "backlogs"
    ]

    categorical_features = [
        "gender",
        "extracurricular_activities"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42))
        ]
    )

    return pipeline


def train_classification_pipeline(file_path: str):
    """
    Train classification pipeline and return trained model + split data.
    """
    df = load_data(file_path)
    X, y = get_classification_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pipeline, X_train, X_test, y_train, y_test = train_classification_pipeline("../B.csv")

    print("=== CLASSIFICATION PIPELINE SUCCESS ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
    print("\nPipeline object:")
    print(pipeline)