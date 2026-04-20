from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.data_ingestion import load_data, get_regression_data


def build_regression_pipeline():
    """
    Build end-to-end regression pipeline:
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
            ("model", RandomForestRegressor(random_state=42))
        ]
    )

    return pipeline


def train_regression_pipeline(file_path: str):
    """
    Train regression pipeline and return trained model + split data.
    """
    df = load_data(file_path)
    X, y = get_regression_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = build_regression_pipeline()
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pipeline, X_train, X_test, y_train, y_test = train_regression_pipeline("../B.csv")

    print("=== REGRESSION PIPELINE SUCCESS ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
    print("\nPipeline object:")
    print(pipeline)