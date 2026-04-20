import os
import joblib

from src.train_classification_pipeline import train_classification_pipeline
from src.train_regression_pipeline import train_regression_pipeline


def save_models(file_path: str, output_dir: str = "artifacts"):
    """
    Train and save classification + regression pipelines as .pkl files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Train classification pipeline
    classification_pipeline, _, _, _, _ = train_classification_pipeline(file_path)
    classification_model_path = os.path.join(output_dir, "classification_pipeline.pkl")
    joblib.dump(classification_pipeline, classification_model_path)

    # Train regression pipeline
    regression_pipeline, _, _, _, _ = train_regression_pipeline(file_path)
    regression_model_path = os.path.join(output_dir, "regression_pipeline.pkl")
    joblib.dump(regression_pipeline, regression_model_path)

    print("=== MODEL SAVING SUCCESS ===")
    print("Classification model saved to:", classification_model_path)
    print("Regression model saved to    :", regression_model_path)


if __name__ == "__main__":
    save_models("../B.csv")