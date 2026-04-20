import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train_regression_pipeline import train_regression_pipeline


def run_regression_experiment(file_path: str):
    pipeline, X_train, X_test, y_train, y_test = train_regression_pipeline(file_path)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment("UTS_Model_Deployment_Regression")

    with mlflow.start_run(run_name="random_forest_regression_pipeline"):
        # log params
        mlflow.log_param("task", "regression")
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # log model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="regression_model"
        )

        print("=== MLFLOW REGRESSION LOG SUCCESS ===")
        print("MAE :", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R2  :", round(r2, 4))


if __name__ == "__main__":
    run_regression_experiment("../B.csv")