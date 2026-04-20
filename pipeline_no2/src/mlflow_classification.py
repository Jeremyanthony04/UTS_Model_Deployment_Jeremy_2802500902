import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.train_classification_pipeline import train_classification_pipeline


def run_classification_experiment(file_path: str):
    pipeline, X_train, X_test, y_train, y_test = train_classification_pipeline(file_path)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.set_experiment("UTS_Model_Deployment_Classification")

    with mlflow.start_run(run_name="decision_tree_classification_pipeline"):
        # log params
        mlflow.log_param("task", "classification")
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # log model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="classification_model"
        )

        print("=== MLFLOW CLASSIFICATION LOG SUCCESS ===")
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1-Score :", round(f1, 4))


if __name__ == "__main__":
    run_classification_experiment("../B.csv")