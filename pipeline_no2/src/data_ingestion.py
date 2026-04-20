import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def get_classification_data(df: pd.DataFrame):
    """
    Prepare features and target for classification task.

    Target:
    - placement_status

    Leakage prevention:
    - student_id is removed
    - salary_package_lpa is removed from features
    """
    df = df.drop(columns=["student_id"])

    X = df.drop(columns=["placement_status", "salary_package_lpa"])
    y = df["placement_status"]

    return X, y


def get_regression_data(df: pd.DataFrame):
    """
    Prepare features and target for regression task.

    Target:
    - salary_package_lpa

    Leakage prevention:
    - student_id is removed
    - placement_status is removed from features
    """
    df = df.drop(columns=["student_id"])

    X = df.drop(columns=["salary_package_lpa", "placement_status"])
    y = df["salary_package_lpa"]

    return X, y


if __name__ == "__main__":
    df = load_data("../B.csv")
    print("Dataset shape:", df.shape)

    X_class, y_class = get_classification_data(df)
    print("\nClassification data:")
    print("X_class shape:", X_class.shape)
    print("y_class shape:", y_class.shape)

    X_reg, y_reg = get_regression_data(df)
    print("\nRegression data:")
    print("X_reg shape:", X_reg.shape)
    print("y_reg shape:", y_reg.shape)