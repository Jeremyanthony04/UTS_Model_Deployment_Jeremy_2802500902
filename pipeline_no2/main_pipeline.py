from src.data_ingestion import load_data, get_classification_data, get_regression_data


def main():
    df = load_data("../B.csv")

    print("=== DATA INGESTION SUCCESS ===")
    print("Dataset shape:", df.shape)

    X_class, y_class = get_classification_data(df)
    print("\n=== CLASSIFICATION DATA ===")
    print("X_class shape:", X_class.shape)
    print("y_class shape:", y_class.shape)

    X_reg, y_reg = get_regression_data(df)
    print("\n=== REGRESSION DATA ===")
    print("X_reg shape:", X_reg.shape)
    print("y_reg shape:", y_reg.shape)


if __name__ == "__main__":
    main()