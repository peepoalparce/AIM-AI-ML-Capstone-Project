import os
import pandas as pd


def setup_directories():
    """Ensures models and reports directories exist."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def load_dataset(path):
    """
    Loads the credit card fraud dataset and prints basic structural summary.
    Verifies column data types, missing values, duplicates, and class distribution.
    """
    data = pd.read_csv(path)

    print("\n=== DATASET OVERVIEW ===")
    print("Head:")
    print(data.head())
    print("Shape:", data.shape)
    print("\nMissing values per column:")
    print(data.isnull().sum())
    print("\nNumber of duplicated rows:", data.duplicated().sum())
    print("\nClass distribution:")
    print(data["Class"].value_counts())
    print("Fraud percentage:", data["Class"].mean() * 100)
    print("\nData types:")
    print(data.dtypes)

    return data


def create_data_dictionary():
    """
    Generates and exports a data dictionary mapping feature names to their types and units.
    Useful for documenting anonymized PCA features alongside raw features like Time and Amount.
    """
    data_dict = pd.DataFrame(
        [
            ["Time", "float", "Seconds since first transaction", "seconds", ">= 0"],
            ["Amount", "float", "Transaction amount", "currency units", ">= 0"],
        ]
        + [
            [f"V{i}", "float", "PCA-transformed anonymized feature", "N/A", "continuous"]
            for i in range(1, 29)
        ]
        + [
            ["Class", "int", "Target variable: 0 = non-fraud, 1 = fraud", "N/A", "{0,1}"]
        ],
        columns=["Variable", "Type", "Description", "Units", "Allowed Values"]
    )
    print("\n=== DATA DICTIONARY ===")
    print(data_dict)
    data_dict.to_csv("reports/data_dictionary.csv", index=False)


def check_amount_outliers(data):
    """
    Identifies transaction amount outliers using the Interquartile Range (IQR) method.
    Helps quantify severe skewness in financial values prior to scaling.
    """
    Q1 = data["Amount"].quantile(0.25)
    Q3 = data["Amount"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_amount = ((data["Amount"] < lower_bound) | (data["Amount"] > upper_bound)).sum()
    print("\nOutliers in Amount (IQR rule):", outliers_amount)