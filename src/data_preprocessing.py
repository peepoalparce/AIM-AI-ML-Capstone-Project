import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_dataset(path):
    """
    Load the credit card fraud dataset from the given path.
    Returns a pandas DataFrame.
    """
    print(f"Loading dataset from: {path}")
    data = pd.read_csv(path)
    print("Dataset loaded successfully.")
    print("Shape:", data.shape)
    return data


def create_data_dictionary():
    """
    Create a simple data dictionary CSV file describing each column.
    Saves the file under /reports/data_dictionary.csv.
    """
    dictionary = {
        "Feature": [
            "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
            "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
            "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
            "V28", "Amount", "Class"
        ],
        "Description": [
            "Seconds elapsed between transaction and first transaction",
            "PCA Component 1", "PCA Component 2", "PCA Component 3",
            "PCA Component 4", "PCA Component 5", "PCA Component 6",
            "PCA Component 7", "PCA Component 8", "PCA Component 9",
            "PCA Component 10", "PCA Component 11", "PCA Component 12",
            "PCA Component 13", "PCA Component 14", "PCA Component 15",
            "PCA Component 16", "PCA Component 17", "PCA Component 18",
            "PCA Component 19", "PCA Component 20", "PCA Component 21",
            "PCA Component 22", "PCA Component 23", "PCA Component 24",
            "PCA Component 25", "PCA Component 26", "PCA Component 27",
            "PCA Component 28", "Transaction Amount", "Fraud Label"
        ]
    }

    df_dict = pd.DataFrame(dictionary)
    df_dict.to_csv("reports/data_dictionary.csv", index=False)
    print("Data dictionary saved to reports/data_dictionary.csv")


def check_amount_outliers(data):
    """
    Print basic statistics for the Amount feature.
    Helps identify extreme values and understand distribution.
    """
    print("\n=== Amount Feature Summary ===")
    print(data["Amount"].describe())


def basic_plots(data):
    """
    Generate basic EDA plots:
    - Class distribution
    - Amount distribution
    - Correlation heatmap
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=data["Class"])
    plt.title("Class Distribution")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(data["Amount"], bins=50, kde=True)
    plt.title("Amount Distribution")
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def split_data(data):
    """
    Split the dataset into train and test sets.
    Returns X_train, X_test, y_train, y_test.
    """
    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nData split completed.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


def scale_time_amount(X_train, X_test):
    """
    Scale the Time and Amount features using StandardScaler.
    PCA components (V1–V28) are already scaled.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    for col in ["Time", "Amount"]:
        X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
        X_test_scaled[col] = scaler.transform(X_test[[col]])

    print("\nTime and Amount scaled.")
    return X_train_scaled, X_test_scaled


def apply_smote(X_train_scaled, y_train):
    """
    Apply SMOTE to balance the dataset.
    Returns oversampled X_train_res and y_train_res.
    """
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    print("\nSMOTE applied.")
    print("New class distribution:")
    print(pd.Series(y_train_res).value_counts())

    return X_train_res, y_train_res
