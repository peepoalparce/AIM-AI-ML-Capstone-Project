import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def basic_plots(data):
    """
    Renders Exploratory Data Analysis (EDA) visuals including class distributions,
    density plots of transaction amounts, and feature correlation heatmaps.
    """
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Class", data=data)
    plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
    plt.show()

    fraud = data[data["Class"] == 1]
    non_fraud = data[data["Class"] == 0]
    print("\nFraud transactions:", len(fraud))
    print("Non-fraud transactions:", len(non_fraud))
    print("Fraud percentage:", len(fraud) / len(data) * 100)

    plt.figure(figsize=(8, 5))
    sns.histplot(non_fraud["Amount"], bins=50, color="blue", label="Non-Fraud",
                 stat="density", kde=True)
    sns.histplot(fraud["Amount"], bins=50, color="red", label="Fraud",
                 stat="density", kde=True)
    plt.legend()
    plt.title("Transaction Amount Distribution by Class")
    plt.show()

    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()


def split_data(data):
    """
    Performs feature engineering (Log_Amount, Is_High_Amount) and splits data into
    80/20 train and test sets using stratified sampling to preserve fraud ratios.
    """
    df_engineered = data.copy()
    df_engineered["Log_Amount"] = np.log1p(df_engineered["Amount"])
    df_engineered["Is_High_Amount"] = (df_engineered["Amount"] >= 100).astype(int)
    X = df_engineered.drop("Class", axis=1)
    y = df_engineered["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def scale_time_amount(X_train, X_test):
    """
    Applies StandardScaler to unscaled features (Time, Amount, Log_Amount).
    Fits strictly on X_train to prevent data leakage onto X_test.
    """
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_to_scale = ["Time", "Amount", "Log_Amount"]
    for col in cols_to_scale:
        X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
        X_test_scaled[col] = scaler.transform(X_test[[col]])

    return X_train_scaled, X_test_scaled


def apply_smote(X_train_scaled, y_train):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to balance the minority class.
    Generates synthetic fraud examples strictly within the training set.
    """
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print("\nOriginal training set class distribution:")
    print(y_train.value_counts())
    print("\nResampled training set class distribution (SMOTE):")
    print(y_train_res.value_counts())

    return X_train_res, y_train_res