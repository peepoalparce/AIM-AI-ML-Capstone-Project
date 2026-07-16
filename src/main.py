import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils import setup_directories, load_dataset, create_data_dictionary, check_amount_outliers
from data_preprocessing import basic_plots, split_data, scale_time_amount, apply_smote
from dimensionality_reduction import pca_tsne_plots
from feature_selection import select_features_with_rf
from model_training import train_logistic_regression, train_random_forest, train_xgboost
from evaluation import evaluate_model, threshold_tuning, recall_by_amount_group
from explainability import plot_permutation_importance, pdp_ice_plot, lime_explanations
from fairness import fairness_metrics


def main():
    setup_directories()

    data_path = r"D:\AIM\Artificial Inteligence and Machine Learning\Capstone Project\creditcard.csv"
    data = load_dataset(data_path)
    create_data_dictionary()
    check_amount_outliers(data)
    basic_plots(data)

    X_train, X_test, y_train, y_test = split_data(data)
    X_train_scaled, X_test_scaled = scale_time_amount(X_train, X_test)
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
    X_train_sel, X_test_sel, X_test_sel_df, selected_features = select_features_with_rf(
        X_train_res, y_train_res, X_test_scaled, X_train_scaled
    )

    pca_tsne_plots(data.drop("Class", axis=1), data["Class"])

    results = {}

    # Logistic Regression
    log_reg = train_logistic_regression(X_train_sel, y_train_res)
    y_proba_lr = log_reg.predict_proba(X_test_sel)[:, 1]
    y_pred_lr = (y_proba_lr >= 0.5).astype(int)
    results["Logistic Regression"] = evaluate_model(
        "Logistic Regression", y_test, y_pred_lr, y_proba_lr
    )

    # Random Forest
    rf = train_random_forest(X_train_sel, y_train_res)
    y_proba_rf = rf.predict_proba(X_test_sel)[:, 1]
    y_pred_rf = (y_proba_rf >= 0.5).astype(int)
    results["Random Forest"] = evaluate_model(
        "Random Forest", y_test, y_pred_rf, y_proba_rf
    )

    # XGBoost 
    xgb = train_xgboost(X_train_sel, y_train_res)
    if xgb is not None:
        y_proba_xgb = xgb.predict_proba(X_test_sel)[:, 1]
        y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)
        results["XGBoost"] = evaluate_model(
            "XGBoost", y_test, y_pred_xgb, y_proba_xgb
        )

    model_preds = {
        "Logistic Regression": (y_pred_lr, y_proba_lr),
        "Random Forest": (y_pred_rf, y_proba_rf),
    }
    if xgb is not None:
        model_preds["XGBoost"] = (y_pred_xgb, y_proba_xgb)

    comparison_rows = []
    perf_rows = []

    for name, (preds, probs) in model_preds.items():
        cm = confusion_matrix(y_test, preds)
        roc = results[name][0]
        pr = results[name][1]

        perf_rows.append([name, roc, pr])
        comparison_rows.append({
            "Model": name,
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "False Positives (FP)": cm[0, 1],
            "False Negatives (FN)": cm[1, 0],
            "ROC-AUC": roc,
            "PR-AUC": pr
        })

    performance_df = pd.DataFrame(perf_rows, columns=["Model", "ROC-AUC", "PR-AUC"])
    performance_df.to_csv("reports/model_performance.csv", index=False)
    print("\nModel performance saved to reports/model_performance.csv")

    print("\n=== COMPREHENSIVE MULTI-MODEL COMPARISON MATRIX ===")
    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))
    
    plot_permutation_importance(rf, X_test_sel, y_test, selected_features)
    threshold_tuning(y_proba_rf, y_test)
    pdp_ice_plot(rf, X_test_sel_df, selected_features)
    recall_by_amount_group(X_test_scaled, y_test, y_pred_rf)
    lime_explanations(X_train_sel, X_test_sel, selected_features, y_test, rf)
    fairness_metrics(y_test, y_pred_rf, X_test_scaled)


if __name__ == "__main__":
    main()