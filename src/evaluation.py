import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    recall_score
)


def evaluate_model(name, y_true, y_pred, y_proba):
    """
    Evaluates model performance using Precision, Recall, F1-score, Confusion Matrix,
    ROC-AUC, and Precision-Recall AUC (PR-AUC) tailored for imbalanced data.
    """
    print(f"\n=== {name} Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    roc = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    print(f"{name} ROC-AUC:", roc)
    print(f"{name} PR-AUC:", pr_auc)

    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    return roc, pr_auc


def threshold_tuning(y_proba_rf, y_test):
    """
    Evaluates Precision and Recall across classification decision thresholds (0.1 to 0.9).
    Helps tune decision rules based on risk appetite and operational costs of false positives vs negatives.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        y_pred_t = (y_proba_rf >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Threshold {t:.2f} | Precision {precision_t:.4f} | Recall {recall_t:.4f}")


def recall_by_amount_group(X_test_scaled, y_test, y_pred_rf):
    """
    Evaluates model sensitivity (Recall) segmented by transaction size (above vs. below median Amount).
    """
    median_amt = X_test_scaled["Amount"].median()
    high_group = X_test_scaled["Amount"] > median_amt

    recall_high = recall_score(y_test[high_group], y_pred_rf[high_group])
    recall_low = recall_score(y_test[~high_group], y_pred_rf[~high_group])

    print("\n=== Recall by Amount Group ===")
    print("High Amount Recall:", recall_high)
    print("Low Amount Recall:", recall_low)