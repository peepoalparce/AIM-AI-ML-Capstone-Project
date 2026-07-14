import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

def evaluate_model(name, y_true, y_pred, y_proba):
    """
    Evaluate a model using:
        - Classification report
        - Confusion matrix
        - ROC-AUC
        - PR-AUC
        - Precision-Recall curve
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

    print(f"{name} ROC-AUC: {roc}")
    print(f"{name} PR-AUC: {pr_auc}")

    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    return roc, pr_auc
