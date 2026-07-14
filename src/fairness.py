import numpy as np
from sklearn.metrics import confusion_matrix, recall_score

def recall_by_amount_group(X_test_scaled, y_test, y_pred_rf):
    """
    Compute recall for high vs low Amount groups.
    Helps detect performance differences across transaction sizes.
    """
    median_amt = X_test_scaled["Amount"].median()
    high_group = X_test_scaled["Amount"] > median_amt

    recall_high = recall_score(y_test[high_group], y_pred_rf[high_group])
    recall_low = recall_score(y_test[~high_group], y_pred_rf[~high_group])

    print("\n=== Recall by Amount Group ===")
    print("High Amount Recall:", recall_high)
    print("Low Amount Recall:", recall_low)


def fairness_metrics(y_test, y_pred_rf, X_test_scaled):
    """
    Compute fairness metrics:
        - TPR (Recall)
        - FPR
        - Positive Rate
        - Demographic Parity Ratio
        - Equalized Odds
        - Disparate Impact
    """
    median_amt = X_test_scaled["Amount"].median()
    high_group = X_test_scaled["Amount"] > median_amt

    y_true_high = y_test[high_group]
    y_pred_high = y_pred_rf[high_group]

    y_true_low = y_test[~high_group]
    y_pred_low = y_pred_rf[~high_group]

    def group_confusion(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn, fp, fn, tp

    def rates_from_confusion(tn, fp, fn, tp):
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        pos_rate = (tp + fp) / (tp + fp + tn + fn)
        return tpr, fpr, pos_rate

    tn_h, fp_h, fn_h, tp_h = group_confusion(y_true_high, y_pred_high)
    tn_l, fp_l, fn_l, tp_l = group_confusion(y_true_low, y_pred_low)

    tpr_h, fpr_h, pos_rate_h = rates_from_confusion(tn_h, fp_h, fn_h, tp_h)
    tpr_l, fpr_l, pos_rate_l = rates_from_confusion(tn_l, fp_l, fn_l, tp_l)

    print("\n=== Fairness Metrics (High vs Low Amount Groups) ===")
    print(f"High Amount - TPR: {tpr_h:.4f}, FPR: {fpr_h:.4f}, Positive Rate: {pos_rate_h:.4f}")
    print(f"Low Amount  - TPR: {tpr_l:.4f}, FPR: {fpr_l:.4f}, Positive Rate: {pos_rate_l:.4f}")

    dp_ratio = pos_rate_h / pos_rate_l if pos_rate_l > 0 else np.nan
    print(f"Demographic Parity Ratio: {dp_ratio:.4f}")

    tpr_diff = abs(tpr_h - tpr_l)
    fpr_diff = abs(fpr_h - fpr_l)
    print(f"Equalized Odds - TPR Difference: {tpr_diff:.4f}")
    print(f"Equalized Odds - FPR Difference: {fpr_diff:.4f}")

    disparate_impact = dp_ratio
    print(f"Disparate Impact: {disparate_impact:.4f}")
