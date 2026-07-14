import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import confusion_matrix

def plot_permutation_importance(rf, X_test_sel, y_test, selected_features):
    """
    Plot permutation importance for the Random Forest model.
    Shows which features influence predictions the most.
    """
    perm = permutation_importance(rf, X_test_sel, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": selected_features, "importance": perm.importances_mean})
    perm_df = perm_df.sort_values("importance", ascending=False)

    sns.barplot(x="importance", y="feature", data=perm_df.head(10))
    plt.title("Permutation Importance - Random Forest")
    plt.show()


def threshold_tuning(y_proba_rf, y_test):
    """
    Print precision and recall at different classification thresholds.
    Useful for fraud detection where recall is critical.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        y_pred_t = (y_proba_rf >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Threshold {t:.2f} | Precision {precision_t:.4f} | Recall {recall_t:.4f}")


def pdp_ice_plot(rf, X_test_sel_df, selected_features):
    """
    Generate PDP + ICE plots for one selected feature.
    Shows how predictions change as the feature varies.
    """
    feature_for_pdp = "V14" if "V14" in selected_features else selected_features[0]

    PartialDependenceDisplay.from_estimator(
        rf,
        X_test_sel_df,
        [feature_for_pdp],
        kind="both"
    )
    plt.title(f"PDP/ICE for {feature_for_pdp}")
    plt.show()
