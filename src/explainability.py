import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def plot_permutation_importance(rf, X_test_sel, y_test, selected_features):
    """
    Calculates permutation importance on unseen test data to assess feature impact
    on trained model generalization performance.
    """
    perm = permutation_importance(rf, X_test_sel, y_test, n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({"feature": selected_features, "importance": perm.importances_mean})
    perm_df = perm_df.sort_values("importance", ascending=False)

    sns.barplot(x="importance", y="feature", data=perm_df.head(10))
    plt.title("Permutation Importance - Random Forest")
    plt.show()


def pdp_ice_plot(rf, X_test_sel_df, selected_features):
    """
    Generates Partial Dependence (PDP) and Individual Conditional Expectation (ICE) plots
    to visualize non-linear relationships between top features and fraud probabilities.
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


def lime_explanations(X_train_sel, X_test_sel, selected_features, y_test, rf):
    """
    Generates LIME (Local Interpretable Model-agnostic Explanations) for individual fraud predictions
    to provide model transparency and feature-level justification.
    """
    from lime.lime_tabular import LimeTabularExplainer

    X_train_lime = X_train_sel
    X_test_lime = X_test_sel
    feature_names_lime = list(selected_features)
    class_names_lime = ["Non-Fraud", "Fraud"]

    explainer = LimeTabularExplainer(
        training_data=X_train_lime,
        feature_names=feature_names_lime,
        class_names=class_names_lime,
        mode="classification",
        discretize_continuous=True
    )

    fraud_indices = np.where(y_test == 1)[0]
    sample_indices = fraud_indices[:5]

    for idx in sample_indices:
        instance = X_test_lime[idx]
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=rf.predict_proba,
            num_features=10
        )
        print(f"\n=== LIME Explanation for Test Index {idx} (Fraud Case) ===")
        print(exp.as_list())