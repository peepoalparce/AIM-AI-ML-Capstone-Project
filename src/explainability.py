import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def lime_explanations(X_train_sel, X_test_sel, selected_features, y_test, rf):
    """
    Generate LIME explanations for a few fraud cases.
    Shows which features contributed most to the model’s prediction.
    """
    feature_names_lime = list(selected_features)
    class_names_lime = ["Non-Fraud", "Fraud"]

    explainer = LimeTabularExplainer(
        training_data=X_train_sel,
        feature_names=feature_names_lime,
        class_names=class_names_lime,
        mode="classification",
        discretize_continuous=True
    )

    fraud_indices = np.where(y_test == 1)[0]
    sample_indices = fraud_indices[:5]

    for idx in sample_indices:
        instance = X_test_sel[idx]
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=rf.predict_proba,
            num_features=10
        )
        print(f"\n=== LIME Explanation for Test Index {idx} ===")
        print(exp.as_list())
