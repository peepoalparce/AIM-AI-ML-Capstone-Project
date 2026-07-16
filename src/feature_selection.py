import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def select_features_with_rf(X_train_res, y_train_res, X_test_scaled, X_train_scaled):
    """
    Extracts high-value features using Random Forest feature importance.
    Filters out noisy variables below the median importance threshold.
    """
    rf_temp = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_temp.fit(X_train_res, y_train_res)

    selector = SelectFromModel(rf_temp, threshold="median", prefit=True)
    X_train_sel = selector.transform(X_train_res)
    X_test_sel = selector.transform(X_test_scaled)
    selected_features = X_train_scaled.columns[selector.get_support()]

    print("\nSelected features:", list(selected_features))

    X_test_sel_df = pd.DataFrame(X_test_sel, columns=selected_features)
    return X_train_sel, X_test_sel, X_test_sel_df, selected_features