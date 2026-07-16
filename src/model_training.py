import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


def train_logistic_regression(X_train_sel, y_train_res):
    """
    Trains a baseline Logistic Regression model on resampled data and exports serialized model.
    """
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    log_reg.fit(X_train_sel, y_train_res)
    joblib.dump(log_reg, "models/logistic_regression.pkl")
    return log_reg


def train_random_forest(X_train_sel, y_train_res):
    """
    Train Random Forest using RandomizedSearchCV on resampled data.
    Optimizes for PR-AUC (average_precision) for highly imbalanced fraud evaluation.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_grid,
        n_iter=6,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("\n--- Running RandomizedSearchCV for Random Forest ---")
    search.fit(X_train_sel, y_train_res)
    rf = search.best_estimator_
    print(f"Best Hyperparameters: {search.best_params_}")
    
    joblib.dump(rf, "models/random_forest.pkl")
    return rf


def train_xgboost(X_train_sel, y_train_res):
    """
    Trains an XGBoost gradient boosting model if the package is present in the environment.
    """
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=42
        )
        xgb.fit(X_train_sel, y_train_res)
        joblib.dump(xgb, "models/xgboost.pkl")
        return xgb
    except ImportError:
        print("XGBoost not installed.")
        return None