import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train_sel, y_train_res):
    """
    Train and save a Logistic Regression model.
    """
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    log_reg.fit(X_train_sel, y_train_res)
    joblib.dump(log_reg, "models/logistic_regression.pkl")
    return log_reg


def train_random_forest(X_train_sel, y_train_res):
    """
    Train and save a Random Forest model.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_sel, y_train_res)
    joblib.dump(rf, "models/random_forest.pkl")
    return rf


def train_xgboost(X_train_sel, y_train_res):
    """
    Train and save an XGBoost model (if installed).
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
