import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from config import LIGHTGBM_PARAMS


def train_and_score_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Convert target to binary 0/1
    y_train = train_df[target_col].map({"No": 0, "Yes": 1})
    if y_train.isna().any():
        y_train = pd.to_numeric(train_df[target_col], errors="coerce")

    X_train = train_df.drop(columns=[target_col], errors="ignore")
    X_test = test_df.drop(columns=[target_col], errors="ignore")

    # Keep only common columns
    common_cols = [c for c in X_train.columns if c in X_test.columns]

    if len(common_cols) < 5:
        print("⚠️ WARNING: very few overlapping columns between train and test.")
        
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Remove ID column from features
    X_train = X_train.drop(columns=[id_col], errors="ignore")
    X_test = X_test.drop(columns=[id_col], errors="ignore")

    model = LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    train_auprc = average_precision_score(y_train, train_probs)

    test_probs = model.predict_proba(X_test)[:, 1]

    prediction_df = pd.DataFrame({
        "CustomerID": test_df[id_col] if id_col in test_df.columns else range(len(test_df)),
        "probability_score": test_probs
    })

    metrics = {
        "Train AU-PRC": round(float(train_auprc), 6),
        "Test AU-PRC": "N/A"
    }

    # If the synthetic/adversarial test contains target, evaluate it
    if target_col in test_df.columns:
        y_test = test_df[target_col].map({"No": 0, "Yes": 1})
        if y_test.isna().any():
            y_test = pd.to_numeric(test_df[target_col], errors="coerce")
        if not y_test.isna().all():
            test_auprc = average_precision_score(y_test, test_probs)
            metrics["Test AU-PRC"] = round(float(test_auprc), 6)

    return model, prediction_df, metrics
