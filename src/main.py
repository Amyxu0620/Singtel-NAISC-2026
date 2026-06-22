import argparse
import time
import joblib
from config import ID_COL, TARGET_COL, MONTH_COL
from io_utils import load_data, infer_column_types, resolve_special_columns
from drift_numeric_detection import detect_numerical_drift
from drift_numeric_mitigation import mitigate_numerical_drift
from drift_categorical import detect_and_mitigate_categorical_drift
from model_utils import train_and_score_model
from reporting import combine_drift_tables, build_mitigation_table, print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="NAISC drift detection and mitigation pipeline")
    parser.add_argument("--train_data_filepath", type=str, required=True)
    parser.add_argument("--test_data_filepath", type=str, required=True)
    return parser.parse_args()

def main():
    start_time = time.time()

    args = parse_args()

    train_df, test_df = load_data(args.train_data_filepath, args.test_data_filepath)
    
    resolved_id_col, resolved_target_col, resolved_month_col = resolve_special_columns(train_df, test_df)

    id_col = resolved_id_col or ID_COL
    target_col = resolved_target_col or TARGET_COL
    month_col = resolved_month_col or MONTH_COL
    
    numerical_cols, categorical_cols = infer_column_types(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        id_col=id_col,
        month_col=month_col
    )

    numerical_drift_df = detect_numerical_drift(
        train_df=train_df,
        test_df=test_df,
        numerical_cols=numerical_cols
    )

    train_num_fixed, test_num_fixed, num_mitigation_log = mitigate_numerical_drift(
        train_df=train_df,
        test_df=test_df,
        numerical_drift_df=numerical_drift_df,
        month_col=month_col
    )

    train_final, test_final, categorical_drift_df, cat_mitigation_log = (
        detect_and_mitigate_categorical_drift(
            train_df=train_num_fixed,
            test_df=test_num_fixed,
            categorical_cols=categorical_cols
        )
    )

    drift_summary_df = combine_drift_tables(
        numerical_drift_df=numerical_drift_df,
        categorical_drift_df=categorical_drift_df
    )

    mitigation_summary_df = build_mitigation_table(
        num_mitigation_log=num_mitigation_log,
        cat_mitigation_log=cat_mitigation_log
    )

    model, prediction_df, metrics = train_and_score_model(
        train_df=train_final,
        test_df=test_final,
        target_col=target_col,
        id_col=id_col
    )

    prediction_df.to_csv("prediction.csv", index=False)
    joblib.dump(model, "model.joblib")

    elapsed_seconds = time.time() - start_time

    print_summary(
        drift_df=drift_summary_df,
        mitigation_df=mitigation_summary_df,
        metrics=metrics,
        elapsed_seconds=elapsed_seconds
    )

    print("\nSaved prediction.csv and model.joblib")


if __name__ == "__main__":
    main()
