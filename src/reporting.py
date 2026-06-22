import pandas as pd


def combine_drift_tables(
    numerical_drift_df: pd.DataFrame,
    categorical_drift_df: pd.DataFrame
) -> pd.DataFrame:
    return pd.concat([numerical_drift_df, categorical_drift_df], ignore_index=True)


def build_mitigation_table(
    num_mitigation_log: list[dict],
    cat_mitigation_log: list[dict]
) -> pd.DataFrame:
    all_logs = num_mitigation_log + cat_mitigation_log
    if not all_logs:
        return pd.DataFrame(columns=["feature", "column_type", "drift_description", "mitigation_applied"])
    return pd.DataFrame(all_logs)


def print_summary(
    drift_df: pd.DataFrame,
    mitigation_df: pd.DataFrame,
    metrics: dict,
    elapsed_seconds: float
) -> None:
    print("\n=== Data Drift Detection & Mitigation Summary ===")
    if drift_df.empty:
        print("No drift detected.")
    else:
        display_cols = [c for c in [
            "Columns with Drift", "Column Type", "Drift Description", "Drift Mitigation",
            "feature", "type", "drift_description"
        ] if c in drift_df.columns]
        print(drift_df[display_cols].to_string(index=False))

    print("\n=== Runtime (in seconds) ===")
    print(f"\n{elapsed_seconds:.3f}")
    
    print("\n=== Model Performance Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
