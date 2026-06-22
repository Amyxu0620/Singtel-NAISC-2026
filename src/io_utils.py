import pandas as pd


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def infer_column_types(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str,
    month_col: str | None = None
) -> tuple[list[str], list[str]]:
    exclude_cols = {target_col, id_col}
    if month_col is not None:
        exclude_cols.add(month_col)

    common_cols = [
        col for col in train_df.columns
        if col in test_df.columns and col not in exclude_cols
    ]

    numerical_cols = []
    categorical_cols = []

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return numerical_cols, categorical_cols

def resolve_special_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols & test_cols

    # ID column
    id_candidates = [c for c in common_cols if "id" in c.lower()]
    id_col = id_candidates[0] if id_candidates else None

    # Month / time column
    month_candidates = [c for c in common_cols if "month" in c.lower() or "date" in c.lower() or "time" in c.lower()]
    month_col = month_candidates[0] if month_candidates else None

    # Target column
    target_candidates = [c for c in train_df.columns if "churn" in c.lower() or "target" in c.lower() or "label" in c.lower()]
    target_col = target_candidates[0] if target_candidates else None

    return id_col, target_col, month_col
