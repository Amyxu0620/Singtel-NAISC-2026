import numpy as np
import pandas as pd


def generate_synthetic_drift(train_df: pd.DataFrame, output_path: str, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    df = train_df.copy()

    # Remove target for test-like dataset
    if "ChurnStatus" in df.columns:
        df = df.drop(columns=["ChurnStatus"])

    # Numeric drift
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["CustomerID"]]

    for col in numeric_cols[:5]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        shift = df[col].std() * 0.5 if df[col].std() > 0 else 1.0
        df[col] = df[col] + shift

    for col in numeric_cols[5:10]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col] * 1.5

    # Categorical drift
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["CustomerID", "Month"]]

    for col in cat_cols[:3]:
        if len(df) > 0:
            idx = rng.choice(len(df), size=max(1, len(df)//10), replace=False)
            df.loc[idx, col] = "UNSEEN_CATEGORY"

    df.to_csv(output_path, index=False)

def generate_adversarial_cases(train_df: pd.DataFrame, out_dir: str):
    base = train_df.copy()
    if "ChurnStatus" in base.columns:
        base = base.drop(columns=["ChurnStatus"])

    # 1. Missing columns
    missing_cols_df = base.drop(columns=base.columns[:3], errors="ignore")
    missing_cols_df.to_csv(f"{out_dir}/test_missing_cols.csv", index=False)

    # 2. Extra columns
    extra_cols_df = base.copy()
    extra_cols_df["ExtraNoiseFeature"] = np.random.randn(len(extra_cols_df))
    extra_cols_df["AnotherUnexpectedCol"] = "noise"
    extra_cols_df.to_csv(f"{out_dir}/test_extra_cols.csv", index=False)

    # 3. Renamed columns
    renamed_df = base.copy()
    rename_map = {}
    for i, col in enumerate(renamed_df.columns[:5]):
        rename_map[col] = f"mystery_feature_{i+1}"
    renamed_df = renamed_df.rename(columns=rename_map)
    renamed_df.to_csv(f"{out_dir}/test_renamed_cols.csv", index=False)
