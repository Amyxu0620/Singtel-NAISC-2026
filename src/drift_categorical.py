import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
 
 
def detect_drift_single_column(train_col, test_col):
    """Checks one column for drift. Returns a dict with results."""
    train_cats = set(train_col.dropna().unique())
    test_cats = set(test_col.dropna().unique())
    new_in_test = test_cats - train_cats
 
    # Chi-squared test
    all_cats = sorted([str(x) for x in (train_cats | test_cats)])
    train_counts = train_col.value_counts().reindex(all_cats, fill_value=0)
    test_counts = test_col.value_counts().reindex(all_cats, fill_value=0)
    contingency = pd.DataFrame({'train': train_counts, 'test': test_counts}).T
 
    try:
        _, p_value, _, _ = chi2_contingency(contingency)
    except ValueError:
        p_value = 1.0
 
    is_drifted = len(new_in_test) > 0 or p_value < 0.05
 
    # Decide mitigation
    if not is_drifted:
        mitigation = "No action needed"
    elif len(new_in_test) > 5 or p_value < 0.001:
        mitigation = "Drop Feature"
    else:
        mitigation = "Map Unknown"
 
    # Build description
    parts = []
    if new_in_test:
        parts.append(f"New categories in test set: {sorted(new_in_test)}")
    if p_value < 0.05:
        parts.append(f"Proportions shifted (p={p_value:.4f})")
 
    return {
        'column': train_col.name,
        'column_type': str(train_col.dtype),
        'is_drifted': is_drifted,
        'drift_description': "; ".join(parts) if parts else "No drift",
        'mitigation': mitigation,
        'new_categories': new_in_test,
        'p_value': round(p_value, 6)
    }

def detect_and_mitigate_categorical_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Detects and mitigates drift in categorical columns.
 
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Test data.
    categorical_cols : list[str]
        List of categorical column names to check.
        (Person A already figured out which columns are categorical
         and passes them to you here — you don't need to find them.)
 
    Returns
    -------
    train_df : pd.DataFrame
        Training data after mitigation + encoding.
    test_df : pd.DataFrame
        Test data after mitigation + encoding.
    drift_summary_df : pd.DataFrame
        Table showing which columns drifted, how, and what was done.
        Columns: [Columns with Drift, Column Type, Drift Description, Drift Mitigation]
    mitigation_actions : list[dict]
        List of actions taken, one dict per drifted column.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
 
    drift_results = []
    mitigation_actions = []
 
    # Detect and Mitigate
    for col in categorical_cols:
        # Handle column missing from test
        if col not in test_df.columns:
            drift_results.append({
                'column': col, 
                'column_type': 'object', 
                'is_drifted': True,
                'drift_description': 'Column missing from test set',
                'mitigation': 'Drop Feature', 
                'new_categories': set(), 
                'p_value': 0.0
            })
            train_df = train_df.drop(columns=[col])
            mitigation_actions.append({
                "feature": col,
                "column_type": "object",
                "drift_description": "Column missing from test set",
                "mitigation_applied": "Drop Feature"
            })
            continue
 
        # Handle column missing from train
        if col not in train_df.columns:
            continue
 
        # Detect drift
        result = detect_drift_single_column(train_df[col], test_df[col])
 
        # Apply mitigation
        if result['mitigation'] == "Drop Feature":
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])
            mitigation_actions.append({
                "feature": col,
                "column_type": "object",
                "drift_description": result["drift_description"],
                "mitigation_applied": "Drop Feature"
            })
 
        elif result['mitigation'] == "Map Unknown":
            known = set(train_df[col].dropna().unique())
            test_df[col] = test_df[col].apply(
                lambda x, k=known: x if pd.isna(x) or x in k else 'Unknown'
            )
            mitigation_actions.append({
                "feature": col,
                "column_type": "object",
                "drift_description": result["drift_description"],
                "mitigation_applied": "Map Unknown"
            })
 
        if result['is_drifted']:
            drift_results.append(result)
 
    #Encode remaining categorical columns to numbers
    remaining_cats = [c for c in categorical_cols
                      if c in train_df.columns and c in test_df.columns]
 
    for col in remaining_cats:
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        mapping = {val: idx for idx, val in enumerate(sorted(set(combined.fillna("nan").astype(str).tolist())))}
        train_df[col] = train_df[col].astype(str).map(mapping)
        test_df[col] = test_df[col].astype(str).map(mapping)
 
    # Build Drift Summary Table
    if drift_results:
        drift_summary_df = pd.DataFrame([
            {
                'Columns with Drift': r['column'],
                'Column Type': r['column_type'],
                'Drift Description': r['drift_description'],
                'Drift Mitigation': r['mitigation']
            }
            for r in drift_results
        ])
    else:
        drift_summary_df = pd.DataFrame(
            columns=['Columns with Drift', 'Column Type', 'Drift Description', 'Drift Mitigation']
        )
 
    return train_df, test_df, drift_summary_df, mitigation_actions

# Test
if __name__ == "__main__":
    train = pd.DataFrame({
        'CustomerID': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Plan': ['Basic', 'Premium', 'Basic', 'Basic', 'Premium'],
        'Region': ['North', 'South', 'North', 'South', 'North'],
        'Age': [25, 30, 35, 40, 45],
        'ChurnStatus': ['No', 'Yes', 'No', 'No', 'Yes'],
    })
    test = pd.DataFrame({
        'CustomerID': ['B1', 'B2', 'B3', 'B4', 'B5'],
        'Gender': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'Plan': ['Basic', 'Enterprise', 'Premium', 'Enterprise', 'Basic'],
        'Region': ['East', 'West', 'North', 'East', 'West'],
        'Age': [28, 33, 45, 52, 61],
        'ChurnStatus': ['No', 'Yes', 'No', 'No', 'Yes'],
    })
 
    cat_cols = ['Gender', 'Plan', 'Region']
    train_out, test_out, summary, actions = detect_and_mitigate_categorical_drift(
        train, test, cat_cols
    )
 
    print("Drift Summary Table:")
    print(summary.to_string(index=False))
    print("\nMitigation Actions:", actions)
    print("\nProcessed train:\n", train_out)
    print("\nProcessed test:\n", test_out)
