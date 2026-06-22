import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def compute_psi(train_col, test_col, bins=10):
    train_col = pd.to_numeric(train_col, errors="coerce").dropna()
    test_col = pd.to_numeric(test_col, errors="coerce").dropna()
    if len(train_col) == 0 or len(test_col) == 0:        
        return 0.0        
    breakpoints = np.linspace(train_col.min(), train_col.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    train_counts = pd.cut(train_col, bins=breakpoints).value_counts(sort=False)
    test_counts = pd.cut(test_col, bins=breakpoints).value_counts(sort=False)
    train_pct = (train_counts / len(train_col)).replace(0, 1e-4)
    test_pct = (test_counts / len(test_col)).replace(0, 1e-4)
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))    
    return round(psi, 4)

def detect_numerical_drift(train_df, test_df, numerical_cols):    
    if numerical_cols is None:
        numerical_cols = train_df.select_dtypes(include=['number']).columns.tolist()        
    rows = []
    for col in numerical_cols:        
        if col not in train_df.columns or col not in test_df.columns:            
            continue
        train_col = pd.to_numeric(train_df[col], errors="coerce").dropna()
        test_col = pd.to_numeric(test_df[col], errors="coerce").dropna()
        if len(train_col) == 0 or len(test_col) == 0:            
            continue
        ks_stat, ks_pvalue = ks_2samp(train_col, test_col)
        psi = compute_psi(train_col, test_col)
        is_drifted = (ks_pvalue < 0.05) or (psi > 0.10)
        if psi > 0.25:
            description = "Feature ranges explode in test set"        
        elif psi > 0.10:
            description = "Feature demonstrates greater skewness in test set compared to training set"        
        elif ks_pvalue < 0.05:
            description = "Subtle distribution shift detected in test set"        
        else:
            description = "No significant drift detected"
        rows.append({            
            "feature": col,            
            "type": str(train_df[col].dtype),            
            "ks_stat": round(float(ks_stat), 4),            
            "ks_pvalue": round(float(ks_pvalue), 6),            
            "psi": round(float(psi), 4),            
            "is_drifted": bool(is_drifted),            
            "drift_description": description,        
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    train = pd.read_csv("public_data/train.csv")
    test = pd.read_csv("public_data/test.csv")
    drift_results = detect_numerical_drift(train, test)
    print(f"\n{'='*60}")    
    print("NUMERICAL DRIFT DETECTION RESULTS")    
    print(f"{'='*60}")
    drifted = drift_results[drift_results["is_drifted"] == True]
    clean = drift_results[drift_results["is_drifted"] == False]
    print(f"\nDRIFTED COLUMNS ({len(drifted)}):")    
    for _, row in drifted.iterrows():
        print(f"  {row['feature']} | KS={row['ks_stat']} p={row['ks_pvalue']} PSI={row['psi']}")
        print(f"    -> {row['drift_description']}")
    print(f"\nCLEAN COLUMNS ({len(clean)}):")    
    for _, row in clean.iterrows():
        print(f"  {row['feature']}")
