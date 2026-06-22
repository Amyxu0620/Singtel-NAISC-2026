ID_COL = "CustomerID"
TARGET_COL = "ChurnStatus"
MONTH_COL = "Month"

LIGHTGBM_PARAMS = {
    "verbosity": -1,
    "objective": "binary",
    "is_unbalance": True,
    "random_state": 42,
    "importance_type": "gain",
}
