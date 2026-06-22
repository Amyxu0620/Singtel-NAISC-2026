
**Team Name:** Parallax

Parallax is a data drift detection and mitigation pipeline built for the NAISC Singtel Hackathon. It compares training and test datasets, detects numerical and categorical drift, applies mitigation strategies, trains a churn prediction model, and produces prediction outputs for evaluation.

## Project Overview
In real-world machine learning systems, data distributions can change over time. When the test data no longer resembles the training data, model performance may degrade. This project addresses that problem by building an end-to-end pipeline that:

- loads train and test datasets
- identifies special columns such as ID, target, and month/time columns
- detects numerical drift using:  
- Kolmogorov–Smirnov (KS) test  
- Population Stability Index (PSI)
- detects categorical drift using:  
- unseen-category checks  
- chi-squared tests on category distributions
- applies mitigation strategies based on drift severity
- trains a LightGBM binary classifier
- saves both:  
- `prediction.csv`  
- `model.joblib`

## Repository Structure

```text
.
├── main.py
├── config.py
├── io_utils.py
├── drift_numeric_detection.py
├── drift_numeric_mitigation.py
├── drift_categorical.py
├── model_utils.py
├── reporting.py
├── generate_test_cases.py
├── run_stress_test.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Main Workflow

The main pipeline is run from `main.py`.

It performs the following steps:

1. Load the training and test datasets
2. Resolve key special columns such as:   
- ID column   
- target column   
- month/time column
3. Infer numerical and categorical feature columns
4. Detect numerical drift
5. Apply numerical drift mitigation
6. Detect and mitigate categorical drift
7. Combine drift and mitigation summaries
8. Train the final LightGBM model
9. Save:   
- `prediction.csv`   
- `model.joblib`

## File Descriptions

### `main.py`
Entry point for the full pipeline. Runs the complete drift detection, mitigation, training, and output workflow.

### `config.py`
Stores default configuration values such as:
- `CustomerID` as the ID column
- `ChurnStatus` as the target column
- `Month` as the month column
- LightGBM hyperparameters

### `io_utils.py`
Handles:
- loading CSV data
- inferring numerical and categorical columns
- resolving special columns such as ID, target, and month/time fields

### `drift_numeric_detection.py`
Detects numerical drift using:
- KS test
- PSI
It produces a DataFrame containing drift statistics and drift descriptions for each numerical feature.

### `drift_numeric_mitigation.py`
Applies numerical drift mitigation strategies based on detected drift severity.
Mitigation logic:
- PSI > 0.25 → Robust scaling
- 0.10 < PSI ≤ 0.25 → Log transform + robust scaling
- PSI ≤ 0.10 but KS significant → Input re-alignment
It also supports an optional sliding-window strategy for temporally ordered training data.

### `drift_categorical.py`
Detects and mitigates categorical drift using:
- unseen-category checks
- chi-squared tests on category distributions
Mitigation logic:
- moderate drift → map unseen values to `Unknown`
- severe drift → drop feature
Remaining categorical features are then encoded numerically for model training.

### `model_utils.py`
Trains the final LightGBM classifier and computes AU-PRC metrics.
Outputs:
- trained model object
- prediction DataFrame
- training/test AU-PRC metrics

### `reporting.py`
Builds and prints summary tables for:
- detected drift
- mitigation actions
- runtime
- model performance

### `generate_test_cases.py`
Utility script for generating synthetic and adversarial test datasets, including:
- synthetic numeric drift
- synthetic categorical drift
- missing-column cases
- extra-column cases
- renamed-column cases

This is optional and not required to run the main pipeline.

### `run_stress_test.py`
Utility script for running the pipeline on multiple test cases and writing logs to a file.

This is optional and not required to run the main pipeline.

## Installation

Install dependencies with:
```bash
pip install -r requirements.txt
```
## Usage

Run the full pipeline with:
```bash
python main.py --train_data_filepath <train_csv_path> --test_data_filepath <test_csv_path>
```

### Example
```bash
python main.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
```

## Outputs

After running the pipeline, the following files are produced:

### `prediction.csv`
Contains:
- `CustomerID`
- `probability_score`

### `model.joblib`
Serialized trained LightGBM model.

The pipeline also prints:
- drift detection summary
- mitigation summary
- runtime
- model performance metrics

## Drift Detection And Mitigation Summary

### Numerical Drift
Numerical drift is detected using KS test and PSI.

Interpretation used in the project:
- PSI > 0.25: strong drift / range explosion
- 0.10 < PSI ≤ 0.25: moderate drift / skewness shift
- KS significant with low PSI: subtle distribution shift

### Categorical Drift
Categorical drift is detected when:
- new categories appear in the test set
- category proportions differ significantly based on chi-squared testing

Mitigation options:
- Map unknown
- Drop feature
- No action needed

## Model

The final predictive model is a LightGBM binary classifier configured for imbalanced classification.

Performance is reported using AU-PRC, which is appropriate for churn prediction and other imbalanced binary classification tasks.

## Notes

- The pipeline automatically attempts to infer key dataset columns if they are present under expected names.
- Only columns shared by both training and test datasets are used in modelling.
- Categorical features are encoded after drift mitigation.
- The stress-testing scripts are optional and mainly intended for robustness testing.
