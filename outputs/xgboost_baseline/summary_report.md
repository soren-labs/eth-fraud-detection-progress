# XGBoost Baseline Summary

## Experiment Setup

- Dataset: `transaction_dataset_cleaned.csv`
- Training samples: `7852`
- Test samples: `1964`
- Numeric features used: `38`
- Excluded columns: `FLAG, Address, ERC20 most sent token type, ERC20_most_rec_token_type`

## Core Metrics

- Accuracy: `0.9588`
- Precision: `0.8918`
- Recall: `0.9266`
- F1-score: `0.9089`
- ROC-AUC: `0.9906`
- Average Precision: `0.9758`

## Top 10 Features

- `total ether received`: 0.145424
- `avg val received`: 0.087299
- `Time Diff between first and last (Mins)`: 0.074037
- `ERC20 min val rec`: 0.062006
- `Avg min between received tnx`: 0.053814
- `Unique Received From Addresses`: 0.052308
- `ERC20 max val rec`: 0.033094
- `min value received`: 0.032572
- `total transactions (including tnx to create contract)`: 0.032189
- `Sent tnx`: 0.027470

## Output Files

- Model: `xgboost_baseline.joblib`
- Metrics: `metrics.json`
- Classification report: `classification_report.txt`
- Feature importance: `feature_importance.csv`
- Test predictions: `test_predictions.csv`
- Confusion matrix chart: `confusion_matrix.png`
- ROC curve chart: `roc_curve.png`
- Precision-recall chart: `precision_recall_curve.png`
- Feature importance chart: `feature_importance_top20.png`
