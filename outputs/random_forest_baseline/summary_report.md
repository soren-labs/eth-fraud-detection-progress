# Random Forest Baseline Summary

## Experiment Setup

- Dataset: `transaction_dataset_cleaned.csv`
- Training samples: `7852`
- Test samples: `1964`
- Numeric features used: `38`
- Excluded columns: `FLAG, Address, ERC20 most sent token type, ERC20_most_rec_token_type`

## Core Metrics

- Accuracy: `0.9603`
- Precision: `0.9686`
- Recall: `0.8486`
- F1-score: `0.9046`
- ROC-AUC: `0.9863`
- Average Precision: `0.9651`

## Top 10 Features

- `Time Diff between first and last (Mins)`: 0.114998
- `avg val received`: 0.090261
- `total ether received`: 0.081628
- `Avg min between received tnx`: 0.071855
- `Unique Received From Addresses`: 0.060893
- `max value received`: 0.056617
- `total ether balance`: 0.049147
- `total Ether sent`: 0.047899
- `min value received`: 0.045165
- `ERC20 min val rec`: 0.043988

## Output Files

- Model: `random_forest_baseline.joblib`
- Metrics: `metrics.json`
- Classification report: `classification_report.txt`
- Feature importance: `feature_importance.csv`
- Test predictions: `test_predictions.csv`
- Confusion matrix chart: `confusion_matrix.png`
- ROC curve chart: `roc_curve.png`
- Precision-recall chart: `precision_recall_curve.png`
- Feature importance chart: `feature_importance_top20.png`
