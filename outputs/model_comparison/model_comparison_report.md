# Model Comparison Report

## Metrics Table

| Metric | Random Forest | XGBoost | Better Model |
| --- | ---: | ---: | --- |
| accuracy | 0.9603 | 0.9588 | Random Forest |
| precision | 0.9686 | 0.8918 | Random Forest |
| recall | 0.8486 | 0.9266 | XGBoost |
| f1 | 0.9046 | 0.9089 | XGBoost |
| roc_auc | 0.9863 | 0.9906 | XGBoost |
| average_precision | 0.9651 | 0.9758 | XGBoost |

## Interpretation

- On `accuracy`, better model: `Random Forest`.
- On `precision`, better model: `Random Forest`.
- On `recall`, better model: `XGBoost`.
- On `f1`, better model: `XGBoost`.
- On `roc_auc`, better model: `XGBoost`.
- On `average_precision`, better model: `XGBoost`.

## Output Files

- Metrics comparison CSV: `model_metrics_comparison.csv`
- Metrics bar chart: `metrics_comparison_bar.png`
- ROC comparison chart: `roc_curve_comparison.png`
- PR comparison chart: `precision_recall_curve_comparison.png`
