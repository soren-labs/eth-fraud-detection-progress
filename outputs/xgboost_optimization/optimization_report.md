# XGBoost Optimization Report

## Experiment Design

- Fixed dataset split: train/validation/test derived from the same cleaned dataset.
- Optimization methods tried:
  - multiple hyperparameter combinations
  - feature engineering with ratio features
  - text frequency encoding for ERC20 token type fields
  - threshold optimization on validation set

## Search Summary

- Total experiments: `32`
- Best experiment: `deeper_more_trees__numeric_textfreq`
- Best feature variant: `numeric_textfreq`
- Best validation threshold: `0.66`
- Best validation F1 (tuned): `0.9429`
- Best validation Recall (tuned): `0.9226`

## Final Test Metrics (Optimized Model, Tuned Threshold)

- Accuracy: `0.9700`
- Precision: `0.9631`
- Recall: `0.8991`
- F1-score: `0.9300`
- ROC-AUC: `0.9923`
- Average Precision: `0.9799`

## Final Test Metrics (Optimized Model, Default Threshold 0.5)

- Accuracy: `0.9695`
- Precision: `0.9393`
- Recall: `0.9220`
- F1-score: `0.9306`
- ROC-AUC: `0.9923`
- Average Precision: `0.9799`

## Improvement Over XGBoost Baseline

- F1 improvement: `+0.0217`
- Recall improvement: `-0.0046`
- ROC-AUC improvement: `+0.0016`
- Recommended final strategy on test set: `default_0.5`

## Top 5 Features of Optimized Model

- `total ether received`: 0.138435
- `avg val received`: 0.072912
- `ERC20 min val rec`: 0.067481
- `Time Diff between first and last (Mins)`: 0.061990
- `Unique Received From Addresses`: 0.058632

## Output Files

- Search results: `search_results.csv`
- Best config: `best_config.json`
- Best metrics: `best_metrics.json`
- Best predictions: `best_test_predictions.csv`
- Validation search chart: `validation_top12_f1.png`
- Threshold chart: `best_threshold_curve.png`
- Comparison chart: `baseline_vs_optimized_metrics.png`
