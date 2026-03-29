# Interpretable Ethereum Fraud Detection Thesis Progress

This repository records the current progress of an undergraduate thesis project on **interpretable Ethereum fraud detection**.

The project focuses on building a machine learning based fraud risk detection pipeline for Ethereum addresses, and then using model explanation results to analyze **fraud type tendency**.  
The full pipeline now combines:

- machine learning based fraud detection
- SHAP-based interpretability
- business-rule-based fraud type tendency analysis
- LLM-assisted natural language explanation

At the current stage, the core pipeline of **data cleaning -> baseline modeling -> XGBoost optimization -> SHAP interpretation -> LLM-assisted explanation generation** has been completed.

## Thesis Scope

The full thesis is designed around the following major parts:

1. Research problem definition
2. Fraud type analysis
3. Dataset analysis and data cleaning
4. Feature system construction
5. Baseline model training
6. XGBoost model training and optimization
7. Model evaluation and comparison
8. SHAP global and case-level explanation
9. Fraud type tendency analysis
10. LLM-assisted report generation

## Current Progress

### Completed

The following parts have been completed:

#### 1. Fraud type analysis

The project currently focuses on three representative fraud categories:

- phishing and money laundering
- ponzi schemes
- ICO scams and rug pulls

Relevant documents:

- `docs/01-课题研究的欺诈类型分析.md`
- `docs/03-欺诈类型与数据集字段映射.md`
- `docs/欺诈类型报告-中文版.md`

#### 2. Dataset analysis and cleaning

The dataset has been analyzed and cleaned, including:

- column name normalization
- duplicate address merging
- missing value handling
- ERC20 feature normalization
- feature field dictionary generation

Relevant files:

- `docs/02-数据集分析情况.md`
- `docs/04-数据清洗方案.md`
- `processed/transaction_dataset_cleaning_report.md`
- `processed/transaction_dataset_cleaning_summary.json`
- `processed/transaction_dataset_field_dictionary.csv`

#### 3. Feature system construction

The feature system has been organized into several groups:

- temporal and lifecycle features
- transaction frequency and network features
- ETH amount features
- contract interaction features
- ERC20 features

Relevant document:

- `docs/03-欺诈类型与数据集字段映射.md`

#### 4. Baseline model training

A Random Forest baseline model has already been trained and evaluated.

Relevant outputs:

- `outputs/random_forest_baseline/summary_report.md`
- `outputs/random_forest_baseline/metrics.json`
- `outputs/random_forest_baseline/confusion_matrix.png`
- `outputs/random_forest_baseline/roc_curve.png`
- `outputs/random_forest_baseline/precision_recall_curve.png`

#### 5. XGBoost main model and optimization

An XGBoost baseline model has been trained, and a full optimization process has been completed, including:

- hyperparameter tuning
- lightweight feature engineering
- token frequency encoding
- threshold exploration

Relevant outputs:

- `outputs/xgboost_baseline/summary_report.md`
- `outputs/xgboost_optimization/优化结果汇总.md`
- `outputs/xgboost_optimization/optimization_report.md`
- `outputs/xgboost_optimization/search_results.csv`
- `outputs/xgboost_optimization/baseline_vs_optimized_metrics.png`

#### 6. Model evaluation and comparison

The repository already includes model comparison results between:

- Random Forest
- XGBoost baseline
- optimized XGBoost

Relevant outputs:

- `outputs/model_comparison/模型对比结果汇总.md`
- `outputs/model_comparison/model_metrics_comparison.csv`
- `outputs/model_comparison/metrics_comparison_bar.png`
- `outputs/model_comparison/roc_curve_comparison.png`
- `outputs/model_comparison/precision_recall_curve_comparison.png`

#### 7. SHAP interpretation and fraud type tendency analysis

SHAP interpretation has been completed at two levels:

- global feature importance
- address-level case explanation

Three fraud-tendency case examples have also been prepared for:

- phishing and money laundering tendency
- ponzi tendency
- ICO / rug pull tendency

Relevant outputs:

- `outputs/shap_explanations/SHAP结果汇总.md`
- `outputs/shap_explanations/global_shap_report.md`
- `outputs/shap_explanations/global_shap_bar_top20.png`
- `outputs/shap_explanations/global_shap_summary_top15.png`
- `outputs/shap_explanations/case_explanations.md`

#### 8. LLM-assisted explanation generation

The final explanation stage has now been implemented using the official DeepSeek API.  
This stage takes:

- optimized XGBoost model outputs
- SHAP global explanation results
- SHAP case-level explanation results

and generates:

- natural-language global progress summary
- case-level fraud risk explanations
- thesis-ready paragraphs
- defense-oriented talking points

Relevant outputs:

- `outputs/llm_explanations/LLM结果汇总.md`
- `outputs/llm_explanations/global_summary.md`
- `outputs/llm_explanations/case_reports.md`
- `outputs/llm_explanations/case_reports.csv`

## Current Best Result

The current recommended main model is the **optimized XGBoost model**.

Current best test result:

- `F1 = 0.9306`
- `ROC-AUC = 0.9923`

This result is currently the main experimental result used for thesis writing and advisor progress reporting.

## Repository Structure

### `docs/`

Research notes, progress documents, chapter design, prompt drafts, and thesis planning materials.

### `scripts/`

Core scripts for:

- data cleaning
- baseline training
- XGBoost training
- model comparison
- XGBoost optimization
- SHAP explanation

### `processed/`

Cleaned-data metadata and cleaning reports.

### `outputs/`

Experiment results, charts, comparison reports, optimization outputs, and SHAP explanation outputs.

## Excluded Content

The following items are intentionally excluded from this repository:

- raw thesis Word documents
- raw source dataset
- cleaned full dataset file
- model binary files (`.joblib`)
- local IDE files
- external third-party projects and environments

## Recommended Reading Order

If someone wants to quickly understand the current project progress, the recommended reading order is:

1. `docs/09-XGBoost训练计划与论文答辩要点.md`
2. `docs/10-论文整体章节设计.md`
3. `outputs/model_comparison/模型对比结果汇总.md`
4. `outputs/xgboost_optimization/优化结果汇总.md`
5. `outputs/shap_explanations/SHAP结果汇总.md`

## Current Project Status in One Sentence

The project has completed the full current pipeline of **fraud type analysis -> data cleaning -> baseline model -> XGBoost optimization -> SHAP interpretation -> LLM-assisted explanation generation**.
