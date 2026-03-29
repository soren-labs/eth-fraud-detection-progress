# English Progress Diagram Prompt

## 1. Diagram Spec

```text
DIAGRAM_TYPE:
TECHNICAL ROADMAP / RESEARCH FLOWCHART

GOAL:
Show the current thesis progress of an undergraduate project on interpretable Ethereum fraud detection, with a clean academic layout that clearly distinguishes completed stages from the unfinished LLM stage.

AUDIENCE:
thesis advisor, mid-term review audience, undergraduate defense committee

CANVAS_RATIO:
16:9

TITLE:
Interpretable Ethereum Fraud Detection with ML and LLM

SECTIONS:
Section A: Problem and fraud scope
Section B: Data and feature preparation
Section C: Modeling and optimization
Section D: Interpretation and pending extension
Bottom strip: current progress and best result

NODES:
Research Objective
Fraud Type Analysis
Data Cleaning
Feature System
Baseline Model
XGBoost Training and Optimization
Evaluation and Comparison
SHAP Interpretation
LLM Report Generation (Pending)

EDGES:
Research Objective -> Fraud Type Analysis
Fraud Type Analysis -> Data Cleaning
Data Cleaning -> Feature System
Feature System -> Baseline Model
Baseline Model -> XGBoost Training and Optimization
XGBoost Training and Optimization -> Evaluation and Comparison
Evaluation and Comparison -> SHAP Interpretation
SHAP Interpretation -> LLM Report Generation (Pending)

LABELS:
"Research Objective"
"Fraud Type Analysis"
"Data Cleaning"
"Feature System"
"Baseline Model"
"XGBoost Training and Optimization"
"Evaluation and Comparison"
"SHAP Interpretation"
"LLM Report Generation (Pending)"
"Completed"
"Pending"
"Best Result"

DATA_POINTS:
F1 = 0.9306
ROC-AUC = 0.9923

VISUAL_STYLE:
clean academic infographic
white or very light gray background
dark gray text
one accent color in deep blue
one secondary accent color in muted teal
flat vector blocks
thin arrows
subtle shadows
clear spacing
publication-ready layout

NON_NEGOTIABLES:
do not invent extra workflow steps
do not add unsupported metrics
preserve the exact metric values
keep the last LLM module visually marked as pending
keep text short and readable
avoid dense paragraph blocks inside nodes
```

## 2. Render Prompt

```text
Create a clean academic technical roadmap diagram for an undergraduate thesis project titled "Interpretable Ethereum Fraud Detection with ML and LLM". Use a 16:9 canvas and a left-to-right then slightly wrapped roadmap layout with nine rounded rectangular modules connected by thin directional arrows.

The diagram should include these modules in order:
1. "Research Objective"
2. "Fraud Type Analysis"
3. "Data Cleaning"
4. "Feature System"
5. "Baseline Model"
6. "XGBoost Training and Optimization"
7. "Evaluation and Comparison"
8. "SHAP Interpretation"
9. "LLM Report Generation (Pending)"

Use these short internal labels for each module:

"Research Objective"
- fraud risk detection
- interpretable analysis
- type tendency judgment

"Fraud Type Analysis"
- phishing and laundering
- ponzi schemes
- ICO scams and rug pulls

"Data Cleaning"
- column normalization
- missing value handling
- duplicate address merge
- ERC20 feature cleanup

"Feature System"
- temporal features
- network features
- ETH amount features
- contract interaction features
- ERC20 features

"Baseline Model"
- Random Forest baseline
- initial comparable result

"XGBoost Training and Optimization"
- main model
- hyperparameter tuning
- text frequency encoding
- lightweight feature engineering
- threshold exploration

"Evaluation and Comparison"
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- final recommended model

"SHAP Interpretation"
- global importance
- address-level explanation
- fraud type tendency analysis

"LLM Report Generation (Pending)"
- natural language risk report
- SHAP-guided explanation
- current status: pending

Visually group the modules into four higher-level stages:
Stage 1: Problem Definition
Stage 2: Data Preparation
Stage 3: Modeling
Stage 4: Interpretation and Extension

Use standard solid blue-outline modules for the completed stages. Use a light gray dashed-outline module for "LLM Report Generation (Pending)" so it is visibly unfinished.

At the bottom of the figure, add two clean summary strips:
Left strip: "Completed: data cleaning, baseline model, XGBoost optimization, SHAP interpretation"
Right strip: "Best Result: F1 = 0.9306, ROC-AUC = 0.9923"

Style the figure as a publication-ready academic roadmap:
- white or near-white background
- dark gray or black text
- deep blue as the main accent color
- muted teal as a secondary accent color
- flat vector appearance
- subtle section grouping
- no decorative clutter
- no heavy gradients
- no thick borders
- no photorealistic elements
- balanced whitespace
- highly legible English typography

Keep the structure faithful to the provided content. Do not invent extra nodes, extra metrics, or unsupported claims. Keep labels concise and visually balanced.
```

## 3. Fast Alternate Prompt

```text
Design a clean academic roadmap diagram on a 16:9 canvas for the thesis workflow "Interpretable Ethereum Fraud Detection with ML and LLM". Show these stages in order with thin arrows and rounded boxes: Research Objective, Fraud Type Analysis, Data Cleaning, Feature System, Baseline Model, XGBoost Training and Optimization, Evaluation and Comparison, SHAP Interpretation, LLM Report Generation (Pending). Use short English labels inside each box, white background, dark text, one deep blue accent, minimal teal secondary accent, flat vector styling, balanced spacing, and high legibility. Mark the final LLM stage as pending with a gray dashed box. Add a bottom summary strip with "Completed: data cleaning, baseline model, XGBoost optimization, SHAP interpretation" and "Best Result: F1 = 0.9306, ROC-AUC = 0.9923". Do not add extra steps or unsupported metrics.
```

## 4. Review Checklist

- structure correct
- all workflow stages present
- completed vs pending visually clear
- labels short and readable
- hierarchy clear
- spacing balanced
- exact metrics preserved
- no invented claims

## 5. Revision Moves

1. Reduce text inside each box to at most 3 bullet lines if the layout still feels crowded.
2. Replace large arrows with thinner connector lines if the figure looks too heavy.
3. Increase whitespace between Stage 2 and Stage 3 if the middle region feels compressed.
4. Make the pending LLM box lighter and more separated so the unfinished status is obvious.
5. If the figure still looks busy, collapse the four stage titles into subtle top labels rather than large section headers.
