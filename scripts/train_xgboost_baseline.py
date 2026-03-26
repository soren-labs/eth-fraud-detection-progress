from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "processed" / "transaction_dataset_cleaned.csv"
OUTPUT_DIR = ROOT / "outputs" / "xgboost_baseline"

MODEL_PATH = OUTPUT_DIR / "xgboost_baseline.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "feature_importance.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
SUMMARY_MD = OUTPUT_DIR / "summary_report.md"

CONFUSION_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = OUTPUT_DIR / "roc_curve.png"
PR_CURVE_PNG = OUTPUT_DIR / "precision_recall_curve.png"
FEATURE_IMPORTANCE_PNG = OUTPUT_DIR / "feature_importance_top20.png"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    label_col = "FLAG"
    id_cols = ["Address"]
    text_cols = ["ERC20 most sent token type", "ERC20_most_rec_token_type"]
    excluded_cols = [label_col] + id_cols + text_cols
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].copy()
    y = df[label_col].copy()
    address_series = df["Address"].copy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        addr_train,
        addr_test,
    ) = train_test_split(
        X,
        y,
        address_series,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_path": str(DATA_PATH),
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
        "feature_count": int(len(feature_cols)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "parameters": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 4,
            "scale_pos_weight": scale_pos_weight,
        },
        "excluded_columns": excluded_cols,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    CLASSIFICATION_REPORT_PATH.write_text(
        classification_report(y_test, y_pred, digits=4), encoding="utf-8"
    )
    joblib.dump(model, MODEL_PATH)

    feature_importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame(
        {
            "Address": addr_test.values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_prob_fraud": y_prob,
        }
    ).sort_values("y_prob_fraud", ascending=False)
    predictions.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title("XGBoost Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}", linewidth=2, color="#2f7d4a")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        recall_vals,
        precision_vals,
        label=f"AP = {metrics['average_precision']:.4f}",
        linewidth=2,
        color="#2f7d4a",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("XGBoost Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PR_CURVE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    top20 = feature_importance.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["importance"], color="#2f7d4a")
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Top 20 Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    top10_lines = []
    for row in feature_importance.head(10).itertuples(index=False):
        top10_lines.append(f"- `{row.feature}`: {row.importance:.6f}")

    summary_lines = [
        "# XGBoost Baseline Summary",
        "",
        "## Experiment Setup",
        "",
        f"- Dataset: `{DATA_PATH.name}`",
        f"- Training samples: `{X_train.shape[0]}`",
        f"- Test samples: `{X_test.shape[0]}`",
        f"- Numeric features used: `{len(feature_cols)}`",
        f"- Excluded columns: `{', '.join(excluded_cols)}`",
        "",
        "## Core Metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Precision: `{metrics['precision']:.4f}`",
        f"- Recall: `{metrics['recall']:.4f}`",
        f"- F1-score: `{metrics['f1']:.4f}`",
        f"- ROC-AUC: `{metrics['roc_auc']:.4f}`",
        f"- Average Precision: `{metrics['average_precision']:.4f}`",
        "",
        "## Top 10 Features",
        "",
        *top10_lines,
        "",
        "## Output Files",
        "",
        f"- Model: `{MODEL_PATH.name}`",
        f"- Metrics: `{METRICS_PATH.name}`",
        f"- Classification report: `{CLASSIFICATION_REPORT_PATH.name}`",
        f"- Feature importance: `{FEATURE_IMPORTANCE_CSV.name}`",
        f"- Test predictions: `{PREDICTIONS_CSV.name}`",
        f"- Confusion matrix chart: `{CONFUSION_MATRIX_PNG.name}`",
        f"- ROC curve chart: `{ROC_CURVE_PNG.name}`",
        f"- Precision-recall chart: `{PR_CURVE_PNG.name}`",
        f"- Feature importance chart: `{FEATURE_IMPORTANCE_PNG.name}`",
    ]
    SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
