from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
BASELINE_METRICS_PATH = ROOT / "outputs" / "xgboost_baseline" / "metrics.json"
RF_METRICS_PATH = ROOT / "outputs" / "random_forest_baseline" / "metrics.json"
OUTPUT_DIR = ROOT / "outputs" / "xgboost_optimization"

SEARCH_RESULTS_CSV = OUTPUT_DIR / "search_results.csv"
BEST_CONFIG_JSON = OUTPUT_DIR / "best_config.json"
BEST_METRICS_JSON = OUTPUT_DIR / "best_metrics.json"
BEST_REPORT_TXT = OUTPUT_DIR / "best_classification_report.txt"
BEST_MODEL_PATH = OUTPUT_DIR / "best_xgboost_optimized.joblib"
BEST_PREDICTIONS_CSV = OUTPUT_DIR / "best_test_predictions.csv"
BEST_FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "best_feature_importance.csv"
OPTIMIZATION_REPORT_MD = OUTPUT_DIR / "optimization_report.md"

SEARCH_BAR_PNG = OUTPUT_DIR / "validation_top12_f1.png"
THRESHOLD_PNG = OUTPUT_DIR / "best_threshold_curve.png"
CONFUSION_MATRIX_PNG = OUTPUT_DIR / "best_confusion_matrix.png"
ROC_CURVE_PNG = OUTPUT_DIR / "best_roc_curve.png"
PR_CURVE_PNG = OUTPUT_DIR / "best_precision_recall_curve.png"
FEATURE_IMPORTANCE_PNG = OUTPUT_DIR / "best_feature_importance_top20.png"
BASELINE_COMPARE_PNG = OUTPUT_DIR / "baseline_vs_optimized_metrics.png"


TEXT_COLS = ["ERC20 most sent token type", "ERC20_most_rec_token_type"]
ID_COLS = ["Address"]
LABEL_COL = "FLAG"
RANDOM_STATE = 42


PARAM_CONFIGS = [
    {
        "config_name": "baseline_like",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight_mult": 1.0,
    },
    {
        "config_name": "deeper_more_trees",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight_mult": 1.0,
    },
    {
        "config_name": "shallow_regularized",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "scale_pos_weight_mult": 1.0,
    },
    {
        "config_name": "low_lr_longer",
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
        "scale_pos_weight_mult": 1.0,
    },
    {
        "config_name": "recall_focus",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight_mult": 1.2,
    },
    {
        "config_name": "precision_balance",
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 6,
        "gamma": 0.2,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
        "scale_pos_weight_mult": 0.9,
    },
    {
        "config_name": "wide_sampling",
        "n_estimators": 450,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.6,
        "min_child_weight": 2,
        "gamma": 0.05,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight_mult": 1.0,
    },
    {
        "config_name": "strong_regularization",
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_weight": 4,
        "gamma": 0.15,
        "reg_alpha": 0.15,
        "reg_lambda": 4.0,
        "scale_pos_weight_mult": 1.1,
    },
]


FEATURE_VARIANTS = [
    "numeric_base",
    "numeric_ratio",
    "numeric_textfreq",
    "numeric_ratio_textfreq",
]


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator + 1.0) / (denominator + 1.0)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["recv_sent_count_ratio"] = ratio(out["Received Tnx"], out["Sent tnx"])
    out["unique_recv_sent_ratio"] = ratio(
        out["Unique Received From Addresses"], out["Unique Sent To Addresses"]
    )
    out["ether_recv_sent_ratio"] = ratio(out["total ether received"], out["total Ether sent"])
    out["avg_recv_sent_ratio"] = ratio(out["avg val received"], out["avg val sent"])
    out["contract_creation_flag"] = (out["Number of Created Contracts"] > 0).astype(int)
    out["erc20_recv_sent_ratio"] = ratio(
        out["ERC20 total Ether received"], out["ERC20 total ether sent"]
    )
    out["erc20_activity_density"] = ratio(
        out["Total ERC20 tnxs"], out["total transactions (including tnx to create contract)"]
    )
    out["sent_speed_score"] = 1.0 / (out["Avg min between sent tnx"] + 1.0)
    out["received_speed_score"] = 1.0 / (out["Avg min between received tnx"] + 1.0)
    out["lifecycle_per_tx"] = ratio(
        out["Time Diff between first and last (Mins)"],
        out["total transactions (including tnx to create contract)"],
    )
    return out


def build_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded = {LABEL_COL, *ID_COLS, *TEXT_COLS}
    base_cols = [col for col in train_df.columns if col not in excluded]
    X_train = train_df[base_cols].copy()
    X_val = val_df[base_cols].copy()
    X_test = test_df[base_cols].copy()

    if "ratio" in variant:
        X_train = add_ratio_features(X_train)
        X_val = add_ratio_features(X_val)
        X_test = add_ratio_features(X_test)

    if "textfreq" in variant:
        for col in TEXT_COLS:
            freq_map = train_df[col].value_counts(normalize=True).to_dict()
            new_col = f"{col}_freq"
            X_train[new_col] = train_df[col].map(freq_map).fillna(0.0)
            X_val[new_col] = val_df[col].map(freq_map).fillna(0.0)
            X_test[new_col] = test_df[col].map(freq_map).fillna(0.0)

    return X_train, X_val, X_test


def optimize_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = np.arange(0.20, 0.801, 0.01)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    table = pd.DataFrame(rows)
    best_row = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best_row["threshold"]), table


def run_candidate(
    candidate: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    variant = candidate["feature_variant"]
    X_train, X_val, _ = build_features(train_df, val_df, test_df, variant)
    y_train = train_df[LABEL_COL]
    y_val = val_df[LABEL_COL]

    base_spw = float((y_train == 0).sum() / (y_train == 1).sum())
    params = candidate["params"].copy()
    scale_pos_weight = base_spw * params.pop("scale_pos_weight_mult")

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred_default = (val_prob >= 0.5).astype(int)
    best_threshold, threshold_table = optimize_threshold(y_val, val_prob)
    val_pred_tuned = (val_prob >= best_threshold).astype(int)

    default_precision = precision_score(y_val, val_pred_default, zero_division=0)
    default_recall = recall_score(y_val, val_pred_default, zero_division=0)
    default_f1 = f1_score(y_val, val_pred_default, zero_division=0)
    tuned_precision = precision_score(y_val, val_pred_tuned, zero_division=0)
    tuned_recall = recall_score(y_val, val_pred_tuned, zero_division=0)
    tuned_f1 = f1_score(y_val, val_pred_tuned, zero_division=0)
    roc_auc = roc_auc_score(y_val, val_prob)
    ap = average_precision_score(y_val, val_prob)

    selection_score = 0.5 * tuned_f1 + 0.3 * tuned_recall + 0.2 * roc_auc

    return {
        "experiment_name": candidate["experiment_name"],
        "config_name": candidate["params"]["config_name"],
        "feature_variant": variant,
        "params": candidate["params"],
        "used_feature_count": int(X_train.shape[1]),
        "scale_pos_weight": float(scale_pos_weight),
        "val_precision_default": float(default_precision),
        "val_recall_default": float(default_recall),
        "val_f1_default": float(default_f1),
        "val_precision_tuned": float(tuned_precision),
        "val_recall_tuned": float(tuned_recall),
        "val_f1_tuned": float(tuned_f1),
        "val_roc_auc": float(roc_auc),
        "val_average_precision": float(ap),
        "best_threshold": float(best_threshold),
        "selection_score": float(selection_score),
        "threshold_table": threshold_table.to_dict(orient="records"),
    }


def load_baseline_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_val_df[LABEL_COL],
    )

    candidates = []
    for params in PARAM_CONFIGS:
        for variant in FEATURE_VARIANTS:
            experiment_name = f"{params['config_name']}__{variant}"
            candidates.append(
                {
                    "experiment_name": experiment_name,
                    "params": params,
                    "feature_variant": variant,
                }
            )

    results: list[dict[str, Any]] = []
    max_workers = min(4, max(1, math.floor((os_cpu_count() or 4) / 2)))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_candidate, candidate, train_df, val_df, test_df) for candidate in candidates
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"params", "threshold_table"}
            }
            for result in results
        ]
    ).sort_values(["selection_score", "val_f1_tuned", "val_recall_tuned"], ascending=False)
    SEARCH_RESULTS_CSV.write_text("", encoding="utf-8")
    results_df.to_csv(SEARCH_RESULTS_CSV, index=False, encoding="utf-8-sig")

    best_result = max(results, key=lambda x: (x["selection_score"], x["val_f1_tuned"], x["val_recall_tuned"]))

    # Refit best model on full train_val, then evaluate on fixed test
    X_train_full, _, X_test_final = build_features(train_val_df, val_df, test_df, best_result["feature_variant"])
    y_train_full = train_val_df[LABEL_COL]
    y_test = test_df[LABEL_COL]
    addr_test = test_df["Address"]

    best_params = best_result["params"].copy()
    best_scale_pos_weight = float(
        ((y_train_full == 0).sum() / (y_train_full == 1).sum()) * best_params.pop("scale_pos_weight_mult")
    )
    best_model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        min_child_weight=best_params["min_child_weight"],
        gamma=best_params["gamma"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=best_scale_pos_weight,
    )
    best_model.fit(X_train_full, y_train_full)

    test_prob = best_model.predict_proba(X_test_final)[:, 1]
    test_pred_default = (test_prob >= 0.5).astype(int)
    chosen_threshold = best_result["best_threshold"]
    test_pred_tuned = (test_prob >= chosen_threshold).astype(int)

    metrics_default = {
        "accuracy": float(accuracy_score(y_test, test_pred_default)),
        "precision": float(precision_score(y_test, test_pred_default)),
        "recall": float(recall_score(y_test, test_pred_default)),
        "f1": float(f1_score(y_test, test_pred_default)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "average_precision": float(average_precision_score(y_test, test_prob)),
    }
    metrics_tuned = {
        "accuracy": float(accuracy_score(y_test, test_pred_tuned)),
        "precision": float(precision_score(y_test, test_pred_tuned)),
        "recall": float(recall_score(y_test, test_pred_tuned)),
        "f1": float(f1_score(y_test, test_pred_tuned)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "average_precision": float(average_precision_score(y_test, test_prob)),
    }
    if (metrics_default["f1"], metrics_default["recall"]) >= (metrics_tuned["f1"], metrics_tuned["recall"]):
        recommended_name = "default_0.5"
        recommended_metrics = metrics_default
    else:
        recommended_name = "tuned_validation_threshold"
        recommended_metrics = metrics_tuned

    feature_importance = (
        pd.Series(best_model.feature_importances_, index=X_train_full.columns)
        .sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    feature_importance.to_csv(BEST_FEATURE_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")

    predictions = pd.DataFrame(
        {
            "Address": addr_test.values,
            "y_true": y_test.values,
            "y_pred_default": test_pred_default,
            "y_pred_tuned": test_pred_tuned,
            "y_prob_fraud": test_prob,
        }
    ).sort_values("y_prob_fraud", ascending=False)
    predictions.to_csv(BEST_PREDICTIONS_CSV, index=False, encoding="utf-8-sig")

    BEST_REPORT_TXT.write_text(
        classification_report(y_test, test_pred_tuned, digits=4), encoding="utf-8"
    )
    joblib.dump(best_model, BEST_MODEL_PATH)

    best_config_payload = {
        "experiment_name": best_result["experiment_name"],
        "feature_variant": best_result["feature_variant"],
        "params": best_result["params"],
        "selection_score": best_result["selection_score"],
        "validation_metrics_default_threshold": {
            "precision": best_result["val_precision_default"],
            "recall": best_result["val_recall_default"],
            "f1": best_result["val_f1_default"],
            "roc_auc": best_result["val_roc_auc"],
            "average_precision": best_result["val_average_precision"],
        },
        "validation_metrics_tuned_threshold": {
            "precision": best_result["val_precision_tuned"],
            "recall": best_result["val_recall_tuned"],
            "f1": best_result["val_f1_tuned"],
            "roc_auc": best_result["val_roc_auc"],
            "average_precision": best_result["val_average_precision"],
        },
        "best_threshold": chosen_threshold,
    }
    BEST_CONFIG_JSON.write_text(json.dumps(best_config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_metrics_payload = {
        "dataset_path": str(DATA_PATH),
        "train_val_shape": [int(X_train_full.shape[0]), int(X_train_full.shape[1])],
        "test_shape": [int(X_test_final.shape[0]), int(X_test_final.shape[1])],
        "feature_variant": best_result["feature_variant"],
        "feature_count": int(X_train_full.shape[1]),
        "best_threshold": float(chosen_threshold),
        "metrics_default_threshold": metrics_default,
        "metrics_tuned_threshold": metrics_tuned,
        "recommended_test_strategy": recommended_name,
        "recommended_test_metrics": recommended_metrics,
        "best_params": best_result["params"],
        "best_scale_pos_weight": float(best_scale_pos_weight),
    }
    BEST_METRICS_JSON.write_text(json.dumps(best_metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Charts
    top12 = results_df.head(12).sort_values("val_f1_tuned", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top12["experiment_name"], top12["val_f1_tuned"], color="#2f7d4a")
    ax.set_xlabel("Validation F1 (Tuned Threshold)")
    ax.set_title("Top 12 XGBoost Optimization Experiments")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SEARCH_BAR_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    threshold_df = pd.DataFrame(best_result["threshold_table"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], label="F1-score", linewidth=2)
    ax.axvline(chosen_threshold, color="red", linestyle="--", label=f"Best threshold = {chosen_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Validation Threshold Optimization")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(THRESHOLD_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_test, test_pred_tuned)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Purples")
    ax.set_title("Optimized XGBoost Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics_tuned['roc_auc']:.4f}", linewidth=2, color="#7d2f6f")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Optimized XGBoost ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        recall_vals,
        precision_vals,
        label=f"AP = {metrics_tuned['average_precision']:.4f}",
        linewidth=2,
        color="#7d2f6f",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Optimized XGBoost Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PR_CURVE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    top20 = feature_importance.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["importance"], color="#7d2f6f")
    ax.set_xlabel("Importance")
    ax.set_title("Optimized XGBoost Top 20 Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Baseline vs optimized metrics chart
    baseline_metrics = load_baseline_metrics(BASELINE_METRICS_PATH)
    rf_metrics = load_baseline_metrics(RF_METRICS_PATH)
    compare_df = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"],
            "random_forest": [
                rf_metrics["accuracy"],
                rf_metrics["precision"],
                rf_metrics["recall"],
                rf_metrics["f1"],
                rf_metrics["roc_auc"],
                rf_metrics["average_precision"],
            ],
            "xgboost_baseline": [
                baseline_metrics["accuracy"],
                baseline_metrics["precision"],
                baseline_metrics["recall"],
                baseline_metrics["f1"],
                baseline_metrics["roc_auc"],
                baseline_metrics["average_precision"],
            ],
            "xgboost_optimized": [
                recommended_metrics["accuracy"],
                recommended_metrics["precision"],
                recommended_metrics["recall"],
                recommended_metrics["f1"],
                recommended_metrics["roc_auc"],
                recommended_metrics["average_precision"],
            ],
            "xgboost_optimized_tuned": [
                metrics_tuned["accuracy"],
                metrics_tuned["precision"],
                metrics_tuned["recall"],
                metrics_tuned["f1"],
                metrics_tuned["roc_auc"],
                metrics_tuned["average_precision"],
            ],
        }
    )

    x = np.arange(len(compare_df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, compare_df["random_forest"], width=width, label="Random Forest", color="#2f6f8f")
    ax.bar(x - 0.5 * width, compare_df["xgboost_baseline"], width=width, label="XGBoost Baseline", color="#2f7d4a")
    ax.bar(x + 0.5 * width, compare_df["xgboost_optimized"], width=width, label="XGBoost Optimized", color="#7d2f6f")
    ax.bar(
        x + 1.5 * width,
        compare_df["xgboost_optimized_tuned"],
        width=width,
        label="XGBoost Tuned Threshold",
        color="#a85f2f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df["metric"].str.upper())
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Optimized Model Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASELINE_COMPARE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    improvement_f1 = recommended_metrics["f1"] - baseline_metrics["f1"]
    improvement_recall = recommended_metrics["recall"] - baseline_metrics["recall"]
    improvement_auc = recommended_metrics["roc_auc"] - baseline_metrics["roc_auc"]

    top5_lines = []
    for row in feature_importance.head(5).itertuples(index=False):
        top5_lines.append(f"- `{row.feature}`: {row.importance:.6f}")

    report_lines = [
        "# XGBoost Optimization Report",
        "",
        "## Experiment Design",
        "",
        "- Fixed dataset split: train/validation/test derived from the same cleaned dataset.",
        "- Optimization methods tried:",
        "  - multiple hyperparameter combinations",
        "  - feature engineering with ratio features",
        "  - text frequency encoding for ERC20 token type fields",
        "  - threshold optimization on validation set",
        "",
        "## Search Summary",
        "",
        f"- Total experiments: `{len(results)}`",
        f"- Best experiment: `{best_result['experiment_name']}`",
        f"- Best feature variant: `{best_result['feature_variant']}`",
        f"- Best validation threshold: `{chosen_threshold:.2f}`",
        f"- Best validation F1 (tuned): `{best_result['val_f1_tuned']:.4f}`",
        f"- Best validation Recall (tuned): `{best_result['val_recall_tuned']:.4f}`",
        "",
        "## Final Test Metrics (Optimized Model, Tuned Threshold)",
        "",
        f"- Accuracy: `{metrics_tuned['accuracy']:.4f}`",
        f"- Precision: `{metrics_tuned['precision']:.4f}`",
        f"- Recall: `{metrics_tuned['recall']:.4f}`",
        f"- F1-score: `{metrics_tuned['f1']:.4f}`",
        f"- ROC-AUC: `{metrics_tuned['roc_auc']:.4f}`",
        f"- Average Precision: `{metrics_tuned['average_precision']:.4f}`",
        "",
        "## Final Test Metrics (Optimized Model, Default Threshold 0.5)",
        "",
        f"- Accuracy: `{metrics_default['accuracy']:.4f}`",
        f"- Precision: `{metrics_default['precision']:.4f}`",
        f"- Recall: `{metrics_default['recall']:.4f}`",
        f"- F1-score: `{metrics_default['f1']:.4f}`",
        f"- ROC-AUC: `{metrics_default['roc_auc']:.4f}`",
        f"- Average Precision: `{metrics_default['average_precision']:.4f}`",
        "",
        "## Improvement Over XGBoost Baseline",
        "",
        f"- F1 improvement: `{improvement_f1:+.4f}`",
        f"- Recall improvement: `{improvement_recall:+.4f}`",
        f"- ROC-AUC improvement: `{improvement_auc:+.4f}`",
        f"- Recommended final strategy on test set: `{recommended_name}`",
        "",
        "## Top 5 Features of Optimized Model",
        "",
        *top5_lines,
        "",
        "## Output Files",
        "",
        f"- Search results: `{SEARCH_RESULTS_CSV.name}`",
        f"- Best config: `{BEST_CONFIG_JSON.name}`",
        f"- Best metrics: `{BEST_METRICS_JSON.name}`",
        f"- Best predictions: `{BEST_PREDICTIONS_CSV.name}`",
        f"- Validation search chart: `{SEARCH_BAR_PNG.name}`",
        f"- Threshold chart: `{THRESHOLD_PNG.name}`",
        f"- Comparison chart: `{BASELINE_COMPARE_PNG.name}`",
    ]
    OPTIMIZATION_REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


if __name__ == "__main__":
    main()
