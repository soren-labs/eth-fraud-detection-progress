from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
RF_DIR = ROOT / "outputs" / "random_forest_baseline"
XGB_DIR = ROOT / "outputs" / "xgboost_baseline"
OUTPUT_DIR = ROOT / "outputs" / "model_comparison"

COMPARISON_CSV = OUTPUT_DIR / "model_metrics_comparison.csv"
COMPARISON_MD = OUTPUT_DIR / "model_comparison_report.md"
COMPARISON_MD_ZH = OUTPUT_DIR / "模型对比结果汇总.md"
METRICS_BAR_PNG = OUTPUT_DIR / "metrics_comparison_bar.png"
ROC_COMPARE_PNG = OUTPUT_DIR / "roc_curve_comparison.png"
PR_COMPARE_PNG = OUTPUT_DIR / "precision_recall_curve_comparison.png"


def load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rf_metrics = load_metrics(RF_DIR / "metrics.json")
    xgb_metrics = load_metrics(XGB_DIR / "metrics.json")

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    comparison_df = pd.DataFrame(
        {
            "metric": metric_names,
            "random_forest": [rf_metrics[m] for m in metric_names],
            "xgboost": [xgb_metrics[m] for m in metric_names],
        }
    )
    comparison_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

    # Metrics bar chart
    plot_df = comparison_df.copy()
    plot_df["metric"] = plot_df["metric"].str.upper()
    x = range(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], plot_df["random_forest"], width=width, label="Random Forest", color="#2f6f8f")
    ax.bar([i + width / 2 for i in x], plot_df["xgboost"], width=width, label="XGBoost", color="#2f7d4a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["metric"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(METRICS_BAR_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ROC comparison
    rf_pred = pd.read_csv(RF_DIR / "test_predictions.csv")
    xgb_pred = pd.read_csv(XGB_DIR / "test_predictions.csv")

    rf_y_true = rf_pred["y_true"]
    rf_y_prob = rf_pred["y_prob_fraud"]
    xgb_y_true = xgb_pred["y_true"]
    xgb_y_prob = xgb_pred["y_prob_fraud"]

    rf_fpr, rf_tpr, _ = roc_curve(rf_y_true, rf_y_prob)
    xgb_fpr, xgb_tpr, _ = roc_curve(xgb_y_true, xgb_y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rf_fpr, rf_tpr, label=f"Random Forest AUC = {roc_auc_score(rf_y_true, rf_y_prob):.4f}", linewidth=2)
    ax.plot(xgb_fpr, xgb_tpr, label=f"XGBoost AUC = {roc_auc_score(xgb_y_true, xgb_y_prob):.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_COMPARE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # PR comparison
    rf_precision, rf_recall, _ = precision_recall_curve(rf_y_true, rf_y_prob)
    xgb_precision, xgb_recall, _ = precision_recall_curve(xgb_y_true, xgb_y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        rf_recall,
        rf_precision,
        label=f"Random Forest AP = {average_precision_score(rf_y_true, rf_y_prob):.4f}",
        linewidth=2,
    )
    ax.plot(
        xgb_recall,
        xgb_precision,
        label=f"XGBoost AP = {average_precision_score(xgb_y_true, xgb_y_prob):.4f}",
        linewidth=2,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PR_COMPARE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Markdown summary
    better = {}
    for metric in metric_names:
        rf_val = rf_metrics[metric]
        xgb_val = xgb_metrics[metric]
        if xgb_val > rf_val:
            better[metric] = "XGBoost"
        elif rf_val > xgb_val:
            better[metric] = "Random Forest"
        else:
            better[metric] = "Tie"

    comparison_lines = [
        "# Model Comparison Report",
        "",
        "## Metrics Table",
        "",
        "| Metric | Random Forest | XGBoost | Better Model |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in comparison_df.itertuples(index=False):
        comparison_lines.append(
            f"| {row.metric} | {row.random_forest:.4f} | {row.xgboost:.4f} | {better[row.metric]} |"
        )

    comparison_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- On `accuracy`, better model: `{better['accuracy']}`.",
            f"- On `precision`, better model: `{better['precision']}`.",
            f"- On `recall`, better model: `{better['recall']}`.",
            f"- On `f1`, better model: `{better['f1']}`.",
            f"- On `roc_auc`, better model: `{better['roc_auc']}`.",
            f"- On `average_precision`, better model: `{better['average_precision']}`.",
            "",
            "## Output Files",
            "",
            f"- Metrics comparison CSV: `{COMPARISON_CSV.name}`",
            f"- Metrics bar chart: `{METRICS_BAR_PNG.name}`",
            f"- ROC comparison chart: `{ROC_COMPARE_PNG.name}`",
            f"- PR comparison chart: `{PR_COMPARE_PNG.name}`",
        ]
    )
    COMPARISON_MD.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")

    zh_lines = [
        "# 模型对比结果汇总",
        "",
        "## 对比设置",
        "",
        "- 数据集：`transaction_dataset_cleaned.csv`",
        "- 训练集 / 测试集划分：`80% / 20%`",
        "- 特征范围：`38 个数值字段`",
        "- 未纳入基线训练的字段：`Address`、`ERC20 most sent token type`、`ERC20_most_rec_token_type`",
        "- 对比模型：`Random Forest` 与 `XGBoost`",
        "",
        "## 核心指标对比",
        "",
        "| 指标 | 随机森林 | XGBoost | 表现更优的模型 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in comparison_df.itertuples(index=False):
        zh_metric = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "roc_auc": "ROC-AUC",
            "average_precision": "Average Precision",
        }[row.metric]
        better_label = {
            "Random Forest": "随机森林",
            "XGBoost": "XGBoost",
            "Tie": "相同",
        }[better[row.metric]]
        zh_lines.append(
            f"| {zh_metric} | {row.random_forest:.4f} | {row.xgboost:.4f} | {better_label} |"
        )

    zh_lines.extend(
        [
            "",
            "## 结果解读",
            "",
            f"- 从 `Accuracy` 看，随机森林略高（{rf_metrics['accuracy']:.4f} vs {xgb_metrics['accuracy']:.4f}）。",
            f"- 从 `Precision` 看，随机森林更高（{rf_metrics['precision']:.4f} vs {xgb_metrics['precision']:.4f}），说明它把地址判成欺诈时更保守。",
            f"- 从 `Recall` 看，XGBoost 明显更高（{xgb_metrics['recall']:.4f} vs {rf_metrics['recall']:.4f}），说明它能抓到更多真实欺诈地址。",
            f"- 从 `F1-score` 看，XGBoost 更高（{xgb_metrics['f1']:.4f} vs {rf_metrics['f1']:.4f}），说明其在精确率与召回率之间的综合平衡更好。",
            f"- 从 `ROC-AUC` 和 `Average Precision` 看，XGBoost 也更优，说明其整体排序能力和风险区分能力更强。",
            "",
            "## 结论",
            "",
            "- 如果更看重“判成欺诈时尽量少误报”，随机森林更有优势，因为它的 Precision 更高。",
            "- 如果更看重“尽量多识别真实欺诈地址”，XGBoost 更适合作为主模型，因为它的 Recall、F1 和 AUC 都更好。",
            "- 对于当前毕业设计场景，欺诈检测通常更重视 `Recall`、`F1-score` 和 `AUC`，因此更推荐将 `XGBoost` 作为后续主模型，随机森林作为基线对照模型。",
            "",
            "## 推荐在论文中的表述",
            "",
            "> 对比实验表明，随机森林在 Accuracy 与 Precision 上略占优势，而 XGBoost 在 Recall、F1-score、ROC-AUC 以及 Average Precision 等指标上整体更优。考虑到欺诈检测任务更强调对高风险样本的识别能力，本文最终选择 XGBoost 作为主模型，随机森林作为基线对照模型。",
            "",
            "## 对应图表",
            "",
            f"- 指标对比柱状图：`{METRICS_BAR_PNG.name}`",
            f"- ROC 对比图：`{ROC_COMPARE_PNG.name}`",
            f"- PR 对比图：`{PR_COMPARE_PNG.name}`",
        ]
    )
    COMPARISON_MD_ZH.write_text("\n".join(zh_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
