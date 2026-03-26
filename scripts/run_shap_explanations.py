from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from optimize_xgboost_experiments import (
    DATA_PATH,
    ID_COLS,
    LABEL_COL,
    RANDOM_STATE,
    TEXT_COLS,
    build_features,
)


ROOT = Path(__file__).resolve().parents[1]
BEST_MODEL_PATH = ROOT / "outputs" / "xgboost_optimization" / "best_xgboost_optimized.joblib"
BEST_METRICS_PATH = ROOT / "outputs" / "xgboost_optimization" / "best_metrics.json"
OUTPUT_DIR = ROOT / "outputs" / "shap_explanations"

GLOBAL_IMPORTANCE_CSV = OUTPUT_DIR / "global_shap_importance.csv"
GLOBAL_IMPORTANCE_PNG = OUTPUT_DIR / "global_shap_bar_top20.png"
GLOBAL_SUMMARY_PNG = OUTPUT_DIR / "global_shap_summary_top15.png"
CASE_SUMMARY_CSV = OUTPUT_DIR / "case_explanations.csv"
CASE_REPORT_MD = OUTPUT_DIR / "case_explanations.md"
GLOBAL_REPORT_MD = OUTPUT_DIR / "global_shap_report.md"


def normalize_series(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - min_val) / (max_val - min_val)


def determine_threshold(best_metrics: dict) -> float:
    if best_metrics.get("recommended_test_strategy") == "default_0.5":
        return 0.5
    return float(best_metrics["best_threshold"])


def select_case_indices(
    features: pd.DataFrame,
    probs: np.ndarray,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, int]:
    cases = features.copy()
    cases["y_true"] = y_true.values
    cases["y_pred"] = y_pred
    cases["prob"] = probs
    cases = cases[(cases["y_true"] == 1) & (cases["y_pred"] == 1)].copy()

    if cases.empty:
        raise ValueError("No correctly predicted fraud cases available for SHAP case study.")

    balance_abs = cases["total ether balance"].abs()
    phishing_score = (
        (1 - normalize_series(cases["Time Diff between first and last (Mins)"]))
        + (1 - normalize_series(cases["Avg min between sent tnx"]))
        + (1 - normalize_series(balance_abs))
    )
    ponzi_score = (
        normalize_series(cases["Unique Received From Addresses"])
        + normalize_series((cases["Received Tnx"] + 1) / (cases["Sent tnx"] + 1))
        + normalize_series(cases["avg val received"])
    )
    ico_score = (
        normalize_series(cases["Number of Created Contracts"])
        + normalize_series(cases["Total ERC20 tnxs"])
        + normalize_series(cases["ERC20 total Ether received"])
        + normalize_series(cases["has_erc20_activity"])
    )

    cases["phishing_score"] = phishing_score
    cases["ponzi_score"] = ponzi_score
    cases["ico_score"] = ico_score

    chosen: dict[str, int] = {}
    used_indices: set[int] = set()
    for label, score_col in [
        ("钓鱼诈骗与资金清洗倾向", "phishing_score"),
        ("庞氏骗局倾向", "ponzi_score"),
        ("发币骗局与跑路盘倾向", "ico_score"),
    ]:
        ranked = cases.sort_values([score_col, "prob"], ascending=[False, False])
        for idx in ranked.index:
            if idx not in used_indices:
                chosen[label] = idx
                used_indices.add(idx)
                break

    return chosen


def plot_local_explanation(
    shap_row: pd.Series,
    case_title: str,
    output_path: Path,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    top_positive = shap_row.sort_values(ascending=False).head(6)
    top_negative = shap_row.sort_values(ascending=True).head(4)
    selected = pd.concat([top_negative, top_positive]).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c44e52" if v < 0 else "#55a868" for v in selected.values]
    ax.barh(selected.index, selected.values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"Local SHAP Explanation: {case_title}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return list(top_positive.items()), list(top_negative.items())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_metrics = json.loads(BEST_METRICS_PATH.read_text(encoding="utf-8"))
    feature_variant = best_metrics["feature_variant"]
    threshold = determine_threshold(best_metrics)

    df = pd.read_csv(DATA_PATH)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )

    model = joblib.load(BEST_MODEL_PATH)

    # Build the same test features used in the optimized model
    X_train_full, _, X_test = build_features(train_val_df, test_df, test_df, feature_variant)
    X_test = X_test.reset_index(drop=True)
    y_test = test_df[LABEL_COL].reset_index(drop=True)
    addr_test = test_df["Address"].reset_index(drop=True)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    booster = model.get_booster()
    dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
    contribs = booster.predict(dtest, pred_contribs=True)

    feature_names = list(X_test.columns)
    shap_values = pd.DataFrame(contribs[:, :-1], columns=feature_names)
    base_values = contribs[:, -1]

    global_importance = (
        shap_values.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    global_importance.to_csv(GLOBAL_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")

    # Global bar chart
    top20 = global_importance.head(20).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["feature"], top20["mean_abs_shap"], color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global SHAP Importance Top 20")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GLOBAL_IMPORTANCE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Global summary scatter (approximate beeswarm)
    top15_features = global_importance.head(15)["feature"].tolist()[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    color_mappable = None
    for i, feature in enumerate(top15_features):
        feature_vals = X_test[feature]
        shap_vals = shap_values[feature]
        feature_norm = normalize_series(feature_vals)
        jitter = np.random.normal(loc=i, scale=0.10, size=len(shap_vals))
        color_mappable = ax.scatter(
            shap_vals,
            jitter,
            c=feature_norm,
            cmap="coolwarm",
            s=12,
            alpha=0.55,
            edgecolors="none",
        )
    ax.set_yticks(range(len(top15_features)))
    ax.set_yticklabels(top15_features)
    ax.set_xlabel("SHAP value")
    ax.set_title("Global SHAP Summary Top 15")
    ax.grid(axis="x", alpha=0.3)
    if color_mappable is not None:
        cbar = fig.colorbar(color_mappable, ax=ax)
        cbar.set_label("Normalized feature value")
    fig.tight_layout()
    fig.savefig(GLOBAL_SUMMARY_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Global report
    global_lines = [
        "# Global SHAP Report",
        "",
        f"- Model feature variant: `{feature_variant}`",
        f"- Recommended threshold used for case selection: `{threshold:.2f}`",
        "",
        "## Top 15 Global Features",
        "",
    ]
    for row in global_importance.head(15).itertuples(index=False):
        global_lines.append(f"- `{row.feature}`: {row.mean_abs_shap:.6f}")
    GLOBAL_REPORT_MD.write_text("\n".join(global_lines) + "\n", encoding="utf-8")

    # Local cases
    case_indices = select_case_indices(X_test.copy(), test_prob, y_test, test_pred)
    case_rows = []
    case_lines = [
        "# SHAP 单地址案例解释",
        "",
        f"- 使用模型：`best_xgboost_optimized.joblib`",
        f"- 采用阈值：`{threshold:.2f}`",
        "",
    ]

    case_slug_map = {
        "钓鱼诈骗与资金清洗倾向": "phishing_money_laundering",
        "庞氏骗局倾向": "ponzi_scheme",
        "发币骗局与跑路盘倾向": "ico_rugpull",
    }

    for case_label, idx in case_indices.items():
        address = addr_test.loc[idx]
        prob = float(test_prob[idx])
        true_label = int(y_test.loc[idx])
        pred_label = int(test_pred[idx])
        shap_row = shap_values.loc[idx]
        base_value = float(base_values[idx])
        case_slug = case_slug_map[case_label]
        case_chart_path = OUTPUT_DIR / f"case_{idx}_{case_slug}.png"
        top_positive, top_negative = plot_local_explanation(shap_row, case_slug, case_chart_path)

        top_positive_str = "; ".join([f"{feat}={val:.4f}" for feat, val in top_positive])
        top_negative_str = "; ".join([f"{feat}={val:.4f}" for feat, val in top_negative])

        case_rows.append(
            {
                "case_label": case_label,
                "address": address,
                "y_true": true_label,
                "y_pred": pred_label,
                "fraud_probability": prob,
                "base_value": base_value,
                "top_positive_features": top_positive_str,
                "top_negative_features": top_negative_str,
                "chart_path": str(case_chart_path),
            }
        )

        case_lines.extend(
            [
                f"## {case_label}",
                "",
                f"- 地址：`{address}`",
                f"- 真实标签 `FLAG`：`{true_label}`",
                f"- 模型预测标签：`{pred_label}`",
                f"- 欺诈概率：`{prob:.4f}`",
                f"- 主要正向推动特征：`{top_positive_str}`",
                f"- 主要负向拉回特征：`{top_negative_str}`",
                f"- 对应图表：`{case_chart_path.name}`",
                "",
            ]
        )

    pd.DataFrame(case_rows).to_csv(CASE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    CASE_REPORT_MD.write_text("\n".join(case_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
