from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BEST_METRICS_PATH = ROOT / "outputs" / "xgboost_optimization" / "best_metrics.json"
MODEL_COMPARE_PATH = ROOT / "outputs" / "model_comparison" / "model_metrics_comparison.csv"
GLOBAL_SHAP_PATH = ROOT / "outputs" / "shap_explanations" / "global_shap_importance.csv"
CASE_EXPLANATIONS_PATH = ROOT / "outputs" / "shap_explanations" / "case_explanations.csv"

OUTPUT_DIR = ROOT / "outputs" / "llm_explanations"
GLOBAL_JSON_PATH = OUTPUT_DIR / "global_summary.json"
GLOBAL_MD_PATH = OUTPUT_DIR / "global_summary.md"
CASE_JSON_PATH = OUTPUT_DIR / "case_reports.json"
CASE_CSV_PATH = OUTPUT_DIR / "case_reports.csv"
CASE_MD_PATH = OUTPUT_DIR / "case_reports.md"
FINAL_MD_PATH = OUTPUT_DIR / "LLM结果汇总.md"

API_URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(
    api_key: str,
    messages: list[dict[str, str]],
    model: str = "deepseek-chat",
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def format_bullets(items: list[str]) -> str:
    return "\n".join([f"- {item}" for item in items])


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_metrics = json.loads(BEST_METRICS_PATH.read_text(encoding="utf-8"))
    compare_df = pd.read_csv(MODEL_COMPARE_PATH)
    global_shap = pd.read_csv(GLOBAL_SHAP_PATH).head(10)
    case_df = pd.read_csv(CASE_EXPLANATIONS_PATH)

    metric_map = {row["metric"]: row for _, row in compare_df.iterrows()}
    top_features_text = "\n".join(
        [f"- {row.feature}: {row.mean_abs_shap:.6f}" for row in global_shap.itertuples(index=False)]
    )

    global_system = (
        "You are an academic writing assistant for an undergraduate cybersecurity thesis. "
        "Your task is to summarize model results faithfully. "
        "Do not invent unsupported conclusions. "
        "Output valid JSON only."
    )
    global_user = f"""
Project topic: interpretable Ethereum fraud detection.
Main task: binary fraud risk detection, followed by fraud type tendency analysis.
Best optimized XGBoost metrics:
- Accuracy: {best_metrics['recommended_test_metrics']['accuracy']:.4f}
- Precision: {best_metrics['recommended_test_metrics']['precision']:.4f}
- Recall: {best_metrics['recommended_test_metrics']['recall']:.4f}
- F1-score: {best_metrics['recommended_test_metrics']['f1']:.4f}
- ROC-AUC: {best_metrics['recommended_test_metrics']['roc_auc']:.4f}
- Average Precision: {best_metrics['recommended_test_metrics']['average_precision']:.4f}

Model comparison summary:
- Random Forest F1: {metric_map['f1']['random_forest']:.4f}
- XGBoost baseline F1: {metric_map['f1']['xgboost']:.4f}
- XGBoost optimized F1: {best_metrics['recommended_test_metrics']['f1']:.4f}

Top global SHAP features:
{top_features_text}

Please output JSON with the following keys:
- research_progress_summary: one concise paragraph in Chinese
- model_result_summary: one concise paragraph in Chinese
- interpretability_summary: one concise paragraph in Chinese
- advisor_brief_points: array of 4 short Chinese bullet points suitable for progress reporting
"""
    global_text = call_deepseek(
        api_key=api_key,
        messages=[
            {"role": "system", "content": global_system},
            {"role": "user", "content": global_user},
        ],
    )
    global_json = extract_json_block(global_text)
    GLOBAL_JSON_PATH.write_text(json.dumps(global_json, indent=2, ensure_ascii=False), encoding="utf-8")

    global_md = [
        "# LLM Global Summary",
        "",
        "## Research Progress Summary",
        "",
        global_json["research_progress_summary"],
        "",
        "## Model Result Summary",
        "",
        global_json["model_result_summary"],
        "",
        "## Interpretability Summary",
        "",
        global_json["interpretability_summary"],
        "",
        "## Advisor Brief Points",
        "",
        *[f"- {item}" for item in global_json["advisor_brief_points"]],
    ]
    GLOBAL_MD_PATH.write_text("\n".join(global_md) + "\n", encoding="utf-8")

    case_results: list[dict[str, Any]] = []
    case_md_lines = [
        "# LLM Address Risk Explanations",
        "",
        "These explanations are generated from the optimized XGBoost prediction results and SHAP feature contributions.",
        "",
    ]

    for row in case_df.itertuples(index=False):
        tendency = row.case_label
        address = row.address
        prob = float(row.fraud_probability)
        top_positive = row.top_positive_features
        top_negative = row.top_negative_features

        case_system = (
            "You are an academic writing assistant. "
            "Generate a faithful Chinese explanation for an Ethereum address risk case. "
            "Do not invent facts beyond the provided model output and SHAP features. "
            "Output valid JSON only."
        )
        case_user = f"""
Address: {address}
Predicted fraud probability: {prob:.4f}
Fraud tendency label: {tendency}
Top positive SHAP-driving features: {top_positive}
Top negative SHAP-driving features: {top_negative}

Please output JSON with these keys:
- risk_summary: 2 to 3 Chinese sentences explaining why the address is high risk
- tendency_explanation: 2 Chinese sentences explaining why it is closer to this fraud tendency
- thesis_paragraph: 1 polished Chinese paragraph suitable for the thesis case analysis section
- defense_points: array of 3 short Chinese points suitable for oral defense
"""
        case_text = call_deepseek(
            api_key=api_key,
            messages=[
                {"role": "system", "content": case_system},
                {"role": "user", "content": case_user},
            ],
            max_tokens=1000,
        )
        case_json = extract_json_block(case_text)
        case_json.update(
            {
                "case_label": tendency,
                "address": address,
                "fraud_probability": prob,
                "top_positive_features": top_positive,
                "top_negative_features": top_negative,
            }
        )
        case_results.append(case_json)

        case_md_lines.extend(
            [
                f"## {tendency}",
                "",
                f"- 地址：`{address}`",
                f"- 欺诈概率：`{prob:.4f}`",
                f"- 主要正向特征：`{top_positive}`",
                f"- 主要负向特征：`{top_negative}`",
                "",
                "### 风险解释",
                "",
                case_json["risk_summary"],
                "",
                "### 类型倾向解释",
                "",
                case_json["tendency_explanation"],
                "",
                "### 论文可用段落",
                "",
                case_json["thesis_paragraph"],
                "",
                "### 答辩要点",
                "",
                *[f"- {item}" for item in case_json["defense_points"]],
                "",
            ]
        )

    CASE_JSON_PATH.write_text(json.dumps(case_results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(case_results).to_csv(CASE_CSV_PATH, index=False, encoding="utf-8-sig")
    CASE_MD_PATH.write_text("\n".join(case_md_lines) + "\n", encoding="utf-8")

    final_md = [
        "# LLM 结果汇总",
        "",
        "## 当前阶段说明",
        "",
        "本部分基于优化后的 XGBoost 结果与 SHAP 特征贡献，调用 DeepSeek 官方 API 生成自然语言风险说明，补全了毕业设计中的 LLM 辅助解释环节。",
        "",
        "## 全局总结",
        "",
        global_json["research_progress_summary"],
        "",
        global_json["model_result_summary"],
        "",
        global_json["interpretability_summary"],
        "",
        "## 代表性地址案例",
        "",
    ]
    for item in case_results:
        final_md.extend(
            [
                f"### {item['case_label']}",
                "",
                f"- 地址：`{item['address']}`",
                f"- 欺诈概率：`{item['fraud_probability']:.4f}`",
                f"- 风险解释：{item['risk_summary']}",
                f"- 类型倾向解释：{item['tendency_explanation']}",
                "",
            ]
        )
    final_md.extend(
        [
            "## 生成文件",
            "",
            f"- `{GLOBAL_JSON_PATH.name}`",
            f"- `{GLOBAL_MD_PATH.name}`",
            f"- `{CASE_JSON_PATH.name}`",
            f"- `{CASE_CSV_PATH.name}`",
            f"- `{CASE_MD_PATH.name}`",
        ]
    )
    FINAL_MD_PATH.write_text("\n".join(final_md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
