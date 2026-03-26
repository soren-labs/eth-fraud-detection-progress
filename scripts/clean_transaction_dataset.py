from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "transaction_dataset.csv"
OUTPUT_DIR = ROOT / "processed"
CLEANED_CSV = OUTPUT_DIR / "transaction_dataset_cleaned.csv"
FIELD_DICT_CSV = OUTPUT_DIR / "transaction_dataset_field_dictionary.csv"
SUMMARY_JSON = OUTPUT_DIR / "transaction_dataset_cleaning_summary.json"
REPORT_MD = OUTPUT_DIR / "transaction_dataset_cleaning_report.md"


def standardize_column_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned == "total transactions (including tnx to create contract":
        return "total transactions (including tnx to create contract)"
    return cleaned


def normalize_token_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value == "0":
        return None
    return value


def choose_numeric_value(series: pd.Series) -> Any:
    values = pd.Series(series).dropna().unique()
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return values[0]
    return float(np.median(pd.to_numeric(values, errors="coerce")))


def choose_text_value(series: pd.Series) -> str:
    values = [v for v in pd.Series(series).tolist() if v not in (None, "", "NoTokenInfo")]
    unique_values = sorted(set(values))
    if not unique_values:
        return "NoTokenInfo"
    if len(unique_values) == 1:
        return unique_values[0]
    return "AmbiguousTokenType"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(SOURCE_CSV)
    original_rows, original_cols = raw_df.shape

    original_columns = list(raw_df.columns)
    standardized_columns = [standardize_column_name(col) for col in original_columns]
    rename_map = dict(zip(original_columns, standardized_columns))
    df = raw_df.rename(columns=rename_map).copy()

    stripped_collision_count = len(standardized_columns) - len(set(standardized_columns))
    if stripped_collision_count:
        raise ValueError("Column rename collision detected after standardization.")

    id_drop_cols = {"Unnamed: 0", "Index"}
    ambiguous_drop_cols = {"ERC20 uniq sent addr.1"}
    token_text_cols = ["ERC20 most sent token type", "ERC20_most_rec_token_type"]

    for col in token_text_cols:
        if col in df.columns:
            df[col] = df[col].map(normalize_token_value)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col not in {"FLAG"} | id_drop_cols]

    duplicate_address_rows = int(df["Address"].duplicated(keep=False).sum())
    duplicate_address_count = int(df.loc[df["Address"].duplicated(keep=False), "Address"].nunique())

    text_conflict_addresses: list[str] = []
    numeric_conflict_addresses: list[str] = []
    merged_rows: list[dict[str, Any]] = []

    for address, group in df.groupby("Address", sort=False):
        merged: dict[str, Any] = {"Address": address}

        flag_values = group["FLAG"].dropna().unique().tolist()
        if len(flag_values) == 0:
            merged["FLAG"] = np.nan
        elif len(flag_values) == 1:
            merged["FLAG"] = int(flag_values[0])
        else:
            merged["FLAG"] = int(max(flag_values))
            numeric_conflict_addresses.append(address)

        for col in df.columns:
            if col in {"Address", "FLAG"}:
                continue

            if col in numeric_feature_cols:
                values = group[col].dropna().unique()
                if len(values) > 1:
                    numeric_conflict_addresses.append(address)
                merged[col] = choose_numeric_value(group[col])
            elif col in token_text_cols:
                chosen = choose_text_value(group[col])
                non_null_values = sorted(set(v for v in group[col].tolist() if v))
                if len(non_null_values) > 1:
                    text_conflict_addresses.append(address)
                merged[col] = chosen
            else:
                non_null_values = [v for v in group[col].tolist() if pd.notna(v)]
                merged[col] = non_null_values[0] if non_null_values else ""

        merged_rows.append(merged)

    dedup_df = pd.DataFrame(merged_rows)
    rows_after_dedup = len(dedup_df)

    original_dtypes = df.dtypes.to_dict()
    for col, dtype in original_dtypes.items():
        if col not in dedup_df.columns:
            continue
        if str(dtype) == "int64":
            numeric_series = pd.to_numeric(dedup_df[col], errors="coerce")
            non_null = numeric_series.dropna()
            if non_null.empty or np.allclose(non_null, np.round(non_null)):
                dedup_df[col] = np.round(numeric_series).astype("Int64")
            else:
                dedup_df[col] = numeric_series
        elif str(dtype) == "float64":
            dedup_df[col] = pd.to_numeric(dedup_df[col], errors="coerce")

    erc20_numeric_cols = [
        col
        for col in dedup_df.columns
        if col == "Total ERC20 tnxs" or col.startswith("ERC20 ")
    ]
    erc20_numeric_cols = [
        col for col in erc20_numeric_cols if pd.api.types.is_numeric_dtype(dedup_df[col])
    ]

    erc20_missing_before_fill = {
        col: int(dedup_df[col].isna().sum()) for col in erc20_numeric_cols if dedup_df[col].isna().sum() > 0
    }
    for col in erc20_numeric_cols:
        dedup_df[col] = dedup_df[col].fillna(0)

    for col in token_text_cols:
        if col in dedup_df.columns:
            dedup_df[col] = dedup_df[col].fillna("NoTokenInfo")
            dedup_df[col] = dedup_df[col].replace("", "NoTokenInfo")

    if "Total ERC20 tnxs" in dedup_df.columns:
        dedup_df["has_erc20_activity"] = (
            (dedup_df["Total ERC20 tnxs"] > 0)
            | (dedup_df.get("ERC20 total Ether received", 0) > 0)
            | (dedup_df.get("ERC20 total ether sent", 0) > 0)
            | (dedup_df.get("ERC20 most sent token type", "NoTokenInfo") != "NoTokenInfo")
            | (dedup_df.get("ERC20_most_rec_token_type", "NoTokenInfo") != "NoTokenInfo")
        ).astype(int)
    else:
        dedup_df["has_erc20_activity"] = 0

    constant_numeric_cols = [
        col
        for col in dedup_df.select_dtypes(include=["number"]).columns
        if col not in {"FLAG", "has_erc20_activity"} and dedup_df[col].nunique(dropna=False) <= 1
    ]

    dropped_columns: dict[str, str] = {}
    for col in sorted(id_drop_cols):
        if col in dedup_df.columns:
            dropped_columns[col] = "identifier column"
    for col in sorted(ambiguous_drop_cols):
        if col in dedup_df.columns:
            dropped_columns[col] = "ambiguous field name and very low information"
    for col in constant_numeric_cols:
        dropped_columns[col] = "constant numeric column"

    cols_to_drop = list(dropped_columns.keys())
    reported_erc20_fill_counts = {
        col: count for col, count in erc20_missing_before_fill.items() if col not in dropped_columns
    }
    cleaned_df = dedup_df.drop(columns=cols_to_drop, errors="ignore").copy()

    final_int_cols = ["FLAG", "has_erc20_activity"]
    for col in final_int_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce").astype("int64")

    first_cols = [col for col in ["Address", "FLAG", "has_erc20_activity"] if col in cleaned_df.columns]
    other_cols = [col for col in cleaned_df.columns if col not in first_cols]
    cleaned_df = cleaned_df[first_cols + other_cols]

    cleaned_df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")

    field_dict_rows: list[dict[str, str]] = []
    for raw_name in original_columns:
        standard_name = standardize_column_name(raw_name)
        final_name = standard_name if standard_name in cleaned_df.columns else ""
        notes: list[str] = []
        action = "kept"

        if raw_name != standard_name:
            notes.append("trimmed column name")
        if standard_name == "total transactions (including tnx to create contract)":
            notes.append("fixed truncated column name")
        if standard_name in id_drop_cols:
            action = "dropped"
            notes.append("identifier column")
        elif standard_name in ambiguous_drop_cols:
            action = "dropped"
            notes.append("ambiguous field name and very low information")
        elif standard_name in constant_numeric_cols:
            action = "dropped"
            notes.append("constant numeric column")
        else:
            if standard_name in token_text_cols:
                notes.append("normalized token text missing values")
            if standard_name in erc20_missing_before_fill:
                notes.append("filled structural ERC20 missing values with 0")

        field_dict_rows.append(
            {
                "raw_name": raw_name,
                "standard_name": standard_name,
                "final_name": final_name,
                "action": action,
                "notes": "; ".join(notes),
            }
        )

    field_dict_rows.append(
        {
            "raw_name": "",
            "standard_name": "has_erc20_activity",
            "final_name": "has_erc20_activity",
            "action": "added",
            "notes": "derived indicator after ERC20 cleaning",
        }
    )
    pd.DataFrame(field_dict_rows).to_csv(FIELD_DICT_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "source_csv": str(SOURCE_CSV),
        "output_csv": str(CLEANED_CSV),
        "original_rows": int(original_rows),
        "original_cols": int(original_cols),
        "rows_after_dedup": int(rows_after_dedup),
        "cleaned_rows": int(cleaned_df.shape[0]),
        "cleaned_cols": int(cleaned_df.shape[1]),
        "removed_rows": int(original_rows - cleaned_df.shape[0]),
        "removed_columns": dropped_columns,
        "duplicate_address_rows": duplicate_address_rows,
        "duplicate_address_count": duplicate_address_count,
        "text_conflict_addresses": sorted(set(text_conflict_addresses)),
        "numeric_conflict_addresses": sorted(set(numeric_conflict_addresses)),
        "erc20_missing_before_fill": reported_erc20_fill_counts,
        "erc20_token_no_info_count": {
            col: int((cleaned_df[col] == "NoTokenInfo").sum()) for col in token_text_cols if col in cleaned_df.columns
        },
        "has_erc20_activity_count": int(cleaned_df["has_erc20_activity"].sum()),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Transaction Dataset Cleaning Report",
        "",
        "## Overview",
        "",
        f"- Source file: `{SOURCE_CSV.name}`",
        f"- Original shape: `{original_rows} x {original_cols}`",
        f"- Cleaned shape: `{cleaned_df.shape[0]} x {cleaned_df.shape[1]}`",
        f"- Removed rows during address deduplication: `{original_rows - cleaned_df.shape[0]}`",
        f"- Removed columns: `{len(dropped_columns)}`",
        "",
        "## Applied Cleaning Rules",
        "",
        "- Standardized column names by trimming whitespace.",
        "- Fixed the truncated transaction count column name.",
        "- Dropped identifier columns `Unnamed: 0` and `Index`.",
        "- Merged duplicate addresses into one record per address.",
        "- Normalized token text missing values to `NoTokenInfo`.",
        "- Resolved token text conflicts across duplicate addresses with `AmbiguousTokenType`.",
        "- Filled structural ERC20 numeric missing values with `0`.",
        "- Added `has_erc20_activity` as a derived indicator.",
        "- Dropped the ambiguous column `ERC20 uniq sent addr.1`.",
        "- Dropped constant numeric columns.",
        "",
        "## Duplicate Address Handling",
        "",
        f"- Duplicate address rows in source: `{duplicate_address_rows}`",
        f"- Duplicate address count in source: `{duplicate_address_count}`",
        f"- Duplicate addresses with token text conflicts: `{len(sorted(set(text_conflict_addresses)))}`",
        f"- Duplicate addresses with numeric conflicts: `{len(sorted(set(numeric_conflict_addresses)))}`",
        "",
        "## Dropped Columns",
        "",
    ]

    for col, reason in dropped_columns.items():
        report_lines.append(f"- `{col}`: {reason}")

    report_lines.extend(
        [
            "",
            "## ERC20 Missing Value Fill",
            "",
        ]
    )
    for col, count in reported_erc20_fill_counts.items():
        report_lines.append(f"- `{col}`: filled `{count}` missing values with `0`")

    report_lines.extend(
        [
            "",
            "## Token Text Normalization",
            "",
        ]
    )
    for col in token_text_cols:
        if col in cleaned_df.columns:
            report_lines.append(
                f"- `{col}`: `NoTokenInfo` count = `{int((cleaned_df[col] == 'NoTokenInfo').sum())}`"
            )

    report_lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Cleaned dataset: `{CLEANED_CSV.name}`",
            f"- Field dictionary: `{FIELD_DICT_CSV.name}`",
            f"- Summary JSON: `{SUMMARY_JSON.name}`",
        ]
    )

    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
