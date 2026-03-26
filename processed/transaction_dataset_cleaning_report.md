# Transaction Dataset Cleaning Report

## Overview

- Source file: `transaction_dataset.csv`
- Original shape: `9841 x 51`
- Cleaned shape: `9816 x 42`
- Removed rows during address deduplication: `25`
- Removed columns: `10`

## Applied Cleaning Rules

- Standardized column names by trimming whitespace.
- Fixed the truncated transaction count column name.
- Dropped identifier columns `Unnamed: 0` and `Index`.
- Merged duplicate addresses into one record per address.
- Normalized token text missing values to `NoTokenInfo`.
- Resolved token text conflicts across duplicate addresses with `AmbiguousTokenType`.
- Filled structural ERC20 numeric missing values with `0`.
- Added `has_erc20_activity` as a derived indicator.
- Dropped the ambiguous column `ERC20 uniq sent addr.1`.
- Dropped constant numeric columns.

## Duplicate Address Handling

- Duplicate address rows in source: `50`
- Duplicate address count in source: `25`
- Duplicate addresses with token text conflicts: `7`
- Duplicate addresses with numeric conflicts: `0`

## Dropped Columns

- `Index`: identifier column
- `Unnamed: 0`: identifier column
- `ERC20 uniq sent addr.1`: ambiguous field name and very low information
- `ERC20 avg time between sent tnx`: constant numeric column
- `ERC20 avg time between rec tnx`: constant numeric column
- `ERC20 avg time between rec 2 tnx`: constant numeric column
- `ERC20 avg time between contract tnx`: constant numeric column
- `ERC20 min val sent contract`: constant numeric column
- `ERC20 max val sent contract`: constant numeric column
- `ERC20 avg val sent contract`: constant numeric column

## ERC20 Missing Value Fill

- `Total ERC20 tnxs`: filled `829` missing values with `0`
- `ERC20 total Ether received`: filled `829` missing values with `0`
- `ERC20 total ether sent`: filled `829` missing values with `0`
- `ERC20 total Ether sent contract`: filled `829` missing values with `0`
- `ERC20 uniq sent addr`: filled `829` missing values with `0`
- `ERC20 uniq rec addr`: filled `829` missing values with `0`
- `ERC20 uniq rec contract addr`: filled `829` missing values with `0`
- `ERC20 min val rec`: filled `829` missing values with `0`
- `ERC20 max val rec`: filled `829` missing values with `0`
- `ERC20 avg val rec`: filled `829` missing values with `0`
- `ERC20 min val sent`: filled `829` missing values with `0`
- `ERC20 max val sent`: filled `829` missing values with `0`
- `ERC20 avg val sent`: filled `829` missing values with `0`
- `ERC20 uniq sent token name`: filled `829` missing values with `0`
- `ERC20 uniq rec token name`: filled `829` missing values with `0`

## Token Text Normalization

- `ERC20 most sent token type`: `NoTokenInfo` count = `8268`
- `ERC20_most_rec_token_type`: `NoTokenInfo` count = `5280`

## Output Files

- Cleaned dataset: `transaction_dataset_cleaned.csv`
- Field dictionary: `transaction_dataset_field_dictionary.csv`
- Summary JSON: `transaction_dataset_cleaning_summary.json`
