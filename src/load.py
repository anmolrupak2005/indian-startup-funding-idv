"""
src/load.py
Step 1: Data Collection & Loading
Handles CSV loading, encoding fallbacks, schema standardization,
and merging of optional 2021 & 2022 datasets into a unified schema.
"""

import os
import glob
import pandas as pd
import numpy as np

COMMON_COLUMNS = [
    "date",
    "year",
    "month",
    "startup",
    "sector",
    "sub_sector",
    "city",
    "investors",
    "investment_type",
    "amount_usd",
    "source_file"
]

def read_csv_with_fallback(filepath, default_encoding="utf-8"):
    """
    Attempts to read a CSV file using default_encoding, falling back
    to 'latin1' or 'cp1252' if UnicodeDecodeError occurs.
    """
    encodings = [default_encoding, "latin1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"  [+] Loaded '{os.path.basename(filepath)}' with encoding '{enc}' ({len(df)} rows, {len(df.columns)} cols)")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  [!] Error reading '{filepath}' with {enc}: {e}")
            break
    # If all fail, try ignore errors
    return pd.read_csv(filepath, encoding="latin1", encoding_errors="ignore", low_memory=False)


def normalize_dataset_1(df, filename="startup_funding.csv"):
    """
    Maps Sudalai Rajkumar's dataset columns to the common schema.
    Columns: Sr No, Date dd/mm/yyyy, Startup Name, Industry Vertical, SubVertical,
             City Location, Investors Name, InvestmentnType, Amount in USD, Remarks
    """
    col_mapping = {}
    for col in df.columns:
        norm = col.strip().lower()
        if "date" in norm:
            col_mapping[col] = "date"
        elif "startup" in norm or "company" in norm:
            col_mapping[col] = "startup"
        elif "industry" in norm or "vertical" == norm:
            col_mapping[col] = "sector"
        elif "subvertical" in norm or "sub-vertical" in norm:
            col_mapping[col] = "sub_sector"
        elif "city" in norm or "location" in norm:
            col_mapping[col] = "city"
        elif "investor" in norm:
            col_mapping[col] = "investors"
        elif "investment" in norm or "type" in norm:
            col_mapping[col] = "investment_type"
        elif "amount" in norm:
            col_mapping[col] = "amount_usd"

    renamed = df.rename(columns=col_mapping)
    standardized = pd.DataFrame()
    for col in COMMON_COLUMNS:
        if col in renamed.columns:
            standardized[col] = renamed[col]
        else:
            standardized[col] = np.nan

    standardized["source_file"] = filename
    return standardized


def normalize_optional_dataset(df, filename):
    """
    Maps 2021/2022 datasets to the common schema.
    Common formats:
    - Company / Brand / Company_Brand -> startup
    - Founded -> optional metadata
    - HeadQuarter / Headquarter -> city
    - Sector -> sector
    - What it does / What_it_does -> sub_sector
    - Investor / Investors -> investors
    - Amount($) / Amount ($) / Amount in USD -> amount_usd
    - Stage -> investment_type
    - Date -> date
    """
    col_mapping = {}
    for col in df.columns:
        norm = col.strip().lower()
        if norm in ["date", "deal_date"]:
            col_mapping[col] = "date"
        elif any(k in norm for k in ["company", "startup", "brand"]):
            col_mapping[col] = "startup"
        elif "sector" in norm or "industry" in norm:
            col_mapping[col] = "sector"
        elif any(k in norm for k in ["what it does", "what_it_does", "sub_sector", "subvertical", "about"]):
            col_mapping[col] = "sub_sector"
        elif any(k in norm for k in ["headquarter", "headquarters", "city", "location"]):
            col_mapping[col] = "city"
        elif "investor" in norm:
            col_mapping[col] = "investors"
        elif any(k in norm for k in ["stage", "round", "investment_type"]):
            col_mapping[col] = "investment_type"
        elif "amount" in norm:
            col_mapping[col] = "amount_usd"

    renamed = df.rename(columns=col_mapping)
    standardized = pd.DataFrame()
    for col in COMMON_COLUMNS:
        if col in renamed.columns:
            standardized[col] = renamed[col]
        else:
            standardized[col] = np.nan

    standardized["source_file"] = filename
    return standardized


def load_all_data(data_dir="data"):
    """
    Main loader function.
    Loads startup_funding.csv (dataset 1) and optionally merges
    2021 and 2022 datasets if found in data_dir.
    """
    print("=" * 60)
    print("STEP 1: DATA COLLECTION & LOADING")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    all_dfs = []
    
    # 1. Primary dataset: startup_funding.csv
    primary_files = glob.glob(os.path.join(data_dir, "*startup_funding*.csv"))
    if not primary_files:
        # Check any csv file in data directory
        primary_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
    if not primary_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'. Please place 'startup_funding.csv' inside '{data_dir}'.")
    
    primary_path = primary_files[0]
    print(f"Loading Primary Dataset: {primary_path}")
    raw_df1 = read_csv_with_fallback(primary_path)
    std_df1 = normalize_dataset_1(raw_df1, os.path.basename(primary_path))
    all_dfs.append(std_df1)
    
    # 2. Check for optional 2021 dataset
    opt_2021_files = [f for f in glob.glob(os.path.join(data_dir, "*2021*.csv")) if f != primary_path]
    if opt_2021_files:
        print(f"Loading Optional 2021 Dataset: {opt_2021_files[0]}")
        raw_df2 = read_csv_with_fallback(opt_2021_files[0])
        std_df2 = normalize_optional_dataset(raw_df2, os.path.basename(opt_2021_files[0]))
        all_dfs.append(std_df2)
    else:
        print("Optional 2021 dataset not found (skipping).")

    # 3. Check for optional 2022 dataset (using encoding cp1252)
    opt_2022_files = [f for f in glob.glob(os.path.join(data_dir, "*2022*.csv")) if f != primary_path]
    if opt_2022_files:
        print(f"Loading Optional 2022 Dataset: {opt_2022_files[0]} (using cp1252)")
        raw_df3 = read_csv_with_fallback(opt_2022_files[0], default_encoding="cp1252")
        std_df3 = normalize_optional_dataset(raw_df3, os.path.basename(opt_2022_files[0]))
        all_dfs.append(std_df3)
    else:
        print("Optional 2022 dataset not found (skipping).")

    # Combine datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[+] Combined Dataset Shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    print(f"[+] Sources Loaded: {combined_df['source_file'].value_counts().to_dict()}")
    print("-" * 60)
    
    return combined_df

if __name__ == "__main__":
    df = load_all_data()
    print("Sample Loaded Data:")
    print(df.head(3))
