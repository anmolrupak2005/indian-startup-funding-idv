"""
src/features.py
Step 3: Feature Engineering
Enriches the cleaned dataset with analytical features:
- quarter
- period label (Pre-COVID, COVID, Post-COVID)
- city_tier (Tier-1 vs Tier-2/3)
- amount_musd (amount in million USD)
"""

import os
import pandas as pd
import numpy as np

TIER_1_CITIES = {
    "Bengaluru", "Mumbai", "Delhi NCR", "Delhi", "Gurugram", "Noida",
    "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"
}

def assign_period(year):
    """
    Categorizes deal year into macro periods:
    - Pre-COVID: <= 2019
    - COVID: 2020-2021
    - Post-COVID: >= 2022
    """
    if pd.isna(year):
        return "Unknown"
    y = int(year)
    if y <= 2019:
        return "Pre-COVID (<=2019)"
    elif 2020 <= y <= 2021:
        return "COVID (2020-2021)"
    else:
        return "Post-COVID (>=2022)"


def assign_city_tier(row):
    """
    Assigns city tier:
    Tier-1: Bengaluru, Mumbai, Delhi NCR, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad
    Tier-2/3: All other cities / emerging hubs
    """
    city = row.get("city")
    city_group = row.get("city_group")
    
    if city in TIER_1_CITIES or city_group in TIER_1_CITIES:
        return "Tier-1"
    return "Tier-2/3"


def engineer_features(df, output_dir="outputs"):
    """
    Adds analytical features to the cleaned DataFrame:
    1. quarter: 'Q1', 'Q2', etc. and 'YYYY-Q#'
    2. period: Pre-COVID, COVID, Post-COVID
    3. city_tier: Tier-1 vs Tier-2/3
    4. amount_musd: Amount in Millions of USD ($ Mn)
    Saves updated clean dataset.
    """
    print("=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    df_feat = df.copy()
    
    # Calculate quarter from month
    df_feat["year"] = df_feat["year"].astype(int)
    df_feat["month"] = df_feat["month"].astype(int)
    quarter_num = ((df_feat["month"] - 1) // 3 + 1).astype(str)
    df_feat["quarter"] = "Q" + quarter_num
    df_feat["year_quarter"] = df_feat["year"].astype(str) + "-" + df_feat["quarter"]
    
    # Period categorization
    df_feat["period"] = df_feat["year"].apply(assign_period)
    print(f"  [+] Period distribution: {df_feat['period'].value_counts().to_dict()}")
    
    # City Tier categorization
    df_feat["city_tier"] = df_feat.apply(assign_city_tier, axis=1)
    print(f"  [+] City tier distribution: {df_feat['city_tier'].value_counts().to_dict()}")
    
    # Amount in Millions of USD
    df_feat["amount_musd"] = df_feat["amount_usd"] / 1e6
    disclosed_deals = df_feat[df_feat["amount_disclosed"]]
    total_funding_musd = disclosed_deals["amount_musd"].sum()
    print(f"  [+] Total Disclosed Funding: ${total_funding_musd:,.2f} Million USD across {len(disclosed_deals)} deals.")
    
    # Overwrite clean CSV with enriched features
    clean_csv_path = os.path.join(output_dir, "clean_startup_funding.csv")
    export_df = df_feat.copy()
    if "investors_list" in export_df.columns:
        export_df = export_df.drop(columns=["investors_list"])
    if "investors_clean" in export_df.columns:
        export_df["investors"] = export_df["investors_clean"]
        export_df = export_df.drop(columns=["investors_clean"])
        
    export_df.to_csv(clean_csv_path, index=False)
    print(f"  [+] Updated clean dataset with enriched features at: {clean_csv_path}")
    print("-" * 60)
    
    return df_feat

if __name__ == "__main__":
    from load import load_all_data
    from clean import clean_dataset
    raw = load_all_data()
    clean = clean_dataset(raw)
    feat = engineer_features(clean)
    print(feat[["date", "startup", "year", "quarter", "period", "city_tier", "amount_musd"]].head(3))
