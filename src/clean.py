"""
src/clean.py
Step 2: Data Cleaning & Preprocessing
Cleans amounts, fixes malformed dates, standardizes city/sector/stage names,
parses investors, handles duplicates, and generates a comprehensive cleaning report.
"""

import os
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import datetime
import pandas as pd
import numpy as np

def clean_amount(val):
    """
    Cleans amount strings:
    - Removes $, commas, unicode artifacts (\\xc2\\xa0)
    - Replaces undisclosed, unknown, n/a, etc. with NaN
    - Returns float
    """
    if pd.isna(val):
        return np.nan
    
    val_str = str(val).strip()
    
    # Handle byte-string artifacts and unicode spaces
    val_str = val_str.replace("\\xc2\\xa0", "").replace("\xc2\xa0", "").replace("&nbsp;", "")
    val_str = val_str.replace("$", "").replace(",", "").replace("+", "").strip()
    
    # Check for text indicating missing/undisclosed
    undisclosed_patterns = [
        "undisclosed", "unknown", "n/a", "na", "nan", "none", "null",
        "private equity", "seed", "series", "undiscolsed"
    ]
    if val_str.lower() in undisclosed_patterns or not val_str:
        return np.nan
    
    # Extract numeric part
    match = re.search(r"[-+]?\d*\.?\d+", val_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return np.nan
    return np.nan


def parse_single_date(val_str):
    """
    Fixes malformed date strings and parses using dayfirst convention:
    - Replaces dots/backslashes with slashes (e.g., '12/05.2015' -> '12/05/2015')
    - Strips whitespace (e.g., '13/04/2015 ')
    - Fixes missing slash in year (e.g., '05/072018' -> '05/07/2018')
    Returns: (iso_date_string, year_int, month_int) or (None, None, None)
    """
    if not val_str or pd.isna(val_str):
        return None, None, None
        
    raw = str(val_str).strip()
    cleaned = raw.replace(".", "/").replace("\\", "/").replace("-", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    
    # Missing slash between month and 4-digit year: DD/MMYYYY -> DD/MM/YYYY
    m1 = re.match(r"^(\d{1,2})/(\d{2})(\d{4})$", cleaned)
    if m1:
        cleaned = f"{m1.group(1)}/{m1.group(2)}/{m1.group(3)}"
        
    # Unseparated 8 digits DDMMYYYY
    m2 = re.match(r"^(\d{2})(\d{2})(\d{4})$", cleaned)
    if m2:
        cleaned = f"{m2.group(1)}/{m2.group(2)}/{m2.group(3)}"
        
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            if 2000 <= dt.year <= 2035:
                iso_str = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
                return iso_str, int(dt.year), int(dt.month)
        except (ValueError, TypeError):
            continue
            
    return None, None, None


def clean_city_name(val):
    """
    Cleans city location:
    - Strips whitespace
    - Takes first city if multiple are given ('Bangalore / USA' -> 'Bangalore')
    - Unifies Bangalore/Bengaluru -> Bengaluru, Gurgaon/Gurugram -> Gurugram,
      New Delhi/Delhi -> Delhi, Bombay -> Mumbai
    """
    if pd.isna(val):
        return "Unknown"
    
    city_str = str(val).strip()
    for sep in ["/", "&", ","]:
        if sep in city_str:
            city_str = city_str.split(sep)[0].strip()
            break
            
    city_lower = city_str.lower().strip()
    
    if city_lower in ["bangalore", "bengaluru"]:
        return "Bengaluru"
    elif city_lower in ["gurgaon", "gurugram"]:
        return "Gurugram"
    elif city_lower in ["new delhi", "delhi"]:
        return "Delhi"
    elif city_lower in ["mumbai", "bombay"]:
        return "Mumbai"
    elif city_lower in ["hyderabad"]:
        return "Hyderabad"
    elif city_lower in ["pune"]:
        return "Pune"
    elif city_lower in ["chennai", "madras"]:
        return "Chennai"
    elif city_lower in ["noida"]:
        return "Noida"
    elif city_lower in ["ahmedabad", "ahmadabad"]:
        return "Ahmedabad"
    elif city_lower in ["kolkata", "calcutta"]:
        return "Kolkata"
    elif city_lower in ["jaipur"]:
        return "Jaipur"
    elif city_lower in ["chandigarh"]:
        return "Chandigarh"
    elif city_lower in ["indore"]:
        return "Indore"
    elif city_lower in ["kochi", "cochin"]:
        return "Kochi"
    elif city_lower in ["goa"]:
        return "Goa"
    else:
        return city_str.title() if city_str else "Unknown"


def assign_city_group(city):
    """
    Groups Delhi, Gurugram, and Noida into 'Delhi NCR'.
    Retains other cities as their clean name.
    """
    if city in ["Delhi", "Gurugram", "Noida"]:
        return "Delhi NCR"
    return city


def standardize_sector(val):
    """
    Standardizes sector names:
    - Merges spelling variants (eCommerce, E-Commerce -> E-Commerce; FinTech, Fin-Tech -> FinTech)
    """
    if pd.isna(val):
        return "Others"
    
    val_str = str(val).strip().lower()
    
    if any(k in val_str for k in ["ecommerce", "e-commerce", "e commerce", "online shopping", "b2b ecommerce"]):
        return "E-Commerce"
    elif any(k in val_str for k in ["fin-tech", "fintech", "finance", "payment", "banking", "lending", "wealth"]):
        return "FinTech"
    elif any(k in val_str for k in ["ed-tech", "edtech", "education", "e-learning", "ed tech"]):
        return "EdTech"
    elif any(k in val_str for k in ["health", "medical", "pharma", "diagnostic"]):
        return "HealthTech"
    elif any(k in val_str for k in ["food", "beverage", "restaurant", "cloud kitchen"]):
        return "Food & Beverage"
    elif any(k in val_str for k in ["logistics", "supply chain", "delivery", "freight"]):
        return "Logistics"
    elif any(k in val_str for k in ["consumer internet", "internet", "marketplace"]):
        return "Consumer Internet"
    elif any(k in val_str for k in ["saas", "software", "technology", "tech", "cloud", "api"]):
        return "Technology"
    elif any(k in val_str for k in ["auto", "electric vehicle", "ev", "mobility", "cab"]):
        return "Automotive"
    elif any(k in val_str for k in ["agri", "agriculture"]):
        return "AgriTech"
    elif any(k in val_str for k in ["gaming", "game", "esports"]):
        return "Gaming"
    elif any(k in val_str for k in ["hospitality", "hotel", "travel"]):
        return "Hospitality"
    elif any(k in val_str for k in ["media", "entertainment", "ott", "video"]):
        return "Media & Entertainment"
    elif any(k in val_str for k in ["real estate", "proptech", "co-working", "housing"]):
        return "Real Estate"
    else:
        return val.strip().title() if val else "Others"


def standardize_investment_type(val):
    """
    Maps investment types into standard stage groups:
    Seed, Angel, Series A, Series B, Series C+, Private Equity, Debt, Others
    """
    if pd.isna(val):
        return "Others"
    
    val_str = str(val).strip().lower()
    
    if "angel" in val_str and "seed" not in val_str:
        return "Angel"
    elif any(k in val_str for k in ["seed", "angle", "pre-seed", "pre seed"]):
        return "Seed"
    elif "series a" in val_str or "pre-series a" in val_str:
        return "Series A"
    elif "series b" in val_str or "pre-series b" in val_str:
        return "Series B"
    elif any(f"series {c}" in val_str for c in ["c", "d", "e", "f", "g", "h", "i", "j"]) or "series c+" in val_str:
        return "Series C+"
    elif any(k in val_str for k in ["private equity", "privateequity", "pe", "growth"]):
        return "Private Equity"
    elif any(k in val_str for k in ["debt", "venture debt", "loan"]):
        return "Debt"
    else:
        return "Others"


def clean_investors(val):
    """
    Splits comma-separated investor string into clean list of investor names.
    Returns: (list_of_investors, investor_count, is_undisclosed)
    """
    if pd.isna(val):
        return ["Undisclosed Investors"], 0, True
    
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ["undisclosed", "undisclosed investors", "unknown", "n/a", "nan", "none"]:
        return ["Undisclosed Investors"], 0, True
        
    raw_list = re.split(r"[,;]", val_str)
    clean_list = []
    for inv in raw_list:
        inv_clean = inv.strip()
        inv_clean = re.sub(r"^['\"]|['\"]$", "", inv_clean).strip()
        if inv_clean and inv_clean.lower() not in ["undisclosed", "undisclosed investors", "others"]:
            clean_list.append(inv_clean)
            
    if not clean_list:
        return ["Undisclosed Investors"], 0, True
        
    return clean_list, len(clean_list), False


def clean_dataset(df, output_dir="outputs"):
    """
    Executes full Step 2 Data Cleaning pipeline.
    Saves outputs/cleaning_report.txt and outputs/clean_startup_funding.csv.
    """
    print("=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    initial_rows = len(df)
    
    # 1. Deduplication
    df_dedup = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df_dedup)
    print(f"  [+] Removed {duplicates_removed} exact duplicate rows.")
    
    # 2. Date cleaning & parsing
    parsed_dates = []
    parsed_years = []
    parsed_months = []
    dropped_dates = 0

    for raw_d in df_dedup["date"]:
        iso_str, yr, mo = parse_single_date(raw_d)
        parsed_dates.append(iso_str)
        parsed_years.append(yr)
        parsed_months.append(mo)
        if iso_str is None:
            dropped_dates += 1

    df_dedup["clean_date"] = parsed_dates
    df_dedup["clean_year"] = parsed_years
    df_dedup["clean_month"] = parsed_months

    # Drop rows where date could not be parsed
    valid_date_mask = df_dedup["clean_date"].notna()
    df_clean = df_dedup[valid_date_mask].copy()
    df_clean["date"] = df_clean["clean_date"]
    df_clean["year"] = df_clean["clean_year"].astype(int)
    df_clean["month"] = df_clean["clean_month"].astype(int)
    df_clean.drop(columns=["clean_date", "clean_year", "clean_month"], inplace=True)
    print(f"  [+] Dropped {dropped_dates} rows with unparseable dates. Standardized date, year, month.")
    
    # 3. Clean Amount
    df_clean["amount_usd"] = df_clean["amount_usd"].apply(clean_amount)
    df_clean["amount_disclosed"] = df_clean["amount_usd"].notna() & (df_clean["amount_usd"] > 0)
    disclosed_count = int(df_clean["amount_disclosed"].sum())
    print(f"  [+] Cleaned funding amounts: {disclosed_count} deals with disclosed amounts, {len(df_clean) - disclosed_count} undisclosed.")

    # 4. Clean City and assign city_group
    cities_before_unique = df_clean["city"].nunique(dropna=False)
    df_clean["city"] = df_clean["city"].apply(clean_city_name)
    df_clean["city_group"] = df_clean["city"].apply(assign_city_group)
    cities_after_unique = df_clean["city"].nunique()
    print(f"  [+] Unified city names: {cities_before_unique} raw -> {cities_after_unique} standardized. Generated city_group.")

    # 5. Clean Sector & group rare sectors (<10 deals)
    sectors_before_unique = df_clean["sector"].nunique(dropna=False)
    df_clean["sector"] = df_clean["sector"].apply(standardize_sector)
    
    sector_counts = df_clean["sector"].value_counts()
    frequent_sectors = set(sector_counts[sector_counts >= 10].index)
    df_clean["sector"] = df_clean["sector"].apply(lambda s: s if s in frequent_sectors else "Others")
    sectors_after_unique = df_clean["sector"].nunique()
    print(f"  [+] Unified sectors: {sectors_before_unique} raw -> {sectors_after_unique} standard sectors (rare grouped into 'Others').")

    # 6. Clean Investment Type
    df_clean["investment_type"] = df_clean["investment_type"].apply(standardize_investment_type)
    print(f"  [+] Standardized investment stages into: {sorted(df_clean['investment_type'].unique())}")

    # 7. Clean Investors & investor_count
    investor_results = df_clean["investors"].apply(clean_investors)
    df_clean["investors_list"] = [r[0] for r in investor_results]
    df_clean["investor_count"] = [r[1] for r in investor_results]
    df_clean["investors_clean"] = df_clean["investors_list"].apply(lambda l: ", ".join(l))
    print(f"  [+] Parsed investors: computed investor_count, handled undisclosed investors.")

    # Clean Startup Names
    df_clean["startup"] = df_clean["startup"].astype(str).str.strip()
    df_clean["sub_sector"] = df_clean["sub_sector"].fillna("General / Not Specified").astype(str).str.strip()

    final_rows = len(df_clean)
    
    # Generate Cleaning Report
    report_path = os.path.join(output_dir, "cleaning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("INDIAN STARTUP FUNDING: DATA CLEANING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. ROW SUMMARY:\n")
        f.write(f"   - Initial Rows Loaded:       {initial_rows}\n")
        f.write(f"   - Exact Duplicates Removed:  {duplicates_removed}\n")
        f.write(f"   - Unparseable Dates Dropped: {dropped_dates}\n")
        f.write(f"   - Final Cleaned Rows:        {final_rows} ({(final_rows/initial_rows)*100:.1f}% retained)\n\n")
        f.write("2. STANDARDIZATION & MERGES:\n")
        f.write(f"   - Cities: {cities_before_unique} raw variations merged into {cities_after_unique} unique cities.\n")
        f.write(f"   - City Groups: Delhi, Gurugram, and Noida unified under 'Delhi NCR'.\n")
        f.write(f"   - Sectors: {sectors_before_unique} variations normalized into {sectors_after_unique} active sectors (<10 deals -> 'Others').\n")
        f.write(f"   - Investment Stages: Standardized into {df_clean['investment_type'].nunique()} groups (Seed, Angel, Series A, Series B, Series C+, Private Equity, Debt, Others).\n\n")
        f.write("3. MISSING VALUE AUDIT:\n")
        for col in ["date", "startup", "sector", "city", "investment_type", "amount_usd", "investor_count"]:
            missing = df_clean[col].isna().sum()
            f.write(f"   - {col:18s}: {missing:4d} missing ({missing/final_rows*100:.1f}%)\n")
        f.write("\n4. FUNDING DISCLOSURE AUDIT:\n")
        f.write(f"   - Disclosed Deals:   {disclosed_count} ({disclosed_count/final_rows*100:.1f}%)\n")
        f.write(f"   - Undisclosed Deals: {final_rows - disclosed_count} ({(final_rows - disclosed_count)/final_rows*100:.1f}%)\n")
        f.write("=" * 60 + "\n")

    print(f"  [+] Saved Cleaning Report to: {report_path}")
    
    # Save clean dataset
    clean_csv_path = os.path.join(output_dir, "clean_startup_funding.csv")
    export_df = df_clean.drop(columns=["investors_list"]).copy()
    export_df["investors"] = export_df["investors_clean"]
    export_df.drop(columns=["investors_clean"], inplace=True)
    export_df.to_csv(clean_csv_path, index=False)
    print(f"  [+] Saved Clean Dataset to: {clean_csv_path} ({len(export_df)} rows)")
    print("-" * 60)
    
    return df_clean

if __name__ == "__main__":
    from load import load_all_data
    raw_data = load_all_data()
    clean_df = clean_dataset(raw_data)
