"""
src/insights.py
Automated Statistical Insights and Growth Computations.
Generates verified, purely data-driven analytical narratives (no AI API)
and computes Sector CAGR and Year-over-Year (YoY) trajectories.
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

def generate_auto_insights(df: pd.DataFrame) -> List[str]:
    """
    Generates 6-8 plain-English analytical insight statements computed
    directly from the filtered pandas DataFrame using purely data-driven metrics.
    
    Args:
        df (pd.DataFrame): Filtered startup funding records.
        
    Returns:
        List[str]: List of formatted plain-English insight strings.
    """
    if df.empty:
        return ["No transactions match the selected filter criteria."]

    total_deals = len(df)
    disclosed_deals = df[df["amount_disclosed"] & (df["amount_musd"] > 0)]
    total_funding_musd = disclosed_deals["amount_musd"].sum() if not disclosed_deals.empty else 0.0
    disclosed_pct = (len(disclosed_deals) / total_deals) * 100 if total_deals > 0 else 0.0

    insights: List[str] = []

    # 1. Scope & Capital Overview
    years_in_scope = sorted(df["year"].unique())
    year_range_str = f"{years_in_scope[0]}–{years_in_scope[-1]}" if len(years_in_scope) > 1 else str(years_in_scope[0])
    insights.append(
        f"Across {year_range_str}, a total of **{total_deals:,} deals** were recorded, representing "
        f"**${total_funding_musd:,.1f}M** in disclosed venture capital across {df['city_group'].nunique()} startup hubs."
    )

    # 2. Geographic Dominance
    if not df["city_group"].empty:
        city_funds = df.groupby("city_group")["amount_musd"].sum().sort_values(ascending=False)
        top_city = city_funds.index[0]
        top_city_val = city_funds.iloc[0]
        top_city_pct = (top_city_val / total_funding_musd * 100) if total_funding_musd > 0 else 0.0
        top_city_deals = (df["city_group"] == top_city).sum()
        insights.append(
            f"**{top_city}** captured the largest capital share, securing **${top_city_val:,.1f}M** "
            f"({top_city_pct:.1f}% of total funding) across {top_city_deals} transactions."
        )

    # 3. Sector Leadership
    valid_sectors = df[df["sector"] != "Others"]
    sec_df = valid_sectors if not valid_sectors.empty else df
    if not sec_df["sector"].empty:
        sector_funds = sec_df.groupby("sector")["amount_musd"].sum().sort_values(ascending=False)
        top_sec = sector_funds.index[0]
        top_sec_val = sector_funds.iloc[0]
        top_sec_pct = (top_sec_val / total_funding_musd * 100) if total_funding_musd > 0 else 0.0
        top_sec_deals = (df["sector"] == top_sec).sum()
        insights.append(
            f"**{top_sec}** emerged as the primary investment vertical with **${top_sec_val:,.1f}M** raised "
            f"({top_sec_pct:.1f}% of capital) spanning {top_sec_deals} closed rounds."
        )

    # 4. Round Size Distribution & Outliers
    if not disclosed_deals.empty:
        median_size = disclosed_deals["amount_musd"].median()
        mean_size = disclosed_deals["amount_musd"].mean()
        max_row = disclosed_deals.sort_values(by="amount_musd", ascending=False).iloc[0]
        insights.append(
            f"Disclosed rounds reflect a median deal size of **${median_size:,.2f}M** versus an average of **${mean_size:,.2f}M**, "
            f"anchored by **{max_row['startup']}** raising the peak round of **${max_row['amount_musd']:,.1f}M**."
        )

    # 5. Tier-1 vs Tier-2/3 Concentration
    tier1_deals = (df["city_tier"] == "Tier-1").sum()
    tier1_deal_pct = (tier1_deals / total_deals) * 100
    tier1_funds = df[df["city_tier"] == "Tier-1"]["amount_musd"].sum()
    tier1_fund_pct = (tier1_funds / total_funding_musd * 100) if total_funding_musd > 0 else 0.0
    insights.append(
        f"Tier-1 urban clusters command **{tier1_deal_pct:.1f}% of deal volume** and **{tier1_fund_pct:.1f}% of capital**, "
        f"highlighting that growth-stage capital remains heavily centralized in metro ecosystems."
    )

    # 6. YoY Momentum (comparing last 2 years in selection)
    if len(years_in_scope) >= 2:
        y_curr = years_in_scope[-1]
        y_prev = years_in_scope[-2]
        fund_curr = df[df["year"] == y_curr]["amount_musd"].sum()
        fund_prev = df[df["year"] == y_prev]["amount_musd"].sum()
        if fund_prev > 0:
            growth_pct = ((fund_curr - fund_prev) / fund_prev) * 100
            dir_str = "increased" if growth_pct >= 0 else "contracted"
            insights.append(
                f"Between {y_prev} and {y_curr}, total capital deployed {dir_str} by **{abs(growth_pct):.1f}% YoY** "
                f"(${fund_prev:,.1f}M $\\rightarrow$ ${fund_curr:,.1f}M)."
            )

    # 7. Active Investors & Institutional Syndicates
    exploded_inv = df["investors"].astype(str).str.split(r",\s*").explode().str.strip()
    valid_inv = exploded_inv[~exploded_inv.str.lower().isin(["undisclosed investors", "undisclosed", "n/a", "unknown", "", "nan"])]
    if not valid_inv.empty:
        top_inv = valid_inv.value_counts().index[0]
        top_inv_count = valid_inv.value_counts().iloc[0]
        insights.append(
            f"**{top_inv}** proved to be the most frequent backer in this scope, participating in "
            f"**{top_inv_count} funded rounds** across {valid_inv.nunique():,} unique identified investors."
        )

    # 8. Data Disclosure Audit
    insights.append(
        f"Financial deal size was officially disclosed in **{disclosed_pct:.1f}% of transactions** "
        f"({len(disclosed_deals):,} of {total_deals:,}), with the remainder categorized as undisclosed strategic investments."
    )

    return insights


def compute_sector_cagr_and_yoy(df: pd.DataFrame, min_deals: int = 20) -> Tuple[pd.DataFrame, Optional[Dict], Optional[Dict]]:
    """
    Computes sector-wise Compound Annual Growth Rate (CAGR) and YoY growth trajectories.
    Highlights the fastest growing and fastest declining sectors.
    
    Args:
        df (pd.DataFrame): Input funding dataframe.
        min_deals (int): Minimum deal count threshold to avoid small-sample volatility.
        
    Returns:
        Tuple[pd.DataFrame, dict, dict]: (Growth Summary Table, Fastest Growing, Fastest Declining)
    """
    # Count deals per sector
    sector_counts = df["sector"].value_counts()
    eligible_sectors = sector_counts[sector_counts >= min_deals].index.tolist()
    
    # If fewer than 2 sectors qualify with min_deals=20, fallback gracefully
    if len(eligible_sectors) < 3:
        eligible_sectors = sector_counts[sector_counts >= max(5, min_deals // 2)].index.tolist()
    if not eligible_sectors:
        eligible_sectors = sector_counts.head(5).index.tolist()

    sub_df = df[df["sector"].isin(eligible_sectors)].copy()
    years = sorted(sub_df["year"].unique())

    if len(years) < 2:
        empty_res = pd.DataFrame(columns=["Sector", "Total Deals", "Total Funding ($M)", "CAGR (%)"])
        return empty_res, None, None

    start_year = years[0]
    end_year = years[-1]
    num_years = end_year - start_year

    # Aggregate funding by sector and year
    pivot_fund = sub_df.pivot_table(index="sector", columns="year", values="amount_musd", aggfunc="sum", fill_value=0.0)
    pivot_deals = sub_df.pivot_table(index="sector", columns="year", values="startup", aggfunc="count", fill_value=0)

    results = []
    for sector in eligible_sectors:
        deals_tot = int(sector_counts.get(sector, 0))
        fund_tot = float(df[df["sector"] == sector]["amount_musd"].sum())
        fund_start = float(pivot_fund.loc[sector, start_year]) if start_year in pivot_fund.columns else 0.0
        fund_end = float(pivot_fund.loc[sector, end_year]) if end_year in pivot_fund.columns else 0.0

        # Calculate CAGR
        cagr = np.nan
        if fund_start > 0 and fund_end > 0 and num_years > 0:
            cagr = ((fund_end / fund_start) ** (1.0 / num_years) - 1.0) * 100

        # Calculate latest YoY
        yoy_latest = np.nan
        if len(years) >= 2:
            y_curr = years[-1]
            y_prev = years[-2]
            f_prev = float(pivot_fund.loc[sector, y_prev]) if y_prev in pivot_fund.columns else 0.0
            f_curr = float(pivot_fund.loc[sector, y_curr]) if y_curr in pivot_fund.columns else 0.0
            if f_prev > 0:
                yoy_latest = ((f_curr - f_prev) / f_prev) * 100

        results.append({
            "Sector": sector,
            "Total Deals": deals_tot,
            f"Funding {start_year} ($M)": round(fund_start, 1),
            f"Funding {end_year} ($M)": round(fund_end, 1),
            "Total Funding ($M)": round(fund_tot, 1),
            "CAGR (%)": round(cagr, 1) if not np.isnan(cagr) else None,
            "Latest YoY (%)": round(yoy_latest, 1) if not np.isnan(yoy_latest) else None
        })

    growth_df = pd.DataFrame(results)

    # Determine fastest growing and declining
    valid_cagr = growth_df[growth_df["CAGR (%)"].notna()].sort_values(by="CAGR (%)", ascending=False)
    
    fastest_growing = valid_cagr.iloc[0].to_dict() if not valid_cagr.empty and valid_cagr.iloc[0]["CAGR (%)"] > 0 else None
    fastest_declining = valid_cagr.iloc[-1].to_dict() if not valid_cagr.empty and valid_cagr.iloc[-1]["CAGR (%)"] < 0 else None

    # If CAGR is sparse, fallback to Latest YoY
    if fastest_growing is None:
        valid_yoy = growth_df[growth_df["Latest YoY (%)"].notna()].sort_values(by="Latest YoY (%)", ascending=False)
        if not valid_yoy.empty:
            fastest_growing = valid_yoy.iloc[0].to_dict()
            fastest_declining = valid_yoy.iloc[-1].to_dict()

    return growth_df, fastest_growing, fastest_declining
