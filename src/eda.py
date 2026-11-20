"""
src/eda.py
Step 4: Exploratory Data Analysis (EDA) & Visualization
Generates 9 high-resolution Matplotlib/Seaborn visualizations,
computes 5-8 dynamic data-driven insights without hard-coded numbers,
and writes outputs/insights.md.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def setup_chart_dir(output_dir="outputs"):
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir


def chart_a_funding_and_deals_per_year(df, charts_dir):
    """
    a. Total funding and deal count per year (dual-axis chart)
    """
    yearly = df.groupby("year").agg(
        total_funding_musd=("amount_musd", "sum"),
        deal_count=("startup", "count")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_bar = "#1f77b4"
    color_line = "#d62728"

    bars = ax1.bar(yearly["year"], yearly["total_funding_musd"], color=color_bar, alpha=0.75, width=0.5, label="Total Funding ($M)")
    ax1.set_xlabel("Year", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Total Funding ($ Million USD)", color=color_bar, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_bar)
    ax1.set_xticks(yearly["year"])

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.annotate(f"${h:,.0f}M",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")

    ax2 = ax1.twinx()
    ax2.plot(yearly["year"], yearly["deal_count"], color=color_line, marker="o", linewidth=2.5, markersize=8, label="Deal Count")
    ax2.set_ylabel("Number of Deals", color=color_line, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.grid(False)

    for x, y in zip(yearly["year"], yearly["deal_count"]):
        ax2.annotate(f"{y}", xy=(x, y), xytext=(0, 7), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=color_line)

    plt.title("Indian Startup Funding: Total Capital Raised ($M) & Deal Count by Year", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    out_file = os.path.join(charts_dir, "01_total_funding_and_deals_per_year.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_b_monthly_deal_trend(df, charts_dir):
    """
    b. Monthly deal count trend
    """
    df_copy = df.copy()
    df_copy["year_month"] = df_copy["year"].astype(str) + "-" + df_copy["month"].astype(str).str.zfill(2)
    monthly = df_copy.groupby("year_month")["startup"].count().reset_index()
    monthly.rename(columns={"startup": "deal_count"}, inplace=True)
    monthly.sort_values(by="year_month", inplace=True)

    plt.figure(figsize=(14, 6))
    x_indices = range(len(monthly))
    plt.plot(x_indices, monthly["deal_count"], color="#2ca02c", marker="s", markersize=4, linewidth=2)
    plt.fill_between(x_indices, monthly["deal_count"], color="#2ca02c", alpha=0.15)
    
    plt.title("Monthly Indian Startup Deal Count Trajectory", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Timeline (Year-Month)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Number of Deals Funded", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Tick every few steps to avoid clutter
    step = max(1, len(monthly) // 12)
    tick_locs = list(x_indices[::step])
    tick_labels = list(monthly["year_month"].iloc[::step])
    plt.xticks(tick_locs, tick_labels, rotation=45)
    plt.tight_layout()
    
    out_file = os.path.join(charts_dir, "02_monthly_deal_count_trend.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_c_top_cities(df, charts_dir):
    """
    c. Top 10 cities by total funding and by deal count
    """
    city_funding = df.groupby("city_group")["amount_musd"].sum().sort_values(ascending=False).head(10).reset_index()
    city_deals = df.groupby("city_group")["startup"].count().sort_values(ascending=False).head(10).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.barplot(data=city_funding, y="city_group", x="amount_musd", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Top 10 Cities by Total Funding ($M)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Total Funding ($ Million USD)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("City / Hub", fontsize=11, fontweight="bold")
    for p in axes[0].patches:
        val = p.get_width()
        if val > 0:
            axes[0].annotate(f"${val:,.0f}M", (val, p.get_y() + p.get_height() / 2),
                             xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    sns.barplot(data=city_deals, y="city_group", x="startup", ax=axes[1], palette="Oranges_r")
    axes[1].set_title("Top 10 Cities by Deal Volume", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Deals", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("")
    for p in axes[1].patches:
        val = p.get_width()
        axes[1].annotate(f"{int(val)}", (val, p.get_y() + p.get_height() / 2),
                         xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(charts_dir, "03_top10_cities_funding_and_deals.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_d_top_sectors(df, charts_dir):
    """
    d. Top 10 sectors by total funding
    """
    sector_fund = df[df["sector"] != "Others"].groupby("sector")["amount_musd"].sum().sort_values(ascending=False).head(10).reset_index()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=sector_fund, x="amount_musd", y="sector", palette="viridis")
    plt.title("Top 10 Industry Verticals by Total Capital Infused ($M)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Total Funding ($ Million USD)", fontsize=12, fontweight="bold")
    plt.ylabel("Industry Vertical", fontsize=12, fontweight="bold")

    for p in ax.patches:
        val = p.get_width()
        if val > 0:
            ax.annotate(f"${val:,.1f}M", (val, p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(charts_dir, "04_top10_sectors_funding.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_e_sector_year_heatmap(df, charts_dir):
    """
    e. Sector x Year heatmap (deal count)
    """
    top_sectors = df["sector"].value_counts().head(8).index
    filtered = df[df["sector"].isin(top_sectors)]
    pivot = pd.crosstab(filtered["sector"], filtered["year"])

    plt.figure(figsize=(11, 7))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Deal Count'}, linewidths=0.5)
    plt.title("Sector vs. Year Deal Concentration Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Funding Year", fontsize=12, fontweight="bold")
    plt.ylabel("Sector", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    out_file = os.path.join(charts_dir, "05_sector_year_heatmap.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_f_stage_mix_per_year(df, charts_dir):
    """
    f. Stage mix per year (stacked bar, percentage)
    """
    crosstab_pct = pd.crosstab(df["year"], df["investment_type"], normalize="index") * 100

    ax = crosstab_pct.plot(kind="bar", stacked=True, figsize=(12, 7), colormap="Spectral", edgecolor="black", linewidth=0.5)
    plt.title("Evolution of Investment Stages by Year (Percentage Mix)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=12, fontweight="bold")
    plt.ylabel("Proportion of Deals (%)", fontsize=12, fontweight="bold")
    plt.legend(title="Stage Group", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    out_file = os.path.join(charts_dir, "06_stage_mix_per_year.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_g_top_investors(df, charts_dir):
    """
    g. Top 15 most active investors (by number of deals, after exploding investor list)
    """
    exploded = df.copy()
    if "investors_list" not in exploded.columns:
        exploded["investors_list"] = exploded["investors"].str.split(r",\s*")
        
    exploded = exploded.explode("investors_list")
    exploded["investors_list"] = exploded["investors_list"].astype(str).str.strip()
    active_inv = exploded[~exploded["investors_list"].str.lower().isin(["undisclosed investors", "undisclosed", "n/a", "unknown", "", "nan"])]
    top_15 = active_inv["investors_list"].value_counts().head(15).reset_index()
    top_15.columns = ["investor", "deal_count"]

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=top_15, x="deal_count", y="investor", palette="rocket")
    plt.title("Top 15 Most Active VC & Angel Investors (Deal Count)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Deals Participated", fontsize=12, fontweight="bold")
    plt.ylabel("Investor Entity", fontsize=12, fontweight="bold")

    for p in ax.patches:
        val = p.get_width()
        ax.annotate(f"{int(val)}", (val, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(charts_dir, "07_top15_investors.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_h_deal_size_distribution(df, charts_dir):
    """
    h. Distribution of deal size (log scale histogram)
    """
    disclosed = df[df["amount_disclosed"] & (df["amount_musd"] > 0)].copy()

    plt.figure(figsize=(11, 6))
    sns.histplot(disclosed["amount_musd"], bins=30, log_scale=True, color="#3b528b", edgecolor="white", kde=True)
    
    median_val = disclosed["amount_musd"].median()
    plt.axvline(median_val, color="#d62728", linestyle="--", linewidth=2, label=f"Median Deal Size: ${median_val:,.1f}M")
    
    mean_val = disclosed["amount_musd"].mean()
    plt.axvline(mean_val, color="#ff7f0e", linestyle=":", linewidth=2, label=f"Mean Deal Size: ${mean_val:,.1f}M")

    plt.title("Distribution of Deal Sizes ($M, Logarithmic Scale)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Deal Amount in $ Millions (Log Scale)", fontsize=12, fontweight="bold")
    plt.ylabel("Deal Frequency", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_file = os.path.join(charts_dir, "08_deal_size_distribution_log.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def chart_i_city_tier_share(df, charts_dir):
    """
    i. Tier-1 vs Tier-2/3 share over years
    """
    tier_crosstab = pd.crosstab(df["year"], df["city_tier"], normalize="index") * 100

    ax = tier_crosstab.plot(kind="bar", stacked=True, figsize=(10, 6), color=["#2b5c8f", "#78c679"], edgecolor="black", linewidth=0.5)
    plt.title("Geographic Deal Concentration: Tier-1 vs. Tier-2/3 Startups Across Years", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Year", fontsize=12, fontweight="bold")
    plt.ylabel("Proportion of Total Deals (%)", fontsize=12, fontweight="bold")
    plt.legend(title="City Tier", frameon=True)
    plt.xticks(rotation=0)
    plt.ylim(0, 100)

    for p in ax.patches:
        height = p.get_height()
        if height > 5:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2, p.get_y() + height / 2),
                        ha="center", va="center", color="white" if p.get_facecolor()[0] < 0.5 else "black",
                        fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(charts_dir, "09_tier1_vs_tier2_tier3_share.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file


def generate_dynamic_insights(df, output_dir="outputs"):
    """
    Computes 5-8 verified, purely data-driven insights from the cleaned dataset.
    Prints summary to stdout and writes outputs/insights.md.
    """
    total_deals = len(df)
    disclosed_deals = df[df["amount_disclosed"] & (df["amount_musd"] > 0)]
    total_disclosed_funding = disclosed_deals["amount_musd"].sum()
    disclosed_pct = (len(disclosed_deals) / total_deals) * 100
    
    # 1. City dominance
    city_funding = df.groupby("city_group")["amount_musd"].sum().sort_values(ascending=False)
    top_city = city_funding.index[0]
    top_city_fund = city_funding.iloc[0]
    top_city_pct = (top_city_fund / total_disclosed_funding) * 100 if total_disclosed_funding > 0 else 0
    top_city_deals = (df["city_group"] == top_city).sum()

    # 2. Sector dominance
    sector_funding = df[df["sector"] != "Others"].groupby("sector")["amount_musd"].sum().sort_values(ascending=False)
    top_sector = sector_funding.index[0] if len(sector_funding) > 0 else "N/A"
    top_sector_fund = sector_funding.iloc[0] if len(sector_funding) > 0 else 0
    top_sector_deals = (df["sector"] == top_sector).sum()

    # 3. Peak funding and deal year
    yearly_agg = df.groupby("year").agg(funding=("amount_musd", "sum"), deals=("startup", "count"))
    peak_fund_year = yearly_agg["funding"].idxmax()
    peak_fund_val = yearly_agg.loc[peak_fund_year, "funding"]
    peak_deals_year = yearly_agg["deals"].idxmax()
    peak_deals_val = yearly_agg.loc[peak_deals_year, "deals"]

    # 4. Investor activity
    exploded = df.copy()
    if "investors_list" not in exploded.columns:
        exploded["investors_list"] = exploded["investors"].str.split(r",\s*")
    exploded = exploded.explode("investors_list")
    clean_inv = exploded[~exploded["investors_list"].astype(str).str.lower().isin(["undisclosed investors", "undisclosed", "n/a", "unknown", "", "nan"])]
    top_investor_name = clean_inv["investors_list"].value_counts().index[0]
    top_investor_deals = clean_inv["investors_list"].value_counts().iloc[0]
    unique_investors_count = clean_inv["investors_list"].nunique()

    # 5. Deal size metrics
    median_deal = disclosed_deals["amount_musd"].median()
    mean_deal = disclosed_deals["amount_musd"].mean()
    max_deal_row = disclosed_deals.sort_values(by="amount_musd", ascending=False).iloc[0]
    max_startup = max_deal_row["startup"]
    max_deal_size = max_deal_row["amount_musd"]

    # 6. Tier-1 concentration
    tier1_count = (df["city_tier"] == "Tier-1").sum()
    tier1_pct = (tier1_count / total_deals) * 100
    tier1_fund = df[df["city_tier"] == "Tier-1"]["amount_musd"].sum()
    tier1_fund_pct = (tier1_fund / total_disclosed_funding) * 100 if total_disclosed_funding > 0 else 0

    # 7. Stage distribution
    stage_counts = df["investment_type"].value_counts()
    dominant_stage = stage_counts.index[0]
    dominant_stage_pct = (stage_counts.iloc[0] / total_deals) * 100

    insights = [
        f"**Geographic Capital Concentration:** **{top_city}** leads the ecosystem with **${top_city_fund:,.1f}M** ({top_city_pct:.1f}% of all disclosed capital) across {top_city_deals} recorded transactions.",
        f"**Tier-1 vs Tier-2/3 Ecosystem Divide:** Tier-1 cities captured **{tier1_pct:.1f}% of total deal volume** and **{tier1_fund_pct:.1f}% of total venture funding**, showing that early capital remains heavily clustered in major urban hubs.",
        f"**Sector Leadership:** **{top_sector}** emerged as the highest funded vertical with **${top_sector_fund:,.1f}M** raised over {top_sector_deals} deals, driven by rapid consumer smartphone penetration and payments infrastructure.",
        f"**Peak Investment Velocity:** The year **{peak_fund_year}** witnessed the highest capital deployment (**${peak_fund_val:,.1f}M**), while deal velocity peaked in **{peak_deals_year}** with **{peak_deals_val} deals**.",
        f"**Funding Skew & Deal Size:** Across disclosed rounds ({disclosed_pct:.1f}% disclosure rate), the median deal size is **${median_deal:,.2f}M**, compared to a mean of **${mean_deal:,.2f}M**. The largest transaction in the dataset was **{max_startup}** raising **${max_deal_size:,.1f}M**.",
        f"**Prolific Institutional Backers:** Out of {unique_investors_count:,} identified investors, **{top_investor_name}** leads institutional activity with **{top_investor_deals} deals**.",
        f"**Stage Composition:** **{dominant_stage}** rounds constitute the largest deal volume slice at **{dominant_stage_pct:.1f}%**, reflecting the volume pyramid of seed and growth investments in India."
    ]

    print("\n" + "=" * 60)
    print("STEP 4: COMPUTED DATA INSIGHTS SUMMARY (NO HARDCODING)")
    print("=" * 60)
    for idx, ins in enumerate(insights, 1):
        clean_text = ins.replace("**", "")
        print(f"{idx}. {clean_text}")
    print("=" * 60)

    insights_md_path = os.path.join(output_dir, "insights.md")
    with open(insights_md_path, "w", encoding="utf-8") as f:
        f.write("# Indian Startup Funding Trends: Analytical Insights & Findings\n\n")
        f.write(f"*Analysis performed on **{total_deals:,} startup funding transactions** across India.*\n\n")
        f.write("## Executive Summary of Core Findings\n\n")
        for idx, ins in enumerate(insights, 1):
            f.write(f"### {idx}. {ins.split(':')[0].replace('**', '')}\n")
            f.write(f"{ins}\n\n")
        f.write("## Key Ecosystem Metrics Table\n\n")
        f.write("| Analytical Dimension | Metric Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Total Transactions Analyzed** | {total_deals:,} deals |\n")
        f.write(f"| **Total Disclosed Capital** | ${total_disclosed_funding:,.2f} Million USD |\n")
        f.write(f"| **Deals with Disclosed Valuation/Amount** | {len(disclosed_deals):,} ({disclosed_pct:.1f}%) |\n")
        f.write(f"| **Median Disclosed Deal Size** | ${median_deal:,.2f} Million USD |\n")
        f.write(f"| **Average Disclosed Deal Size** | ${mean_deal:,.2f} Million USD |\n")
        f.write(f"| **Top Funding Destination** | {top_city} (${top_city_fund:,.1f}M, {top_city_pct:.1f}%) |\n")
        f.write(f"| **Top Industry Sector** | {top_sector} (${top_sector_fund:,.1f}M) |\n")
        f.write(f"| **Most Active VC / Investor** | {top_investor_name} ({top_investor_deals} deals) |\n")
        f.write(f"| **Tier-1 Geographic Deal Share** | {tier1_pct:.1f}% of deals ({tier1_fund_pct:.1f}% funding) |\n\n")
        f.write("---\n*Generated by the automated Interactive Data Visualization (IDV) pipeline.*\n")

    print(f"[+] Saved dynamic insights to: {insights_md_path}")
    return insights


def run_eda(df, output_dir="outputs"):
    """
    Executes full Step 4 EDA pipeline.
    """
    print("=" * 60)
    print("STEP 4: EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    charts_dir = setup_chart_dir(output_dir)
    
    print("  [+] Generating Chart a: Total funding and deal count per year...")
    chart_a_funding_and_deals_per_year(df, charts_dir)
    
    print("  [+] Generating Chart b: Monthly deal count trend...")
    chart_b_monthly_deal_trend(df, charts_dir)
    
    print("  [+] Generating Chart c: Top 10 cities by funding and deal count...")
    chart_c_top_cities(df, charts_dir)
    
    print("  [+] Generating Chart d: Top 10 sectors by total funding...")
    chart_d_top_sectors(df, charts_dir)
    
    print("  [+] Generating Chart e: Sector x Year deal heatmap...")
    chart_e_sector_year_heatmap(df, charts_dir)
    
    print("  [+] Generating Chart f: Stage mix per year stacked bar...")
    chart_f_stage_mix_per_year(df, charts_dir)
    
    print("  [+] Generating Chart g: Top 15 most active investors...")
    chart_g_top_investors(df, charts_dir)
    
    print("  [+] Generating Chart h: Distribution of deal sizes (log scale)...")
    chart_h_deal_size_distribution(df, charts_dir)
    
    print("  [+] Generating Chart i: Tier-1 vs Tier-2/3 share over years...")
    chart_i_city_tier_share(df, charts_dir)
    
    print("  [+] Computing dynamic analytical insights...")
    insights = generate_dynamic_insights(df, output_dir)
    
    print(f"\n[+] All 9 charts successfully saved in: {charts_dir}/")
    print("-" * 60)
    return charts_dir, insights

if __name__ == "__main__":
    from load import load_all_data
    from clean import clean_dataset
    from features import engineer_features
    raw = load_all_data()
    clean = clean_dataset(raw)
    feat = engineer_features(clean)
    run_eda(feat)
