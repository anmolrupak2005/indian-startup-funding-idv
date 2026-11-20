"""
app.py
Comprehensive Interactive Data Visualization Dashboard
Title: Indian Startup Funding Trends: An Interactive Visual Analysis by City, Sector, Year and Investor Type
Capstone IDV Project - 7th Semester B.Tech Computer Science & Engineering
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Import modular helper packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from charts import (
    plot_overview_yoy,
    plot_period_donut,
    plot_deal_distribution,
    plot_period_comparison,
    plot_animated_bar_race,
    plot_deal_size_buckets,
    build_investor_network,
    plot_india_bubble_map,
    plot_forecast,
    plot_data_quality_metrics,
    generate_executive_html_report
)
from insights import generate_auto_insights, compute_sector_cagr_and_yoy

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Startup Funding | Interactive Visual Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for KPI cards, colored deltas, badge highlights, and clean typography
st.markdown("""
<style>
    .header-container {
        padding: 10px 0 20px 0;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-meta {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 6px;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-icon {
        font-size: 1.2rem;
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .kpi-delta-pos {
        color: #10B981;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .kpi-delta-neg {
        color: #EF4444;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .kpi-delta-neutral {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .insight-box {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HARD-CODED LAT/LONG FOR TOP INDIAN STARTUP HUBS
# -----------------------------------------------------------------------------
CITY_COORDINATES = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi NCR": (28.6139, 77.2090),
    "Delhi": (28.7041, 77.1025),
    "Gurugram": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Ahmedabad": (23.0225, 72.5714),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
    "Indore": (22.7196, 75.8577),
    "Chandigarh": (30.7333, 76.7794),
    "Kochi": (9.9312, 76.2673),
    "Goa": (15.2993, 74.1240),
    "Coimbatore": (11.0168, 76.9558),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Bhopal": (23.2599, 77.4126),
    "Lucknow": (26.8467, 80.9462),
    "Thiruvananthapuram": (8.5241, 76.9366)
}

# -----------------------------------------------------------------------------
# 3. CACHED DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_clean_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_csv_path = os.path.join(base_dir, "outputs", "clean_startup_funding.csv")
    
    if not os.path.exists(clean_csv_path):
        from load import load_all_data
        from clean import clean_dataset
        from features import engineer_features
        raw = load_all_data(os.path.join(base_dir, "data"))
        cleaned = clean_dataset(raw, os.path.join(base_dir, "outputs"))
        df = engineer_features(cleaned, os.path.join(base_dir, "outputs"))
    else:
        df = pd.read_csv(clean_csv_path)

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    if "amount_musd" not in df.columns:
        df["amount_musd"] = df["amount_usd"] / 1e6
        
    return df

df = load_clean_dataset()

# Read cleaning report metrics for the pipeline diagram
cleaning_report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "cleaning_report.txt")
raw_row_count = len(df) + 16
clean_row_count = len(df)
dups_removed = 2
dates_dropped = 14

if os.path.exists(cleaning_report_path):
    try:
        with open(cleaning_report_path, "r", encoding="utf-8") as f:
            content = f.read()
            m_init = re.search(r"Initial Rows Loaded:\s+(\d+)", content)
            m_dup = re.search(r"Exact Duplicates Removed:\s+(\d+)", content)
            m_drop = re.search(r"Unparseable Dates Dropped:\s+(\d+)", content)
            m_clean = re.search(r"Final Cleaned Rows:\s+(\d+)", content)
            if m_init: raw_row_count = int(m_init.group(1))
            if m_dup: dups_removed = int(m_dup.group(1))
            if m_drop: dates_dropped = int(m_drop.group(1))
            if m_clean: clean_row_count = int(m_clean.group(1))
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS & DYNAMIC FILTERS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/rocket.png", width=60)
st.sidebar.title("Dashboard Filters")

min_year = int(df["year"].min())
max_year = int(df["year"].max())

year_slider = st.sidebar.slider(
    "Select Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

all_cities = sorted(df["city_group"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect(
    "Filter by City / Hub:",
    options=all_cities,
    default=all_cities
)

all_sectors = sorted(df["sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Filter by Sector:",
    options=all_sectors,
    default=all_sectors
)

all_stages = sorted(df["investment_type"].dropna().unique().tolist())
selected_stages = st.sidebar.multiselect(
    "Filter by Stage:",
    options=all_stages,
    default=all_stages
)

if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

st.sidebar.markdown("---")

# Filter application
filtered_df = df[
    (df["year"] >= year_slider[0]) &
    (df["year"] <= year_slider[1]) &
    (df["city_group"].isin(selected_cities if selected_cities else all_cities)) &
    (df["sector"].isin(selected_sectors if selected_sectors else all_sectors)) &
    (df["investment_type"].isin(selected_stages if selected_stages else all_stages))
].copy()

st.sidebar.success(f"**Filtered:** {len(filtered_df):,} of {len(df):,} deals")
st.sidebar.info(f"**Disclosed Capital:** ${filtered_df['amount_musd'].sum():,.1f}M")

# -----------------------------------------------------------------------------
# 5. HEADER (TITLE, DATASET PERIOD & LAST-UPDATED DATE)
# -----------------------------------------------------------------------------
min_ds_year = df["year"].min()
max_ds_year = df["year"].max()
last_updated = datetime.datetime.now().strftime("%B %d, %Y")

st.markdown(f"""
<div class="header-container">
    <div class="header-title">🚀 Indian Startup Funding Trends: Interactive Visual Analysis</div>
    <div class="header-meta">
        📅 <strong>Dataset Scope:</strong> {min_ds_year} – {max_ds_year} &nbsp;|&nbsp; 
        🕒 <strong>Last Updated:</strong> {last_updated} &nbsp;|&nbsp; 
        🎓 <strong>Course:</strong> 7th Semester B.Tech CSE (Interactive Data Visualization Capstone)
    </div>
</div>
""", unsafe_allow_html=True)

if filtered_df.empty:
    st.error("No deals match your current filter settings. Please expand your selections in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. DYNAMIC KPI CARDS WITH YEAR-OVER-YEAR (YoY) DELTAS
# -----------------------------------------------------------------------------
total_fund = filtered_df["amount_musd"].sum()
total_deals = len(filtered_df)
disclosed_df = filtered_df[filtered_df["amount_disclosed"] & (filtered_df["amount_musd"] > 0)]
avg_deal = disclosed_df["amount_musd"].mean() if not disclosed_df.empty else 0.0
med_deal = disclosed_df["amount_musd"].median() if not disclosed_df.empty else 0.0

exploded_inv = filtered_df["investors"].astype(str).str.split(r",\s*").explode().str.strip()
clean_inv = exploded_inv[~exploded_inv.str.lower().isin(["undisclosed investors", "undisclosed", "n/a", "unknown", "", "nan"])]
unique_investors = clean_inv.nunique()

# Compute YoY Delta
years_avail = sorted(filtered_df["year"].unique())
fund_delta_str = "Baseline period"
deals_delta_str = "Baseline period"
avg_delta_str = "Baseline period"
fund_class = "kpi-delta-neutral"
deals_class = "kpi-delta-neutral"
avg_class = "kpi-delta-neutral"

if len(years_avail) >= 2:
    y_curr = years_avail[-1]
    y_prev = years_avail[-2]
    
    fc = filtered_df[filtered_df["year"] == y_curr]["amount_musd"].sum()
    fp = filtered_df[filtered_df["year"] == y_prev]["amount_musd"].sum()
    if fp > 0:
        d_pct = ((fc - fp) / fp) * 100
        sign = "+" if d_pct >= 0 else ""
        fund_delta_str = f"{sign}{d_pct:.1f}% YoY ({y_prev}→{y_curr})"
        fund_class = "kpi-delta-pos" if d_pct >= 0 else "kpi-delta-neg"
        
    dc = len(filtered_df[filtered_df["year"] == y_curr])
    dp = len(filtered_df[filtered_df["year"] == y_prev])
    if dp > 0:
        d_pct = ((dc - dp) / dp) * 100
        sign = "+" if d_pct >= 0 else ""
        deals_delta_str = f"{sign}{d_pct:.1f}% YoY ({y_prev}→{y_curr})"
        deals_class = "kpi-delta-pos" if d_pct >= 0 else "kpi-delta-neg"
        
    ac = filtered_df[(filtered_df["year"] == y_curr) & filtered_df["amount_disclosed"]]["amount_musd"].mean()
    ap = filtered_df[(filtered_df["year"] == y_prev) & filtered_df["amount_disclosed"]]["amount_musd"].mean()
    if pd.notna(ap) and ap > 0 and pd.notna(ac):
        d_pct = ((ac - ap) / ap) * 100
        sign = "+" if d_pct >= 0 else ""
        avg_delta_str = f"{sign}{d_pct:.1f}% YoY ({y_prev}→{y_curr})"
        avg_class = "kpi-delta-pos" if d_pct >= 0 else "kpi-delta-neg"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Total Capital Infused</span><span class="kpi-icon">💰</span></div>
        <div class="kpi-val">${total_fund:,.1f}M</div>
        <div class="{fund_class}">{fund_delta_str}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Total Transactions</span><span class="kpi-icon">📊</span></div>
        <div class="kpi-val">{total_deals:,}</div>
        <div class="{deals_class}">{deals_delta_str}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Average Deal Size</span><span class="kpi-icon">🎯</span></div>
        <div class="kpi-val">${avg_deal:,.2f}M</div>
        <div class="{avg_class}">{avg_delta_str} (Med: ${med_deal:,.1f}M)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header"><span class="kpi-title">Active VC & Angels</span><span class="kpi-icon">🤝</span></div>
        <div class="kpi-val">{unique_investors:,}</div>
        <div class="kpi-delta-pos">Across {filtered_df['city_group'].nunique()} ecosystems</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. ENHANCED ANALYSIS TABS (ALL 11 TABS)
# -----------------------------------------------------------------------------
(
    tab_overview,
    tab_period,
    tab_growth,
    tab_animated,
    tab_dealsize,
    tab_network,
    tab_map,
    tab_forecast,
    tab_quality,
    tab_pipeline,
    tab_export
) = st.tabs([
    "📈 Overview & Auto Insights",
    "⚖️ Period Comparison",
    "🚀 Growth & CAGR",
    "🎬 Animated Trends",
    "💰 Deal Size & Mega Deals",
    "🕸️ Investor Network",
    "🗺️ India Map",
    "🔮 Forecast",
    "🔍 Data Quality",
    "🔄 Pipeline Architecture",
    "📄 Data & Export"
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW & AUTO INSIGHTS
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Venture Capital Trajectory & Live Insights")
    fig_overview = plot_overview_yoy(filtered_df)
    st.plotly_chart(fig_overview, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_period_donut(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_deal_distribution(filtered_df), use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Automated Statistical Insights (Live Computed via Pandas)")
    insights_list = generate_auto_insights(filtered_df)
    
    col_ins1, col_ins2 = st.columns(2)
    mid = (len(insights_list) + 1) // 2
    with col_ins1:
        for ins in insights_list[:mid]:
            st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)
    with col_ins2:
        for ins in insights_list[mid:]:
            st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: PERIOD COMPARISON (PRE-COVID VS COVID VS POST-COVID)
# -----------------------------------------------------------------------------
with tab_period:
    st.subheader("Side-by-Side Macro Period Analysis")
    st.markdown("Comparison across **Pre-COVID (≤2019)**, **COVID (2020–2021)**, and **Post-COVID (≥2022)** funding eras.")

    fig_macro, fig_sec, fig_city = plot_period_comparison(filtered_df)
    st.plotly_chart(fig_macro, use_container_width=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.plotly_chart(fig_sec, use_container_width=True)
    with col_p2:
        st.plotly_chart(fig_city, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: GROWTH & CAGR
# -----------------------------------------------------------------------------
with tab_growth:
    st.subheader("Sector-wise Compound Annual Growth Rate (CAGR) & YoY Velocity")
    st.markdown("Evaluates annualized venture growth for sectors with representative deal volume.")

    growth_table, fast_grow, fast_decl = compute_sector_cagr_and_yoy(filtered_df, min_deals=20)

    cg1, cg2 = st.columns(2)
    with cg1:
        if fast_grow:
            metric_label = f"+{fast_grow['CAGR (%)']:.1f}% CAGR" if fast_grow.get("CAGR (%)") is not None else f"{fast_grow.get('Latest YoY (%)', 0):+.1f}% YoY"
            st.success(f"🚀 **Fastest-Growing Sector:** **{fast_grow['Sector']}** ({metric_label}) with {fast_grow['Total Deals']} total deals.")
        else:
            st.info("Insufficient multi-year data in filtered scope to calculate positive CAGR.")

    with cg2:
        if fast_decl:
            metric_label = f"{fast_decl['CAGR (%)']:.1f}% CAGR" if fast_decl.get("CAGR (%)") is not None else f"{fast_decl.get('Latest YoY (%)', 0):+.1f}% YoY"
            st.warning(f"📉 **Fastest-Declining Sector:** **{fast_decl['Sector']}** ({metric_label}) with {fast_decl['Total Deals']} total deals.")
        else:
            st.info("No declining sectors detected in the current filter scope.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(growth_table, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: ANIMATED TRENDS (BAR CHART RACE)
# -----------------------------------------------------------------------------
with tab_animated:
    st.subheader("Animated Cumulative Capital Accumulation Race")
    st.markdown("Click **Play** on the animation slider to watch venture capital accumulate across years.")

    race_type = st.radio("Select Entity for Bar Race:", options=["Cities / Hubs", "Industry Sectors"], horizontal=True)
    col_dim = "city_group" if race_type == "Cities / Hubs" else "sector"
    
    fig_race = plot_animated_bar_race(filtered_df, category_col=col_dim, top_n=10)
    st.plotly_chart(fig_race, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 5: DEAL SIZE & MEGA DEALS
# -----------------------------------------------------------------------------
with tab_dealsize:
    st.subheader("Deal Size Segmentation & Mega Round Analysis")
    st.markdown("Classifies deals into `<$1M`, `$1–5M`, `$5–20M`, `$20–100M`, and `$100M+ Mega Deals`.")

    fig_buckets, mega_df = plot_deal_size_buckets(filtered_df)
    st.plotly_chart(fig_buckets, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Top 10 Mega Deals ($100M+ Venture Infusions)")
    st.dataframe(mega_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 6: INVESTOR NETWORK & CO-INVESTMENTS
# -----------------------------------------------------------------------------
with tab_network:
    st.subheader("Investor Syndicate Network & Co-Investment Graph")
    st.markdown("Interactive bipartite graph connecting the **Top 30 Institutional Investors** to their funded **Sectors**.")

    html_net, co_invest_df = build_investor_network(filtered_df, top_n_investors=30)
    components.html(html_net, height=480, scrolling=True)

    st.markdown("---")
    st.subheader("🤝 Frequent Co-Investment Syndicates (Shared Deals)")
    st.markdown("Investor pairs that have co-invested together most often across rounds in this dataset.")
    st.dataframe(co_invest_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 7: INDIA MAP
# -----------------------------------------------------------------------------
with tab_map:
    st.subheader("Geospatial Venture Capital Distribution across India")
    
    years_list = ["All Years"] + sorted(filtered_df["year"].unique().tolist())
    map_year_choice = st.selectbox("Select Year for Map View:", options=years_list)
    sel_yr = None if map_year_choice == "All Years" else int(map_year_choice)

    fig_map = plot_india_bubble_map(filtered_df, CITY_COORDINATES, selected_year=sel_yr)
    st.plotly_chart(fig_map, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 8: FORECAST
# -----------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Statistical Trend Modeling & 2-Year Horizon Forecast")
    st.warning("⚠️ **Disclaimer:** Forecast is estimated using standard linear extrapolation on dataset years; strictly for academic and demonstration purposes.")

    fig_deals_fc, fig_fund_fc, errs = plot_forecast(filtered_df)

    cf1, cf2 = st.columns(2)
    with cf1:
        st.plotly_chart(fig_deals_fc, use_container_width=True)
    with cf2:
        st.plotly_chart(fig_fund_fc, use_container_width=True)

    st.info(
        f"**Model Performance & Projections:**\n\n"
        f"• Historical Deal Mean Absolute Error (MAE): **±{errs['mae_deals']} deals**\n\n"
        f"• Historical Funding MAE: **±${errs['mae_funding']:,.1f}M**\n\n"
        f"• Baseline Project for {errs['next_year']}: **~{int(errs['next_year_deals'])} deals** and **~${errs['next_year_fund']:,.1f}M** deployed."
    )


# -----------------------------------------------------------------------------
# TAB 9: DATA QUALITY & AUDIT
# -----------------------------------------------------------------------------
with tab_quality:
    st.subheader("Data Quality Audit, Missingness & Cleaning Verification")

    fig_missing, fig_undisc, audit_df = plot_data_quality_metrics(filtered_df)
    
    cq1, cq2 = st.columns(2)
    with cq1:
        st.plotly_chart(fig_missing, use_container_width=True)
    with cq2:
        st.plotly_chart(fig_undisc, use_container_width=True)

    st.markdown("---")
    st.subheader("Standardization & Merge Audit Table")
    st.markdown("Detailed mapping of raw ingestion variations into clean canonical entities.")
    st.dataframe(audit_df, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 10: PIPELINE ARCHITECTURE (GRAPHVIZ)
# -----------------------------------------------------------------------------
with tab_pipeline:
    st.subheader("Data Pipeline Architecture & Execution Flow")
    st.markdown("Visual flowchart representing the multi-stage ingestion, preprocessing, and analytical workflow.")

    dot_diagram = f"""
    digraph G {{
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#F8FAFC", color="#CBD5E1", fontsize=11, margin="0.2,0.15"];
        edge [color="#2563EB", penwidth=2, arrowsize=0.8];

        Collection [label="1. Ingestion\\n({raw_row_count} raw rows)\\nMulti-encoding fallback", fillcolor="#EFF6FF", color="#3B82F6"];
        Cleaning [label="2. Cleaning\\n({clean_row_count} retained)\\n-{dups_removed} dups, -{dates_dropped} bad dates", fillcolor="#EFF6FF", color="#3B82F6"];
        Features [label="3. Features\\nPeriods, Tiers, $M\\nQuarter categorization", fillcolor="#EFF6FF", color="#3B82F6"];
        EDA [label="4. Statistical EDA\\n9 High-Res PNGs\\nDynamic insights", fillcolor="#EFF6FF", color="#3B82F6"];
        Dashboard [label="5. Dashboard\\n11 Streamlit Tabs\\nInteractive Plotly", fillcolor="#DBEAFE", color="#1D4ED8"];
        Insights [label="6. Live Insights\\nPandas auto-generation\\nPDF/HTML export", fillcolor="#DBEAFE", color="#1D4ED8"];

        Collection -> Cleaning -> Features -> EDA -> Dashboard -> Insights;
    }}
    """
    st.graphviz_chart(dot_diagram)

    st.markdown("""
    **Pipeline Stages Explained:**
    1. **Data Ingestion (`src/load.py`):** Handles multi-file ingestion with fallback encoding (`utf-8`, `latin1`, `cp1252`), normalizing schemas across 2015–2022 datasets into one common tabular schema.
    2. **Data Cleaning (`src/clean.py`):** Deduplicates records, parses malformed date formats with `dayfirst=True`, removes byte characters (`\\xc2\\xa0`) and currency symbols, unifies city and sector names, and standardizes investment types into 8 classes.
    3. **Feature Engineering (`src/features.py`):** Enriches dataset with macroeconomic funding eras (Pre-COVID, COVID, Post-COVID), Tier-1 vs Tier-2/3 classifications, quarters, and USD Million amounts.
    4. **Exploratory Analysis (`src/eda.py`):** Renders 9 high-resolution Seaborn/Matplotlib charts and writes a markdown summary.
    5. **Interactive Dashboard (`app.py` & `src/charts.py`):** Multi-dimensional slicing, dynamic Plotly visualizations, animated bar races, network graphs, and geospatial mapping.
    6. **Automated Insights (`src/insights.py`):** Formulates plain-English statistical statements calculated purely from the data.
    """)


# -----------------------------------------------------------------------------
# TAB 11: DATA EXPLORER & EXPORT
# -----------------------------------------------------------------------------
with tab_export:
    st.subheader("Data Explorer & Comprehensive Report Exporters")

    search_kw = st.text_input("🔍 Search Startup Name or Sub-Sector:", placeholder="e.g. Flipkart, Razorpay, Logistics...")
    export_display_df = filtered_df.copy()
    if search_kw:
        export_display_df = export_display_df[
            export_display_df["startup"].str.contains(search_kw, case=False, na=False) |
            export_display_df["sub_sector"].str.contains(search_kw, case=False, na=False)
        ]

    cols_view = ["date", "startup", "sector", "sub_sector", "city_group", "investment_type", "amount_musd", "investors", "period", "city_tier"]
    st.dataframe(export_display_df[cols_view].rename(columns={
        "date": "Date",
        "startup": "Startup",
        "sector": "Sector",
        "sub_sector": "Sub-Sector",
        "city_group": "City Hub",
        "investment_type": "Stage",
        "amount_musd": "Amount ($M)",
        "investors": "Investors",
        "period": "Period",
        "city_tier": "Tier"
    }), use_container_width=True, height=400)

    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        csv_bytes = export_display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_bytes,
            file_name="filtered_startup_funding.csv",
            mime="text/csv"
        )

    with col_b2:
        kpi_dict = {
            "funding": total_fund,
            "deals": total_deals,
            "disclosed": len(disclosed_df),
            "avg_deal": avg_deal,
            "med_deal": med_deal,
            "investors": unique_investors
        }
        report_html = generate_executive_html_report(export_display_df, kpi_dict, insights_list)
        st.download_button(
            label="📄 Download 1-Page Executive HTML/PDF Report",
            data=report_html.encode("utf-8"),
            file_name="startup_funding_executive_report.html",
            mime="text/html",
            help="Download and open in browser; print via Ctrl+P / Cmd+P to save as PDF."
        )

# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Interactive Data Visualization (IDV) Capstone Project • 7th Semester B.Tech Computer Science & Engineering<br>"
    "Developed with Streamlit, Plotly, Pandas, Matplotlib, Seaborn, NetworkX, Pyvis & Scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
