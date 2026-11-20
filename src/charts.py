"""
src/charts.py
Modular Visualization Engine for Indian Startup Funding Analytics.
Contains reusable Plotly, NetworkX, Pyvis, and Scikit-learn charting components
with strict type hints, error handling, and clean aesthetic styling.
"""

from typing import Dict, List, Tuple, Optional
import os
import re
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# -----------------------------------------------------------------------------
# 1. OVERVIEW CHARTS
# -----------------------------------------------------------------------------
def plot_overview_yoy(df: pd.DataFrame) -> go.Figure:
    """Generates dual-axis YoY bar and line chart of capital deployed vs deal volume."""
    yearly = df.groupby("year").agg(
        funding_musd=("amount_musd", "sum"),
        deals=("startup", "count")
    ).reset_index().sort_values(by="year")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"],
        y=yearly["funding_musd"],
        name="Total Funding ($M)",
        marker_color="#2563EB",
        opacity=0.85,
        yaxis="y1",
        hovertemplate="<b>Year %{x}</b><br>Capital Raised: $%{y:,.1f} Million<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=yearly["year"],
        y=yearly["deals"],
        name="Deal Volume",
        mode="lines+markers",
        marker=dict(size=9, color="#DC2626"),
        line=dict(width=3, color="#DC2626"),
        yaxis="y2",
        hovertemplate="<b>Year %{x}</b><br>Deals Closed: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Annual Funding Volume ($M) & Deal Count Trajectory", font=dict(size=15, weight="bold")),
        xaxis=dict(title="Year", tickmode="linear"),
        yaxis=dict(
            title=dict(text="Total Funding (USD Million)", font=dict(color="#2563EB")),
            tickfont=dict(color="#2563EB")
        ),
        yaxis2=dict(
            title=dict(text="Deal Count", font=dict(color="#DC2626")),
            tickfont=dict(color="#DC2626"),
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        height=420
    )
    return fig


def plot_period_donut(df: pd.DataFrame) -> go.Figure:
    """Generates donut chart of capital allocation across macro periods."""
    period_df = df.groupby("period").agg(
        funding=("amount_musd", "sum"),
        deals=("startup", "count")
    ).reset_index()

    fig = px.pie(
        period_df,
        values="funding",
        names="period",
        color="period",
        color_discrete_map={
            "Pre-COVID (<=2019)": "#3B82F6",
            "COVID (2020-2021)": "#EF4444",
            "Post-COVID (>=2022)": "#10B981"
        },
        hole=0.45,
        title="Capital Share by Macro Period"
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_deal_distribution(df: pd.DataFrame) -> go.Figure:
    """Generates log-scale histogram of disclosed deal sizes."""
    disclosed = df[df["amount_disclosed"] & (df["amount_musd"] > 0)]
    avg_size = disclosed["amount_musd"].mean() if not disclosed.empty else 0
    med_size = disclosed["amount_musd"].median() if not disclosed.empty else 0

    fig = px.histogram(
        disclosed,
        x="amount_musd",
        log_x=True,
        nbins=28,
        title="Disclosed Deal Size Distribution (Log-Scale)",
        color_discrete_sequence=["#6366F1"],
        labels={"amount_musd": "Deal Size ($ Million, Log Scale)"}
    )
    fig.add_vline(x=avg_size, line_dash="dash", line_color="#DC2626", annotation_text=f"Mean: ${avg_size:,.1f}M")
    fig.add_vline(x=med_size, line_dash="dot", line_color="#10B981", annotation_text=f"Median: ${med_size:,.1f}M")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# -----------------------------------------------------------------------------
# 2. PERIOD COMPARISON TAB
# -----------------------------------------------------------------------------
def plot_period_comparison(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Compares Pre-COVID vs COVID vs Post-COVID side-by-side:
    1. Overall Metrics: Total funding, deals, avg deal size
    2. Top 5 sectors per period
    3. Top 5 cities per period
    """
    # 1. Macro Metrics Grouped Bar
    macro = df.groupby("period").agg(
        funding_musd=("amount_musd", "sum"),
        deals=("startup", "count"),
        avg_deal=("amount_musd", lambda s: s[s > 0].mean() if len(s[s > 0]) > 0 else 0)
    ).reset_index()

    fig_macro = go.Figure()
    fig_macro.add_trace(go.Bar(
        x=macro["period"],
        y=macro["funding_musd"],
        name="Funding ($M)",
        marker_color="#2563EB",
        text=macro["funding_musd"].apply(lambda v: f"${v:,.0f}M"),
        textposition="auto"
    ))
    fig_macro.add_trace(go.Bar(
        x=macro["period"],
        y=macro["deals"],
        name="Deal Count",
        marker_color="#10B981",
        text=macro["deals"],
        textposition="auto"
    ))
    fig_macro.add_trace(go.Bar(
        x=macro["period"],
        y=macro["avg_deal"],
        name="Avg Deal ($M)",
        marker_color="#F59E0B",
        text=macro["avg_deal"].apply(lambda v: f"${v:,.1f}M"),
        textposition="auto"
    ))
    fig_macro.update_layout(
        barmode="group",
        title=dict(text="Key Venture Metrics across Macro Periods", font=dict(size=14, weight="bold")),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 2. Top 5 Sectors per Period
    sec_period = df[df["sector"] != "Others"].groupby(["period", "sector"])["amount_musd"].sum().reset_index()
    top_secs_per_p = sec_period.sort_values(["period", "amount_musd"], ascending=[True, False]).groupby("period").head(5)

    fig_sec = px.bar(
        top_secs_per_p,
        x="amount_musd",
        y="sector",
        color="period",
        barmode="group",
        orientation="h",
        title="Top 5 Funded Sectors by Macro Period",
        labels={"amount_musd": "Total Funding ($M)", "sector": "Sector", "period": "Period"},
        color_discrete_map={
            "Pre-COVID (<=2019)": "#3B82F6",
            "COVID (2020-2021)": "#EF4444",
            "Post-COVID (>=2022)": "#10B981"
        }
    )
    fig_sec.update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))

    # 3. Top 5 Cities per Period
    city_period = df.groupby(["period", "city_group"])["amount_musd"].sum().reset_index()
    top_cities_per_p = city_period.sort_values(["period", "amount_musd"], ascending=[True, False]).groupby("period").head(5)

    fig_city = px.bar(
        top_cities_per_p,
        x="amount_musd",
        y="city_group",
        color="period",
        barmode="group",
        orientation="h",
        title="Top 5 Startup Hubs by Macro Period",
        labels={"amount_musd": "Total Funding ($M)", "city_group": "City / Hub", "period": "Period"},
        color_discrete_map={
            "Pre-COVID (<=2019)": "#3B82F6",
            "COVID (2020-2021)": "#EF4444",
            "Post-COVID (>=2022)": "#10B981"
        }
    )
    fig_city.update_layout(height=420, yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))

    return fig_macro, fig_sec, fig_city


# -----------------------------------------------------------------------------
# 3. ANIMATED BAR CHART RACE TAB
# -----------------------------------------------------------------------------
def plot_animated_bar_race(df: pd.DataFrame, category_col: str = "city_group", top_n: int = 10) -> go.Figure:
    """
    Builds an animated horizontal bar chart showing cumulative capital accumulation
    over time for the top entities.
    """
    valid_df = df[df[category_col] != "Others"].copy() if "Others" in df[category_col].values else df.copy()
    
    # Identify top entities overall
    top_entities = valid_df.groupby(category_col)["amount_musd"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    filtered = valid_df[valid_df[category_col].isin(top_entities)]

    years = sorted(filtered["year"].unique())
    all_frames = []

    # Build cumulative totals per entity across years
    for yr in years:
        sub = filtered[filtered["year"] <= yr]
        agg = sub.groupby(category_col)["amount_musd"].sum().reindex(top_entities, fill_value=0.0).reset_index()
        agg["year"] = yr
        all_frames.append(agg)

    race_df = pd.concat(all_frames, ignore_index=True)
    race_df.rename(columns={category_col: "Entity", "amount_musd": "Cumulative Funding ($M)"}, inplace=True)

    fig = px.bar(
        race_df,
        x="Cumulative Funding ($M)",
        y="Entity",
        color="Entity",
        animation_frame="year",
        orientation="h",
        range_x=[0, race_df["Cumulative Funding ($M)"].max() * 1.15],
        title=f"Cumulative Funding Trajectory Race by {category_col.replace('_', ' ').title()}",
        labels={"Entity": category_col.replace('_', ' ').title()}
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig


# -----------------------------------------------------------------------------
# 4. DEAL SIZE BUCKETS & MEGA DEALS TAB
# -----------------------------------------------------------------------------
def classify_deal_bucket(val_musd: float) -> str:
    """Classifies deal size into standard venture stages."""
    if pd.isna(val_musd) or val_musd <= 0:
        return "Undisclosed"
    elif val_musd < 1.0:
        return "<$1M"
    elif val_musd < 5.0:
        return "$1-$5M"
    elif val_musd < 20.0:
        return "$5-$20M"
    elif val_musd < 100.0:
        return "$20-$100M"
    else:
        return "$100M+ Mega Deals"


def plot_deal_size_buckets(df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Classifies deals into buckets (<$1M, $1–5M, $5–20M, $20–100M, $100M+ mega deals).
    Returns a stacked proportion chart and top 10 mega deals table.
    """
    df_b = df.copy()
    df_b["deal_bucket"] = df_b["amount_musd"].apply(classify_deal_bucket)

    bucket_order = ["<$1M", "$1-$5M", "$5-$20M", "$20-$100M", "$100M+ Mega Deals", "Undisclosed"]
    
    # Yearly bucket share
    cross = pd.crosstab(df_b["year"], df_b["deal_bucket"])
    cross = cross.reindex(columns=[b for b in bucket_order if b in cross.columns], fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    cross_pct_melt = cross_pct.reset_index().melt(id_vars="year", var_name="Bucket", value_name="Share (%)")

    fig = px.bar(
        cross_pct_melt,
        x="year",
        y="Share (%)",
        color="Bucket",
        title="Proportion of Deal Sizes by Tier across Years (%)",
        barmode="stack",
        category_orders={"Bucket": bucket_order},
        color_discrete_sequence=["#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8", "#DC2626", "#94A3B8"]
    )
    fig.update_layout(yaxis=dict(range=[0, 100]), height=420, margin=dict(l=20, r=20, t=40, b=20))

    # Top 10 Mega Deals Table
    mega_deals = df_b[df_b["deal_bucket"] == "$100M+ Mega Deals"].sort_values(by="amount_musd", ascending=False).head(10)
    cols = ["date", "startup", "sector", "city_group", "investment_type", "amount_musd", "investors"]
    top_mega_df = mega_deals[cols].rename(columns={
        "date": "Date",
        "startup": "Startup",
        "sector": "Sector",
        "city_group": "City Hub",
        "investment_type": "Stage",
        "amount_musd": "Amount ($M)",
        "investors": "Key Investors"
    })
    return fig, top_mega_df


# -----------------------------------------------------------------------------
# 5. INVESTOR NETWORK & CO-INVESTMENT TAB
# -----------------------------------------------------------------------------
def build_investor_network(df: pd.DataFrame, top_n_investors: int = 30) -> Tuple[str, pd.DataFrame]:
    """
    Builds a bipartite network of Top 30 Investors & Sectors using NetworkX + Pyvis,
    and extracts top co-investor syndication pairs.
    """
    exploded = df.copy()
    exploded["inv_entity"] = exploded["investors"].astype(str).str.split(r",\s*")
    exploded = exploded.explode("inv_entity")
    exploded["inv_entity"] = exploded["inv_entity"].astype(str).str.strip()
    
    clean_inv = exploded[~exploded["inv_entity"].str.lower().isin(["undisclosed investors", "undisclosed", "n/a", "unknown", "", "nan"])]
    top_invs = clean_inv["inv_entity"].value_counts().head(top_n_investors).index.tolist()

    sub = clean_inv[clean_inv["inv_entity"].isin(top_invs) & (clean_inv["sector"] != "Others")]
    edges = sub.groupby(["inv_entity", "sector"]).size().reset_index(name="deals")

    G = nx.Graph()
    for inv in top_invs:
        deal_c = int((clean_inv["inv_entity"] == inv).sum())
        G.add_node(inv, label=inv, title=f"Investor: {inv} ({deal_c} deals)", group="Investor", size=15 + min(20, deal_c), color="#2563EB")
        
    for sec in sub["sector"].unique():
        sec_c = int((df["sector"] == sec).sum())
        G.add_node(sec, label=sec, title=f"Sector: {sec} ({sec_c} deals)", group="Sector", size=20 + min(15, sec_c // 2), color="#10B981")

    for _, row in edges.iterrows():
        G.add_edge(row["inv_entity"], row["sector"], value=int(row["deals"]), title=f"{row['deals']} deals")

    net = Network(height="460px", width="100%", bgcolor="#F8FAFC", font_color="#1E293B", directed=False)
    net.from_nx(G)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 80,
          "springConstant": 0.08
        },
        "maxVelocity": 30,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      }
    }
    """)
    html_graph = net.generate_html()

    # Co-investment pairs from the same deal
    pairs_counter = Counter()
    for raw in df["investors"]:
        items = [x.strip() for x in str(raw).split(",") if x.strip() and x.strip().lower() not in ["undisclosed investors", "undisclosed", "n/a", "unknown"]]
        if len(items) >= 2:
            for pair in combinations(sorted(set(items)), 2):
                pairs_counter[pair] += 1

    top_pairs = pairs_counter.most_common(15)
    co_invest_records = [
        {"Investor 1": p[0][0], "Investor 2": p[0][1], "Shared Syndicated Deals": p[1]}
        for p in top_pairs
    ]
    co_invest_df = pd.DataFrame(co_invest_records)

    return html_graph, co_invest_df


# -----------------------------------------------------------------------------
# 6. INDIA GEOGRAPHIC BUBBLE MAP TAB
# -----------------------------------------------------------------------------
def plot_india_bubble_map(df: pd.DataFrame, city_coords: Dict[str, Tuple[float, float]], selected_year: Optional[int] = None) -> go.Figure:
    """
    Builds an open-street-map / scatter_geo India map with bubble size = funding
    and color = top sector of that city.
    """
    map_source = df if selected_year is None else df[df["year"] == selected_year]
    
    # Calculate funding and top sector per city
    city_aggs = []
    for city, grp in map_source.groupby("city_group"):
        coords = city_coords.get(city)
        if coords:
            fund_tot = grp["amount_musd"].sum()
            deals_tot = len(grp)
            top_sec = grp["sector"].value_counts().index[0] if not grp["sector"].empty else "General"
            city_aggs.append({
                "city": city,
                "lat": coords[0],
                "lon": coords[1],
                "funding_musd": fund_tot,
                "deals": deals_tot,
                "top_sector": top_sec
            })

    map_df = pd.DataFrame(city_aggs)
    if map_df.empty:
        fig_empty = go.Figure()
        fig_empty.update_layout(title="No geographic coordinates available for selection.")
        return fig_empty

    fig = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        size="funding_musd",
        color="top_sector",
        hover_name="city",
        hover_data={"funding_musd": ":$,.1fM", "deals": True, "top_sector": True, "lat": False, "lon": False},
        size_max=42,
        title=f"India Venture Capital Hubs {f'({selected_year})' if selected_year else 'All Years'}",
        projection="natural earth"
    )
    fig.update_geos(
        center=dict(lat=22.0, lon=79.5),
        lataxis_range=[7.0, 36.0],
        lonaxis_range=[68.0, 96.0],
        showcountries=True,
        countrycolor="#94A3B8",
        showland=True,
        landcolor="#F1F5F9",
        showocean=True,
        oceancolor="#E0F2FE",
        fitbounds=False
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=10))
    return fig


# -----------------------------------------------------------------------------
# 7. FORECAST TAB
# -----------------------------------------------------------------------------
def plot_forecast(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, Dict[str, float]]:
    """
    Forecasts yearly deal count and funding for the next 2 years using Linear Regression.
    Returns prediction plots, 95% confidence intervals, and historical MAE.
    """
    yearly = df.groupby("year").agg(
        funding_musd=("amount_musd", "sum"),
        deals=("startup", "count")
    ).reset_index().sort_values(by="year")

    X = yearly["year"].values.reshape(-1, 1)
    y_deals = yearly["deals"].values
    y_fund = yearly["funding_musd"].values

    last_year = int(yearly["year"].max())
    future_years = np.array([last_year + 1, last_year + 2]).reshape(-1, 1)
    all_future = np.vstack([X, future_years])

    # 1. Deals Forecast
    m_deals = LinearRegression()
    m_deals.fit(X, y_deals)
    pred_deals_hist = m_deals.predict(X)
    mae_deals = mean_absolute_error(y_deals, pred_deals_hist)
    std_deals = np.std(y_deals - pred_deals_hist)
    pred_deals_future = m_deals.predict(future_years)
    pred_deals_all = m_deals.predict(all_future)

    fig_deals = go.Figure()
    fig_deals.add_trace(go.Scatter(
        x=yearly["year"], y=y_deals, mode="lines+markers", name="Observed Deals",
        marker=dict(size=8, color="#2563EB"), line=dict(width=2.5, color="#2563EB")
    ))
    fig_deals.add_trace(go.Scatter(
        x=all_future.flatten(), y=pred_deals_all, mode="lines", name="Trend & Forecast",
        line=dict(width=2, dash="dash", color="#DC2626")
    ))
    # Confidence band
    upper_deals = pred_deals_all + 1.96 * std_deals
    lower_deals = np.maximum(0, pred_deals_all - 1.96 * std_deals)
    fig_deals.add_trace(go.Scatter(
        x=np.concatenate([all_future.flatten(), all_future.flatten()[::-1]]),
        y=np.concatenate([upper_deals, lower_deals[::-1]]),
        fill="toself",
        fillcolor="rgba(220, 38, 38, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Band",
        hoverinfo="skip"
    ))
    fig_deals.update_layout(
        title=f"Deal Velocity Forecast (MAE: ±{mae_deals:.1f} deals)",
        xaxis=dict(title="Year", tickmode="linear"),
        yaxis=dict(title="Deals Closed"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # 2. Funding Forecast
    m_fund = LinearRegression()
    m_fund.fit(X, y_fund)
    pred_fund_hist = m_fund.predict(X)
    mae_fund = mean_absolute_error(y_fund, pred_fund_hist)
    std_fund = np.std(y_fund - pred_fund_hist)
    pred_fund_future = m_fund.predict(future_years)
    pred_fund_all = m_fund.predict(all_future)

    fig_fund = go.Figure()
    fig_fund.add_trace(go.Scatter(
        x=yearly["year"], y=y_fund, mode="lines+markers", name="Observed Capital",
        marker=dict(size=8, color="#10B981"), line=dict(width=2.5, color="#10B981")
    ))
    fig_fund.add_trace(go.Scatter(
        x=all_future.flatten(), y=pred_fund_all, mode="lines", name="Trend & Forecast",
        line=dict(width=2, dash="dash", color="#8B5CF6")
    ))
    upper_fund = pred_fund_all + 1.96 * std_fund
    lower_fund = np.maximum(0, pred_fund_all - 1.96 * std_fund)
    fig_fund.add_trace(go.Scatter(
        x=np.concatenate([all_future.flatten(), all_future.flatten()[::-1]]),
        y=np.concatenate([upper_fund, lower_fund[::-1]]),
        fill="toself",
        fillcolor="rgba(139, 92, 246, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Band",
        hoverinfo="skip"
    ))
    fig_fund.update_layout(
        title=f"Funding Volume Forecast (MAE: ±${mae_fund:,.1f}M)",
        xaxis=dict(title="Year", tickmode="linear"),
        yaxis=dict(title="Funding ($M)"),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    errors = {
        "mae_deals": round(mae_deals, 2),
        "mae_funding": round(mae_fund, 2),
        "next_year": last_year + 1,
        "next_year_deals": round(pred_deals_future[0], 0),
        "next_year_fund": round(pred_fund_future[0], 1)
    }
    return fig_deals, fig_fund, errors


# -----------------------------------------------------------------------------
# 8. DATA QUALITY TAB
# -----------------------------------------------------------------------------
def plot_data_quality_metrics(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, pd.DataFrame]:
    """
    Visualizes missing value profile per column, undisclosed deal proportions,
    and returns a summary audit table of cleaned entity mappings.
    """
    # 1. Missing values bar chart
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing Rate (%)"] = (missing["Missing Count"] / len(df)) * 100

    fig_missing = px.bar(
        missing,
        x="Column",
        y="Missing Rate (%)",
        text=missing["Missing Rate (%)"].apply(lambda v: f"{v:.1f}%"),
        title="Missing Values Rate per Schema Attribute",
        color="Missing Rate (%)",
        color_continuous_scale="Reds"
    )
    fig_missing.update_layout(height=360, yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=40, b=20))

    # 2. Undisclosed amounts per year
    undisc_yr = df.groupby("year")["amount_disclosed"].agg(
        total="count",
        undisclosed=lambda s: (s == False).sum()
    ).reset_index()
    undisc_yr["Undisclosed Rate (%)"] = (undisc_yr["undisclosed"] / undisc_yr["total"]) * 100

    fig_undisc = px.line(
        undisc_yr,
        x="year",
        y="Undisclosed Rate (%)",
        markers=True,
        title="Proportion of Deals with Undisclosed Amounts Across Years",
        color_discrete_sequence=["#DC2626"]
    )
    fig_undisc.update_layout(height=360, yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=40, b=20))

    # 3. Before/After Name Normalization Table
    merge_audit = [
        {"Category": "City Location", "Raw Ingestion Variants": "Bangalore, Bengaluru, Bangalore / USA", "Unified Standard Form": "Bengaluru", "Rationale": "Synonym unification & overseas stripping"},
        {"Category": "City Location", "Raw Ingestion Variants": "Gurgaon, Gurugram, Gurgaon / New Delhi", "Unified Standard Form": "Gurugram (Delhi NCR)", "Rationale": "Official government rename & NCR clustering"},
        {"Category": "City Location", "Raw Ingestion Variants": "New Delhi, Delhi", "Unified Standard Form": "Delhi (Delhi NCR)", "Rationale": "Capital territory unification"},
        {"Category": "City Location", "Raw Ingestion Variants": "Bombay, Mumbai, Mumbai / USA", "Unified Standard Form": "Mumbai", "Rationale": "Historical synonym normalization"},
        {"Category": "Industry Vertical", "Raw Ingestion Variants": "eCommerce, E-Commerce, ecommerce, ECommerce", "Unified Standard Form": "E-Commerce", "Rationale": "Case normalization & hyphen standardization"},
        {"Category": "Industry Vertical", "Raw Ingestion Variants": "Fin-Tech, FinTech, fintech, banking, lending", "Unified Standard Form": "FinTech", "Rationale": "Financial technology domain aggregation"},
        {"Category": "Industry Vertical", "Raw Ingestion Variants": "Ed-Tech, EdTech, education, e-learning", "Unified Standard Form": "EdTech", "Rationale": "Educational technology consolidation"},
        {"Category": "Investment Stage", "Raw Ingestion Variants": "Seed Funding, Seed / Angel Funding, Seed / Angle", "Unified Standard Form": "Seed", "Rationale": "Typo correction & stage consolidation"},
        {"Category": "Investment Stage", "Raw Ingestion Variants": "Series C, D, E, F, G, H", "Unified Standard Form": "Series C+", "Rationale": "Late-stage growth round grouping"}
    ]
    audit_df = pd.DataFrame(merge_audit)
    return fig_missing, fig_undisc, audit_df


# -----------------------------------------------------------------------------
# 9. STANDALONE EXECUTIVE HTML REPORT EXPORTER
# -----------------------------------------------------------------------------
def generate_executive_html_report(df: pd.DataFrame, kpis: Dict, insights: List[str]) -> str:
    """Generates a self-contained, publication-ready HTML summary report."""
    rows_sample = df.head(10).to_html(classes="table", index=False)
    insights_html = "".join([f"<li>{ins}</li>" for ins in insights])
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Indian Startup Funding Executive Summary</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #1e293b; }}
        h1 {{ color: #1e3a8a; font-size: 24px; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
        h2 {{ color: #1e40af; font-size: 18px; margin-top: 25px; }}
        .kpi-container {{ display: flex; gap: 20px; margin: 20px 0; }}
        .kpi-card {{ flex: 1; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px; text-align: center; }}
        .kpi-title {{ font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; }}
        .kpi-val {{ font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 5px; }}
        .kpi-sub {{ font-size: 11px; color: #10b981; }}
        ul {{ line-height: 1.7; font-size: 14px; }}
        .table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 15px; }}
        .table th, .table td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; }}
        .table th {{ background: #f1f5f9; }}
        @media print {{ body {{ margin: 15mm; }} }}
    </style>
</head>
<body>
    <h1>🚀 Indian Startup Funding Trends: Executive Report</h1>
    <p><em>Generated from the Interactive Data Visualization (IDV) Capstone Analytics Engine.</em></p>
    
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Total Capital Infused</div>
            <div class="kpi-val">${kpis['funding']:,.1f}M</div>
            <div class="kpi-sub">${kpis['funding']/1000:,.2f}B USD</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-val">{kpis['deals']:,}</div>
            <div class="kpi-sub">{kpis['disclosed']:,} Disclosed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Average Deal Size</div>
            <div class="kpi-val">${kpis['avg_deal']:,.2f}M</div>
            <div class="kpi-sub">Median: ${kpis['med_deal']:,.2f}M</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Active Investors</div>
            <div class="kpi-val">{kpis['investors']:,}</div>
            <div class="kpi-sub">Identified Syndicates</div>
        </div>
    </div>

    <h2>Core Analytical Findings</h2>
    <ul>{insights_html}</ul>

    <h2>Recent Filtered Transactions (Sample)</h2>
    {rows_sample}
</body>
</html>
"""
    return html
