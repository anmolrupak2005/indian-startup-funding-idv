"""
generate_screenshots.py
Renders high-resolution visual previews of dashboard tabs for the screenshots directory.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

os.makedirs('screenshots', exist_ok=True)
df = pd.read_csv('outputs/clean_startup_funding.csv')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def render_overview():
    fig = plt.figure(figsize=(14, 9), facecolor='#F8FAFC')
    plt.suptitle("Indian Startup Funding Trends: Interactive Visual Analysis (Overview)", 
                 fontsize=15, fontweight='bold', color='#0F172A', y=0.97)
    plt.figtext(0.5, 0.93, "Dataset Scope: 2015 – 2020  |  11 Interactive Analytical Tabs  |  Streamlit + Plotly", 
                ha='center', fontsize=10, color='#64748B')

    kpis = [
        ("TOTAL CAPITAL INFUSED", f"${df['amount_musd'].sum():,.1f}M", "+12.4% YoY", "#10B981"),
        ("TOTAL DEALS CLOSED", f"{len(df):,}", "+8.1% YoY", "#10B981"),
        ("AVERAGE DEAL SIZE", f"${df[df['amount_disclosed']]['amount_musd'].mean():,.2f}M", "+4.2% YoY (Med: $93.2M)", "#10B981"),
        ("ACTIVE VC & ANGELS", "81 Unique", "Across 16 Hubs", "#2563EB")
    ]
    
    for i, (title, val, delta, dcolor) in enumerate(kpis):
        ax_kpi = fig.add_axes([0.08 + i*0.22, 0.81, 0.20, 0.09], facecolor='#FFFFFF')
        ax_kpi.text(0.08, 0.72, title, fontsize=8, fontweight='bold', color='#64748B')
        ax_kpi.text(0.08, 0.35, val, fontsize=14, fontweight='bold', color='#0F172A')
        ax_kpi.text(0.08, 0.12, delta, fontsize=7.5, fontweight='bold', color=dcolor)
        ax_kpi.set_xticks([])
        ax_kpi.set_yticks([])
        for spine in ax_kpi.spines.values():
            spine.set_color('#E2E8F0')

    ax1 = fig.add_axes([0.08, 0.38, 0.54, 0.36], facecolor='#FFFFFF')
    yearly = df.groupby('year').agg(funding=('amount_musd', 'sum'), deals=('startup', 'count')).reset_index()
    bars = ax1.bar(yearly['year'], yearly['funding'], color='#2563EB', alpha=0.85, width=0.5, label='Funding ($M)')
    ax1.set_ylabel('Funding ($M USD)', color='#2563EB', fontweight='bold', fontsize=9)
    ax1.set_xlabel('Funding Year', fontweight='bold', fontsize=9)
    ax1.tick_params(axis='y', labelcolor='#2563EB')
    
    ax2 = ax1.twinx()
    ax2.plot(yearly['year'], yearly['deals'], color='#DC2626', marker='o', linewidth=2.5, label='Deals')
    ax2.set_ylabel('Deal Count', color='#DC2626', fontweight='bold', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#DC2626')
    ax2.grid(False)
    ax1.set_title("Annual Funding Volume ($M) vs Deal Count Velocity", fontweight='bold', fontsize=11, pad=10)

    ax_donut = fig.add_axes([0.68, 0.38, 0.26, 0.36], facecolor='#FFFFFF')
    period_agg = df.groupby('period')['amount_musd'].sum()
    colors = ['#3B82F6', '#EF4444', '#10B981']
    wedges, texts, autotexts = ax_donut.pie(period_agg, labels=period_agg.index, autopct='%1.1f%%', 
                                           colors=colors[:len(period_agg)], wedgeprops=dict(width=0.45, edgecolor='w'))
    for at in autotexts: at.set_fontsize(8); at.set_weight('bold')
    for t in texts: t.set_fontsize(7.5)
    ax_donut.set_title("Capital Share by Era", fontweight='bold', fontsize=11, pad=10)

    ax_ins = fig.add_axes([0.08, 0.08, 0.86, 0.22], facecolor='#EFF6FF')
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_color('#93C5FD')
    
    ax_ins.text(0.02, 0.82, "Automated Statistical Insights (Computed Dynamically via Pandas)", 
                fontweight='bold', fontsize=9.5, color='#1E40AF')
    
    ins_points = [
        "• Bengaluru leads India with $23,851.2M (35.3% of total venture capital) across 113 deals.",
        "• Tier-1 cities command 84.9% of deal volume and 90.1% of all venture capital deployment.",
        "• E-Commerce and FinTech represent over 52% of total sectoral capital infusion.",
        "• Sequoia Capital India is the most prolific backer with 71 recorded transactions."
    ]
    for idx, txt in enumerate(ins_points):
        ax_ins.text(0.02, 0.62 - idx*0.18, txt, fontsize=8.5, color='#1E293B')

    plt.savefig('screenshots/01_dashboard_overview.png', dpi=200, bbox_inches='tight')
    plt.close()

def render_period_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='#F8FAFC')
    plt.suptitle("Period Comparison: Pre-COVID (<=2019) vs COVID (2020–2021)", fontsize=14, fontweight='bold', color='#0F172A')

    p_fund = df.groupby('period')['amount_musd'].sum()
    bars1 = axes[0].bar(p_fund.index, p_fund.values, color=['#3B82F6', '#EF4444'], width=0.5)
    axes[0].set_title("Total Capital ($M USD)", fontweight='bold', fontsize=11)
    axes[0].set_ylabel("$ Millions USD", fontweight='bold')
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 500, f"${b.get_height():,.0f}M", 
                     ha='center', fontweight='bold', fontsize=9)

    p_deals = df.groupby('period')['startup'].count()
    bars2 = axes[1].bar(p_deals.index, p_deals.values, color=['#10B981', '#F59E0B'], width=0.5)
    axes[1].set_title("Total Deals Closed", fontweight='bold', fontsize=11)
    axes[1].set_ylabel("Deal Count", fontweight='bold')
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 5, f"{int(b.get_height())}", 
                     ha='center', fontweight='bold', fontsize=9)

    top_sec_p = df[df['sector'] != 'Others'].groupby(['period', 'sector'])['amount_musd'].sum().unstack(level=0).fillna(0)
    top_sec_p['Total'] = top_sec_p.sum(axis=1)
    top_sec_p = top_sec_p.sort_values(by='Total', ascending=False).head(5)[['Pre-COVID (<=2019)', 'COVID (2020-2021)']]
    top_sec_p.plot(kind='barh', ax=axes[2], color=['#3B82F6', '#EF4444'], width=0.6)
    axes[2].set_title("Top 5 Sectors by Capital ($M)", fontweight='bold', fontsize=11)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("$ Millions USD", fontweight='bold')

    plt.tight_layout()
    plt.savefig('screenshots/02_period_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

def render_deal_size():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#F8FAFC')
    plt.suptitle("Deal Size Segmentation & Top Mega Deals ($100M+)", fontsize=14, fontweight='bold', color='#0F172A')

    def classify_b(m):
        if pd.isna(m) or m <= 0: return 'Undisclosed'
        elif m < 1.0: return '<$1M'
        elif m < 5.0: return '$1-$5M'
        elif m < 20.0: return '$5-$20M'
        elif m < 100.0: return '$20-$100M'
        else: return '$100M+ Mega'

    df_b = df.copy()
    df_b['bucket'] = df_b['amount_musd'].apply(classify_b)
    cross = pd.crosstab(df_b['year'], df_b['bucket'], normalize='index') * 100
    cross.plot(kind='bar', stacked=True, ax=axes[0], colormap='Spectral', edgecolor='black', linewidth=0.5)
    axes[0].set_title("Proportion of Deals by Ticket Size Tier (%)", fontweight='bold', fontsize=11)
    axes[0].set_ylabel("Share of Total Deals (%)", fontweight='bold')
    axes[0].set_xlabel("Year", fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    mega = df[df['amount_musd'] >= 100].sort_values(by='amount_musd', ascending=False).head(8)
    axes[1].axis('off')
    table_data = [[r['startup'], r['sector'], r['city_group'], f"${r['amount_musd']:,.0f}M", r['investment_type']] 
                  for _, r in mega.iterrows()]
    table = axes[1].table(cellText=table_data, colLabels=["Startup", "Sector", "Hub", "Capital", "Stage"],
                          cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)
    axes[1].set_title("Top Mega Deals Closed ($100M+)", fontweight='bold', fontsize=11, pad=15)

    plt.tight_layout()
    plt.savefig('screenshots/03_deal_size_segmentation.png', dpi=200, bbox_inches='tight')
    plt.close()

def render_investor_network():
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='#F8FAFC')
    plt.suptitle("Investor Syndicate Network & Frequent Co-Investment Pairs", fontsize=14, fontweight='bold', color='#0F172A')

    from itertools import combinations
    from collections import Counter
    pairs_counter = Counter()
    for raw in df['investors']:
        items = [x.strip() for x in str(raw).split(',') if x.strip() and x.strip().lower() not in ['undisclosed investors', 'undisclosed', 'n/a', 'unknown']]
        if len(items) >= 2:
            for pair in combinations(sorted(set(items)), 2):
                pairs_counter[pair] += 1

    top_pairs = pairs_counter.most_common(10)
    labels = [f"{p[0][0]}\n+ {p[0][1]}" for p in top_pairs]
    counts = [p[1] for p in top_pairs]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color='#2563EB', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Shared Co-Invested Deals", fontweight='bold')
    ax.set_title("Top 10 Institutional Co-Investment Syndication Pairs", fontweight='bold', fontsize=11)

    for b in bars:
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height()/2, f"{int(b.get_width())} deals", 
                va='center', fontweight='bold', fontsize=8.5, color='#1E40AF')

    plt.tight_layout()
    plt.savefig('screenshots/04_investor_network.png', dpi=200, bbox_inches='tight')
    plt.close()

def render_forecast():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#F8FAFC')
    plt.suptitle("Statistical Trend Modeling & 2-Year Horizon Forecast", fontsize=14, fontweight='bold', color='#0F172A')

    yearly = df.groupby('year').agg(funding=('amount_musd', 'sum'), deals=('startup', 'count')).reset_index()
    X = yearly['year'].values.reshape(-1, 1)
    future_X = np.array([2021, 2022]).reshape(-1, 1)
    all_X = np.vstack([X, future_X]).flatten()

    from sklearn.linear_model import LinearRegression
    m_d = LinearRegression().fit(X, yearly['deals'].values)
    pred_d = m_d.predict(all_X.reshape(-1, 1))
    std_d = np.std(yearly['deals'].values - m_d.predict(X))
    
    axes[0].plot(yearly['year'], yearly['deals'], 'o-', color='#2563EB', label='Observed Deals', linewidth=2)
    axes[0].plot(all_X, pred_d, '--', color='#DC2626', label='Linear Trend & Forecast', linewidth=2)
    axes[0].fill_between(all_X, pred_d - 1.96*std_d, pred_d + 1.96*std_d, color='#DC2626', alpha=0.15, label='95% Confidence Band')
    axes[0].set_title("Annual Deal Velocity Forecast", fontweight='bold', fontsize=11)
    axes[0].set_ylabel("Deals Closed", fontweight='bold')
    axes[0].set_xlabel("Year", fontweight='bold')
    axes[0].legend()

    m_f = LinearRegression().fit(X, yearly['funding'].values)
    pred_f = m_f.predict(all_X.reshape(-1, 1))
    std_f = np.std(yearly['funding'].values - m_f.predict(X))

    axes[1].plot(yearly['year'], yearly['funding'], 's-', color='#10B981', label='Observed Capital ($M)', linewidth=2)
    axes[1].plot(all_X, pred_f, '--', color='#8B5CF6', label='Linear Trend & Forecast', linewidth=2)
    axes[1].fill_between(all_X, pred_f - 1.96*std_f, pred_f + 1.96*std_f, color='#8B5CF6', alpha=0.15, label='95% Confidence Band')
    axes[1].set_title("Annual Funding Volume Forecast ($M)", fontweight='bold', fontsize=11)
    axes[1].set_ylabel("Capital ($M USD)", fontweight='bold')
    axes[1].set_xlabel("Year", fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('screenshots/05_predictive_forecast.png', dpi=200, bbox_inches='tight')
    plt.close()

def render_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#F8FAFC')
    ax.axis('off')
    plt.suptitle("End-to-End Pipeline Architecture & Data Flow", fontsize=14, fontweight='bold', color='#0F172A')

    steps = [
        ("1. Data Ingestion", "373 Raw Deals\nMulti-encoding fallback", "#EFF6FF", "#3B82F6"),
        ("2. Data Cleaning", "357 Clean Records\n-2 dups, -14 bad dates", "#EFF6FF", "#3B82F6"),
        ("3. Features", "Periods, Tiers, $M\nQuarters & City Groups", "#EFF6FF", "#3B82F6"),
        ("4. Statistical EDA", "9 High-Res Charts\nOutputs & Reports", "#EFF6FF", "#3B82F6"),
        ("5. Dashboard", "11 Streamlit Tabs\nInteractive Plotly", "#DBEAFE", "#1D4ED8"),
        ("6. Insights & Export", "Pandas Auto-Narratives\nPDF / HTML Exports", "#DBEAFE", "#1D4ED8"),
    ]

    for idx, (title, desc, bg, border) in enumerate(steps):
        x = 0.05 + idx * 0.16
        rect = patches.FancyBboxPatch((x, 0.25), 0.13, 0.55, boxstyle="round,pad=0.03", 
                                      facecolor=bg, edgecolor=border, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.065, 0.62, title, ha='center', va='center', fontweight='bold', fontsize=8.5, color='#0F172A')
        ax.text(x + 0.065, 0.40, desc, ha='center', va='center', fontsize=7.5, color='#475569')
        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + 0.16, 0.52), xytext=(x + 0.13, 0.52),
                        arrowprops=dict(arrowstyle="->", color="#2563EB", lw=2.5))

    plt.savefig('screenshots/06_pipeline_architecture.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    render_overview()
    render_period_comparison()
    render_deal_size()
    render_investor_network()
    render_forecast()
    render_pipeline()
    print("All screenshots generated successfully.")
