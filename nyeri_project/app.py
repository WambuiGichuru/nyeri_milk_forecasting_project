"""
CRISP-DM PHASE 6 — Deployment: Interactive Streamlit Dashboard

app.py — Nyeri County Milk Production Forecasting Dashboard

DESIGN PRINCIPLES ("The Big Book of Dashboards" — Wexler, Shaffer, Cotgreave)

1. Answer the most important question first — KPI cards at top of every page
2. Context always visible — data source, simulation label, date range shown
3. Reference lines — trend lines, averages, thresholds on every chart
4. Clear visual hierarchy — colour guides the eye to what matters
5. Reduce chartjunk — no 3D, no excessive gridlines, clean axes
6. Allow comparison — models shown side-by-side, not separately
7. Uncertainty must be shown — confidence intervals on all forecasts
8. Label directly — annotations on chart, not just legend

PAGES:
  Overview        — KPI cards + production timeline with annotations
  Observed Data   — Trends, cattle population, sub-county breakdown
  Simulated Data  — Monthly series + validation table
  Model Forecast  — Side-by-side forecast comparison on test set
  Model Ranking   — Metrics table + qualitative tradeoff analysis
  Recommendations — Policy recommendation with data collection milestones
  Methodology     — CRISP-DM flow + simulation transparency

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Page configuration
st.set_page_config(
    page_title="Nyeri Milk Forecasting",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# GLOBAL STYLES
# Design language: deep agricultural green + warm amber + clean cream
# Typography: IBM Plex Serif (headlines) + IBM Plex Sans (body)
# Inspired by "The Big Book of Dashboards" — clarity over decoration


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #F7F5F0;
    color: #1A1A1A;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D3B2E !important;
    border-right: 3px solid #C8922A;
}
[data-testid="stSidebar"] * {
    color: #E8DFC8 !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem;
    padding: 6px 0;
}
[data-testid="stSidebar"] hr {
    border-color: #C8922A55;
}

/* ── Main page header ── */
.page-title {
    font-family: 'IBM Plex Serif', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #C8922A;
    border-bottom: 3px solid #C8922A;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}
.page-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem;
    color: #556B5A;
    margin-top: -0.8rem;
    margin-bottom: 1.5rem;
}

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF;
    border-left: 5px solid #0D3B2E;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    height: 100%;
}
.kpi-card.amber { border-left-color: #C8922A; }
.kpi-card.teal  { border-left-color: #1A6B55; }
.kpi-card.red   { border-left-color: #C0392B; }
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7A8B7E;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #0D3B2E;
    line-height: 1.1;
}
.kpi-delta {
    font-size: 0.82rem;
    color: #556B5A;
    margin-top: 0.25rem;
}
.kpi-delta.positive { color: #1A6B55; }
.kpi-delta.negative { color: #C0392B; }

/* ── Section headings ── */
.section-head {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #C8922A;
    margin-top: 1.8rem;
    margin-bottom: 0.5rem;
    padding-left: 0.5rem;
    border-left: 3px solid #C8922A;
}

/* ── Data notice banners ── */
.simulated-banner {
    background: #FFF8E7;
    border: 1.5px solid #C8922A;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: #7A5A00;
    margin-bottom: 1rem;
}
.observed-banner {
    background: #EFF7F2;
    border: 1.5px solid #1A6B55;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: #0D3B2E;
    margin-bottom: 1rem;
}

/* ── Tables ── */
.dataframe { font-size: 0.85rem !important; }

/* ── Metric comparison table ── */
.metric-table {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

/* ── Footer ── */
.dashboard-footer {
    text-align: center;
    font-size: 0.75rem;
    color: #AAB5AA;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #DDD;
}
</style>
""", unsafe_allow_html=True)

#Colour palette (consistent across all charts)
COLOURS = {
    "observed":   "#0D3B2E",   # Deep green
    "simulated":  "#1A6B55",   # Medium green
    "arima":      "#C8922A",   # Amber
    "sarima":     "#2E86AB",   # Teal blue
    "prophet":    "#8B2FC9",   # Purple
    "actual":     "#1A1A1A",   # Near black
    "ci_arima":   "rgba(200,146,42,0.15)",
    "ci_sarima":  "rgba(46,134,171,0.15)",
    "ci_prophet": "rgba(139,47,201,0.15)",
    "reference":  "#C0392B",   # Reference line red
    "grid":       "#E8E4DC",
    "background": "#F7F5F0",
}

PLOTLY_LAYOUT = dict(
    font_family     = "IBM Plex Sans",
    font            = dict(color="#1A1A1A"),
    paper_bgcolor   = "#F7F5F0",
    plot_bgcolor    = "#FFFFFF",
    margin          = dict(t=50, b=40, l=50, r=20),
    xaxis = dict(gridcolor=COLOURS["grid"], linecolor="#CCCCCC", zeroline=False, title_font=dict(color="#1A1A1A")),
    yaxis = dict(gridcolor=COLOURS["grid"], linecolor="#CCCCCC", zeroline=False,title_font=dict(color="#1A1A1A")),
    legend = dict(
        bgcolor     = "rgba(255,255,255,0.85)",
        bordercolor = "#DDDDDD",
        borderwidth = 1,
        font_size   = 11,
        font        = dict(color="#1A1A1A"),
    ),
    hoverlabel = dict(bgcolor="white",font_color="#1A1A1A", font_size=12, font_family="IBM Plex Sans"),
)

# DATA LOADING
@st.cache_data
def load_all_data():
    """Load all datasets. Returns None values if files missing."""
    data = {}

    def safe_read(path, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            return None

    data["annual"]    = safe_read("data/01_county_annual.csv")
    data["subcounty"] = safe_read("data/02_subcounty_population.csv")
    data["monthly"]   = safe_read("data/03_simulated_monthly.csv", parse_dates=["date"])
    data["results"]   = safe_read("data/model_results_monthly.csv", parse_dates=["date"])
    data["annual_results"] = safe_read("data/model_results_annual.csv")
    data["comparison"]     = safe_read("data/model_comparison.csv")
    data["metadata"]       = safe_read("data/model_metadata.csv")

    return data


def data_missing_warning(name: str):
    st.error(
        f"**{name}** not found. Run the pipeline scripts first:\n\n"
        "```\npython 01_data_preparation.py\n"
        "python 02_simulate.py\n"
        "python 04_models.py\n"
        "python 05_evaluate.py\n```"
    )


# HELPER COMPONENTS
def kpi_card(label: str, value: str, delta: str = "", delta_positive: bool = True,
            card_class: str = "") -> str:
    delta_class = "positive" if delta_positive else "negative"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {card_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""


def section_heading(text: str):
    st.markdown(f'<div class="section-head">{text}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <div style="font-family:'IBM Plex Serif',serif; font-size:1.2rem; font-weight:700; color:#C8922A; line-height:1.2;">
            Nyeri County<br>Milk Forecasting
        </div>
        <div style="font-size:0.75rem; color:#9AADA0; margin-top:0.3rem;">
            JKUAT Final Year Project
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Overview",
            "📈  Observed Data",
            "🔬  Simulated Data",
            "🔮  Model Forecasts",
            "🏆  Model Ranking",
            "📋  Recommendations",
            "ℹ️  Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#7A9088; line-height:1.6;">
        <b>Student:</b> Faith Wambui Gichuru<br>
        <b>Reg:</b> SCT213-C002-0003/2022<br>
        <b>Supervisor:</b> Dr Fanon Ananda<br>
        <b>Institution:</b> JKUAT<br>
        <b>Framework:</b> CRISP-DM
    </div>
    """, unsafe_allow_html=True)

# Load data
D = load_all_data()

# PAGE 1: OVERVIEW
if "Overview" in page:
    page_header(
        "Nyeri County Milk Production Forecasting",
        "A comparative study of ARIMA, SARIMA, and Prophet on observed and simulated data"
    )

    #KPI cards
    if D["annual"] is not None:
        df = D["annual"]
        latest_prod  = df["total_milk_production_litres"].iloc[-1]
        prev_prod    = df["total_milk_production_litres"].iloc[-2]
        growth       = (latest_prod - prev_prod) / prev_prod * 100
        latest_cattle = df["dairy_cattle_population"].iloc[-1]
        latest_ppc   = df["avg_production_per_cow_litres"].iloc[-1]
        n_sim        = len(D["monthly"]) if D["monthly"] is not None else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(kpi_card(
            "Latest Annual Production",
            f"{latest_prod/1e6:.1f}M litres",
            f"{'↑' if growth > 0 else '↓'} {abs(growth):.1f}% vs prior year",
            growth > 0
        ), unsafe_allow_html=True)
        col2.markdown(kpi_card(
            "Dairy Cattle Population",
            f"{latest_cattle:,.0f}",
            "2024/2025 count",
            True, "teal"
        ), unsafe_allow_html=True)
        col3.markdown(kpi_card(
            "Avg Yield Per Cow",
            f"{latest_ppc} L/yr",
            "Up from 5 L/yr in 2016",
            True, "amber"
        ), unsafe_allow_html=True)
        col4.markdown(kpi_card(
            "Simulated Monthly Records",
            f"{n_sim}",
            "9 yrs × 12 months (Denton-Cholette)",
            True
        ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        #Main timeline chart with annotations
        section_heading("Nyeri County Annual Milk Production — 2016 to 2025")

        avg_prod = df["total_milk_production_litres"].mean()

        fig = go.Figure()

        # Area fill under line
        fig.add_trace(go.Scatter(
            x=df["year_label"], y=df["total_milk_production_litres"],
            fill="tozeroy",
            fillcolor="rgba(13,59,46,0.08)",
            line=dict(color=COLOURS["observed"], width=3),
            mode="lines+markers",
            marker=dict(size=9, color=COLOURS["observed"], symbol="circle"),
            name="Annual Production",
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} litres<extra></extra>",
        ))

        # Average reference line ("The Big Book" principle: always show reference)
        fig.add_hline(
            y=avg_prod,
            line_dash="dot",
            line_color=COLOURS["reference"],
            line_width=1.5,
            annotation_text=f"9-yr avg: {avg_prod/1e6:.1f}M litres",
            annotation_position="top right",
            annotation_font_color=COLOURS["reference"],
            annotation_font_size=11,
        )

        # Annotations for notable events (direct labelling — "Big Book" principle)
        annotations = [
            dict(x="2017/2018", y=93_826_201, text="▼ 2017 dip",  showarrow=True, arrowhead=2, ay=-35),
            dict(x="2020/2021", y=111_727_403, text="COVID-19",   showarrow=True, arrowhead=2, ay=35),
            dict(x="2024/2025", y=121_031_160, text="▲ Record high", showarrow=True, arrowhead=2, ay=-40),
        ]
        for ann in annotations:
            ann.update(font=dict(size=10, color="#556B5A"), arrowcolor="#556B5A")
            fig.add_annotation(**ann)

        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            showlegend=True,
            yaxis_title="Litres",
            xaxis_title="Year",
            title=None,
            
        )
        fig.update_xaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig.update_yaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig.update_layout(
            legend=dict(
                font=dict(color="#1A1A1A", size=11),
                title_font=dict(color="#1A1A1A", size=11),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#DDDDDD",
                borderwidth=1
            )
        )
        st.plotly_chart(fig, width='stretch')

    # Project summary
    section_heading("About This Project")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        This dashboard presents the complete findings of a JKUAT Data Science final-year project
        for **Nyeri County Government**.

        **The core problem:** With only 9 annual data points, no forecasting model can produce
        reliable predictions. This project demonstrates that limitation rigorously, simulates
        what monthly data would look like, and compares ARIMA, SARIMA, and Prophet on both datasets.

        **The output:** A concrete, evidence-based recommendation to the county on what to start
        recording and when forecasting becomes operationally viable.
        """)
    with col_b:
        st.markdown("""
        | Phase | Status |
        |-------|--------|
        | Data Preparation | ✅ Done |
        | Simulation (Denton-Cholette) | ✅ Done |
        | EDA | ✅ Done |
        | ARIMA / SARIMA / Prophet | ✅ Done |
        | Evaluation | ✅ Done |
        | Dashboard | ✅ Live |
        """)


# PAGE 2: OBSERVED DATA
elif "Observed" in page:
    page_header(
        "Observed County Data",
        "Official records from Nyeri County Department of Livestock — 9 annual observations"
    )
    st.markdown('<div class="observed-banner">🟢 <b>Observed Data</b> — These are official Nyeri County administrative records. All values are as provided by the Directorate of Livestock.</div>', unsafe_allow_html=True)

    if D["annual"] is None:
        data_missing_warning("data/01_county_annual.csv")
        st.stop()

    df = D["annual"]

    #KPI strip
    col1, col2, col3 = st.columns(3)
    col1.markdown(kpi_card("Years of Data", "9", "2016/17 – 2024/25"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Total Production (9yr)", f"{df['total_milk_production_litres'].sum()/1e9:.2f}B litres", "", True, "amber"), unsafe_allow_html=True)
    col3.markdown(kpi_card("Productivity Gain", "+60%", "Avg yield/cow: 5L → 8L/year", True, "teal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    #Production vs cattle (dual axis — comparison)
    with col_left:
        section_heading("Production vs Cattle Population")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=df["year_label"], y=df["dairy_cattle_population"],
            name="Cattle Population",
            marker_color="rgba(26,107,85,0.3)",
            marker_line_color=COLOURS["simulated"],
            marker_line_width=1.5,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["year_label"], y=df["total_milk_production_litres"],
            name="Milk Production",
            mode="lines+markers",
            line=dict(color=COLOURS["observed"], width=3),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} L<extra></extra>",
        ), secondary_y=True)
        
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        fig.update_layout(legend=dict(x=0.02, y=0.98))
        
        # Axis titles and dark styling
        fig.update_xaxes(
            title_text="Year",
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig.update_yaxes(
            title_text="Cattle (head)", secondary_y=False,
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig.update_yaxes(
            title_text="Litres", secondary_y=True,
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        # Ensure legend text is dark
        fig.update_layout(legend=dict(font=dict(color="#1A1A1A", size=11)))
        
        st.plotly_chart(fig, width='stretch')

    #Productivity per cow
    with col_right:
        section_heading("Average Yield Per Cow (Litres/Year)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["year_label"], y=df["avg_production_per_cow_litres"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(200,146,42,0.12)",
            line=dict(color=COLOURS["arima"], width=3),
            marker=dict(size=10, color=COLOURS["arima"]),
            hovertemplate="<b>%{x}</b><br>%{y} L/cow/yr<extra></extra>",
        ))
        fig2.add_hline(y=10, line_dash="dash", line_color=COLOURS["reference"],
                    annotation_text="Target: 10 L/cow/yr", annotation_font_size=10,
                    annotation_font_color=COLOURS["reference"])
        
        fig2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                        yaxis_title="Litres / Cow / Year")
        
        # Add x-axis title and dark styling
        fig2.update_xaxes(
            title_text="Year",
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig2.update_yaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        
        st.plotly_chart(fig2, width='stretch')

    # ── Sub-county breakdown ───────────────────────────────────────────────────
    if D["subcounty"] is not None:
        section_heading("Sub-County Cattle Population (2020–2025)")
        df_sub = D["subcounty"]
        subcounty_cols = ["kieni_east","mathira_east","mathira_west","mukurwe_ini",
                        "kieni_west","nyeri_central","nyeri_south","tetu"]
        df_melt = df_sub.melt(
            id_vars="year_label", value_vars=subcounty_cols,
            var_name="Sub-County", value_name="Cattle Population"
        )
        df_melt["Sub-County"] = df_melt["Sub-County"].str.replace("_", " ").str.title()

        fig3 = px.line(df_melt, x="year_label", y="Cattle Population",
                    color="Sub-County",
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Dark2)
        fig3.update_layout(**PLOTLY_LAYOUT, height=340,
                        xaxis_title="Year", yaxis_title="Cattle (head)")
        
        # Explicit axis styling for dark text
        fig3.update_xaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig3.update_yaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        # Legend text dark
        fig3.update_layout(legend=dict(font=dict(color="#1A1A1A", size=11)))
        
        st.plotly_chart(fig3, width='stretch')

        # Latest year bar chart
        latest_sub = df_sub[df_sub["year"] == df_sub["year"].max()][subcounty_cols].iloc[0]
        latest_sub = latest_sub.sort_values(ascending=True)
        fig4 = go.Figure(go.Bar(
            x=latest_sub.values,
            y=[sc.replace("_", " ").title() for sc in latest_sub.index],
            orientation="h",
            marker_color=COLOURS["simulated"],
            text=[f"{v:,.0f}" for v in latest_sub.values],
            textposition="outside",
        ))
        avg_sub = latest_sub.mean()
        fig4.add_vline(x=avg_sub, line_dash="dot", line_color=COLOURS["reference"],
                    annotation_text=f"Avg: {avg_sub:,.0f}", annotation_font_size=10)
        fig4.update_layout(**PLOTLY_LAYOUT, height=280, title="2024/25 Sub-County Ranking",
                        showlegend=False, xaxis_title="Cattle (head)")
        
        # Axis styling (y-axis has sub-county names, no title needed)
        fig4.update_xaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig4.update_yaxes(
            tickfont=dict(color="#1A1A1A", size=11)
        )
        
        st.plotly_chart(fig4, width='stretch')

    # Raw data table
    with st.expander("📄 View Raw Observed Data Table"):
        display_cols = ["year_label","total_milk_production_litres",
                        "dairy_cattle_population","avg_production_per_cow_litres",
                        "yoy_production_growth_pct"]
        st.dataframe(df[display_cols].rename(columns={
            "year_label": "Year",
            "total_milk_production_litres": "Production (L)",
            "dairy_cattle_population": "Cattle",
            "avg_production_per_cow_litres": "Yield/Cow (L)",
            "yoy_production_growth_pct": "YoY Growth %",
        }), width='stretch')


# PAGE 3: SIMULATED DATA
elif "Simulated" in page:
    page_header(
        "Simulated Monthly Dataset",
        "Denton-Cholette temporal disaggregation — 108 monthly estimates derived from official annual totals"
    )
    st.markdown('<div class="simulated-banner">⚠️ <b>Simulated Data</b> — These monthly values are statistically generated from the 9 official annual totals using the Denton-Cholette method. They are NOT independently observed. Every value is mathematically anchored to an official county record.</div>', unsafe_allow_html=True)

    if D["monthly"] is None:
        data_missing_warning("data/03_simulated_monthly.csv")
        st.stop()

    df_m = D["monthly"]
    df_a = D["annual"]

    #KPI strip
    col1, col2, col3 = st.columns(3)
    col1.markdown(kpi_card("Monthly Records", "108", "9 years × 12 months"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Peak Month (avg)", "November", "Short rains season", True, "amber"), unsafe_allow_html=True)
    col3.markdown(kpi_card("Trough Month (avg)", "February", "Long dry season", False, "red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main simulated monthly chart 
    section_heading("Monthly Production Series with Annual Actual Overlaid")
    fig = go.Figure()

    # Simulated monthly line
    fig.add_trace(go.Scatter(
        x=df_m["date"], y=df_m["monthly_production_litres"],
        mode="lines",
        name="Simulated Monthly",
        line=dict(color=COLOURS["simulated"], width=1.5),
        opacity=0.85,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} L<extra></extra>",
    ))

    # Overlay annual totals as scatter points
    if df_a is not None:
        annual_monthly_avg = df_a["total_milk_production_litres"] / 12
        annual_dates = pd.to_datetime([f"{int(y)}-07-01" for y in df_a["year"]])
        fig.add_trace(go.Scatter(
            x=annual_dates, y=annual_monthly_avg,
            mode="markers",
            name="Official Annual ÷ 12",
            marker=dict(size=12, color=COLOURS["observed"], symbol="diamond",
                        line=dict(color="white", width=1.5)),
            hovertemplate="<b>%{x|%Y} official avg/month</b><br>%{y:,.0f} L<extra></extra>",
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                    xaxis_title="Date",
                    yaxis_title="Monthly Production (Litres)")
    
    # Explicit dark styling for axes and legend
    fig.update_xaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig.update_yaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig.update_layout(legend=dict(
        font=dict(color="#1A1A1A", size=11),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DDDDDD",
        borderwidth=1,
        x=0.02, y=0.98
    ))
    
    st.plotly_chart(fig, width='stretch')

    #Seasonal pattern chart
    col_l, col_r = st.columns(2)
    with col_l:
        section_heading("Average by Calendar Month (Seasonal Pattern)")
        month_avg = df_m.groupby("month")["monthly_production_litres"].mean().reset_index()
        months_str = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_avg["month_name"] = months_str
        overall_avg = month_avg["monthly_production_litres"].mean()

        colors = [COLOURS["simulated"] if v >= overall_avg else "rgba(26,107,85,0.35)"
                for v in month_avg["monthly_production_litres"]]

        fig2 = go.Figure(go.Bar(
            x=month_avg["month_name"],
            y=month_avg["monthly_production_litres"],
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>Avg: %{y:,.0f} L<extra></extra>",
        ))
        fig2.add_hline(y=overall_avg, line_dash="dot", line_color=COLOURS["reference"],
                    annotation_text="Annual mean / 12", annotation_font_size=10,
                    annotation_font_color=COLOURS["reference"])
        
        fig2.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                        yaxis_title="Average Monthly Production (Litres)")
        
        # Add x-axis title and dark styling
        fig2.update_xaxes(
            title_text="Month",
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        fig2.update_yaxes(
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        )
        
        st.plotly_chart(fig2, width='stretch')

    with col_r:
        # Validation table
        section_heading("Simulation Validation (Annual Sum Check)")
        st.caption("Monthly values must sum back to official annual totals within 1% error.")

        if df_a is not None:
            agg = df_m.groupby("year")["monthly_production_litres"].sum().reset_index()
            agg.columns = ["year", "simulated_sum"]
            merged_val = df_a[["year","year_label","total_milk_production_litres"]].merge(agg, on="year")
            merged_val["error_pct"] = (
                (merged_val["simulated_sum"] - merged_val["total_milk_production_litres"]).abs()
                / merged_val["total_milk_production_litres"] * 100
            ).round(4)
            merged_val["✓"] = merged_val["error_pct"].apply(lambda e: "✅" if e < 1.0 else "❌")

            st.dataframe(merged_val[["year_label","total_milk_production_litres","simulated_sum","error_pct","✓"]].rename(columns={
                "year_label": "Year",
                "total_milk_production_litres": "Official (L)",
                "simulated_sum": "Simulated Sum (L)",
                "error_pct": "Error %",
            }), width='stretch', height=320)

            max_err = merged_val["error_pct"].max()
            if max_err < 1.0:
                st.success(f"✅ All years within 1% threshold. Max error: {max_err:.4f}%")
            else:
                st.error(f"❌ Some years exceed 1%. Max error: {max_err:.4f}%")

# PAGE 4: MODEL FORECASTS
elif "Forecast" in page:
    page_header(
        "Model Forecasts — Test Set Comparison",
        "All 3 models evaluated on 22-month held-out test set (80/20 chronological split)"
    )

    if D["results"] is None:
        data_missing_warning("data/model_results_monthly.csv")
        st.stop()

    df_r = D["results"]
    df_m = D["monthly"]

    #Toggle which models to show
    st.markdown('<div class="section-head">Display Options</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    show_arima   = col1.checkbox("ARIMA",   value=True)
    show_sarima  = col2.checkbox("SARIMA",  value=True)
    show_prophet = col3.checkbox("Prophet", value=True)
    show_ci      = st.checkbox("Show 95% Confidence Intervals", value=True)

    section_heading("Test-Set Forecast vs Actual (Simulated Monthly Data)")
    st.caption("Train period: Jan 2016 – Jun 2023 (86 months) | Test period: Jul 2023 – Dec 2024 (22 months)")

    # Full history + test period chart
    fig = go.Figure()

    # Training history (context — "Big Book" principle: always show context)
    if df_m is not None:
        train_period = df_m[df_m["date"] < df_r["date"].min()]
        fig.add_trace(go.Scatter(
            x=train_period["date"], y=train_period["monthly_production_litres"],
            mode="lines", name="Training Data",
            line=dict(color=COLOURS["observed"], width=1.2),
            opacity=0.5,
        ))

    # Actual test values
    fig.add_trace(go.Scatter(
        x=df_r["date"], y=df_r["actual"],
        mode="lines+markers", name="Actual (Test)",
        line=dict(color=COLOURS["actual"], width=2.5),
        marker=dict(size=6),
    ))

    # Train/test split line
    fig.add_vline(
        x=df_r["date"].min().timestamp() * 1000,
        line_dash="dash", line_color="#888888", line_width=1.5,
        annotation_text="Train | Test", annotation_font_size=10,
    )

    if show_arima:
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(df_r["date"]) + list(df_r["date"][::-1]),
                y=list(df_r["arima_upper_95"]) + list(df_r["arima_lower_95"][::-1]),
                fill="toself", fillcolor=COLOURS["ci_arima"],
                line=dict(color="rgba(0,0,0,0)"), name="ARIMA 95% CI", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=df_r["date"], y=df_r["arima_forecast"],
            mode="lines", name="ARIMA",
            line=dict(color=COLOURS["arima"], width=2, dash="dot"),
        ))

    if show_sarima:
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(df_r["date"]) + list(df_r["date"][::-1]),
                y=list(df_r["sarima_upper_95"]) + list(df_r["sarima_lower_95"][::-1]),
                fill="toself", fillcolor=COLOURS["ci_sarima"],
                line=dict(color="rgba(0,0,0,0)"), name="SARIMA 95% CI", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=df_r["date"], y=df_r["sarima_forecast"],
            mode="lines", name="SARIMA",
            line=dict(color=COLOURS["sarima"], width=2, dash="dash"),
        ))

    if show_prophet:
        if show_ci:
            fig.add_trace(go.Scatter(
                x=list(df_r["date"]) + list(df_r["date"][::-1]),
                y=list(df_r["prophet_upper_95"]) + list(df_r["prophet_lower_95"][::-1]),
                fill="toself", fillcolor=COLOURS["ci_prophet"],
                line=dict(color="rgba(0,0,0,0)"), name="Prophet 95% CI", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=df_r["date"], y=df_r["prophet_forecast"],
            mode="lines", name="Prophet",
            line=dict(color=COLOURS["prophet"], width=2, dash="longdash"),
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                    xaxis_title="Date",
                    yaxis_title="Milk Production (Litres)")
    
    # Explicit dark styling for axes and legend
    fig.update_xaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig.update_yaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig.update_layout(legend=dict(
        font=dict(color="#1A1A1A", size=11),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DDDDDD",
        borderwidth=1
    ))
    
    st.plotly_chart(fig, width='stretch')

    #Residual plot
    section_heading("Forecast Errors (Residuals) — Test Period")
    st.caption("Errors close to zero and without pattern = good model. Systematic drift = model misspecification.")

    fig2 = go.Figure()
    fig2.add_hline(y=0, line_color=COLOURS["reference"], line_width=1.5)

    if show_arima:
        fig2.add_trace(go.Scatter(
            x=df_r["date"], y=df_r["actual"] - df_r["arima_forecast"],
            mode="lines+markers", name="ARIMA Error",
            line=dict(color=COLOURS["arima"], width=1.8),
            marker=dict(size=5),
        ))
    if show_sarima:
        fig2.add_trace(go.Scatter(
            x=df_r["date"], y=df_r["actual"] - df_r["sarima_forecast"],
            mode="lines+markers", name="SARIMA Error",
            line=dict(color=COLOURS["sarima"], width=1.8),
            marker=dict(size=5),
        ))
    if show_prophet:
        fig2.add_trace(go.Scatter(
            x=df_r["date"], y=df_r["actual"] - df_r["prophet_forecast"],
            mode="lines+markers", name="Prophet Error",
            line=dict(color=COLOURS["prophet"], width=1.8),
            marker=dict(size=5),
        ))

    fig2.update_layout(**PLOTLY_LAYOUT, height=280,
                    xaxis_title="Date",
                    yaxis_title="Error (Actual − Forecast, Litres)")
    
    # Explicit dark styling for axes and legend
    fig2.update_xaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig2.update_yaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig2.update_layout(legend=dict(
        font=dict(color="#1A1A1A", size=11),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DDDDDD",
        borderwidth=1
    ))
    
    st.plotly_chart(fig2, width='stretch')


# PAGE 5: MODEL RANKING
elif "Ranking" in page:
    page_header(
        "Model Ranking & Evaluation",
        "Quantitative accuracy metrics + qualitative tradeoff analysis"
    )

    if D["comparison"] is None:
        data_missing_warning("data/model_comparison.csv")
        st.stop()

    df_comp = D["comparison"]

    #Metrics table
    section_heading("Quantitative Accuracy — Monthly Simulated Test Set (22 months)")

    best_idx = df_comp["MAPE"].idxmin()
    col1, col2, col3 = st.columns(3)

    for i, (col, row) in enumerate(zip([col1, col2, col3], df_comp.itertuples())):
        medal  = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        card_c = "" if i == 0 else "amber" if i == 1 else "teal"
        col.markdown(kpi_card(
            f"{medal} {row.model}",
            f"{row.MAPE:.2f}% MAPE",
            f"MAE: {row.MAE:,.0f} L  |  RMSE: {row.RMSE:,.0f} L",
            i == 0,
            card_c,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #Bar chart comparison
    section_heading("Side-by-Side Metric Comparison")
    metrics_long = df_comp.melt(id_vars="model", value_vars=["MAE","RMSE","MAPE"],
                                var_name="Metric", value_name="Value")
    fig = px.bar(
        metrics_long, x="model", y="Value", color="Metric",
        barmode="group",
        color_discrete_map={"MAE": COLOURS["arima"], "RMSE": COLOURS["sarima"], "MAPE": COLOURS["prophet"]},
        text_auto=".2f",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                    xaxis_title="Model",
                    yaxis_title="Error Value")
    
    # Explicit dark styling for axes and legend
    fig.update_xaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig.update_yaxes(
        title_font=dict(color="#1A1A1A", size=12),
        tickfont=dict(color="#1A1A1A", size=11)
    )
    fig.update_layout(legend=dict(
        font=dict(color="#1A1A1A", size=11),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#DDDDDD",
        borderwidth=1
    ))
    
    st.plotly_chart(fig, width='stretch')

    #Qualitative tradeoff table
    section_heading("Qualitative Tradeoff Analysis")
    st.caption("Grounded in: Hansen et al. (2024), Platonova & Popov (2025), Perez-Guerra et al. (2023), Taylor & Letham (2018)")

    tradeoffs = pd.DataFrame([
        {"Criterion": "Interpretability",          "ARIMA": "High",             "SARIMA": "High",                   "Prophet": "Medium"},
        {"Criterion": "Min Data Required",         "ARIMA": "~24 months",       "SARIMA": "~36 months",             "Prophet": "~12 months"},
        {"Criterion": "Seasonality Handling",      "ARIMA": "None",             "SARIMA": "Excellent (m=12)",       "Prophet": "Good (Fourier)"},
        {"Criterion": "Uncertainty Intervals",     "ARIMA": "Confidence (freq)","SARIMA": "Confidence (freq)",      "Prophet": "Credible (Bayesian)"},
        {"Criterion": "Sparse Data Behaviour",     "ARIMA": "Degrades quickly", "SARIMA": "Not applicable <36 pts","Prophet": "Most robust"},
        {"Criterion": "County Staff Usability",    "ARIMA": "Moderate",         "SARIMA": "Low (complex)",          "Prophet": "High (automated)"},
        {"Criterion": "Recommended For",           "ARIMA": "Trend baseline",   "SARIMA": "Full seasonal forecast", "Prophet": "Immediate deployment"},
    ])
    st.dataframe(tradeoffs, width='stretch', hide_index=True)

    #Model metadata
    if D["metadata"] is not None:
        with st.expander("📐 View Fitted Model Orders and Parameters"):
            st.dataframe(D["metadata"], width='stretch')


# PAGE 6: RECOMMENDATIONS
elif "Recommend" in page:
    page_header(
        "Policy Recommendations",
        "Evidence-based guidance for Nyeri County Government on data collection and forecasting"
    )

    #Current situation 
    section_heading("Current Data Situation")
    col1, col2, col3 = st.columns(3)
    col1.markdown(kpi_card("Current Records", "9 annual", "Insufficient for any model", False, "red"), unsafe_allow_html=True)
    col2.markdown(kpi_card("Min for Prophet", "12 months", "Start forecasting in Year 1", True, "amber"), unsafe_allow_html=True)
    col3.markdown(kpi_card("Min for SARIMA", "36 months", "Full seasonal model in Year 3", True, "teal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #Milestone timeline visual
    section_heading("Data Collection Milestones — Forecasting Capability Over Time")

    milestones = pd.DataFrame({
        "Milestone":    ["Start monthly recording", "Prophet viable", "ARIMA reliable", "SARIMA fully operational"],
        "Months":       [0, 12, 24, 36],
        "Capability":   ["No forecasting", "Short-term Prophet forecasts", "ARIMA baseline forecasting", "Full seasonal SARIMA forecasting"],
        "Action":       ["Switch annual→monthly tracking at collection centres",
                        "Deploy Prophet on dashboard; begin short-term planning",
                        "Add ARIMA as validation model",
                        "Full SARIMA seasonal forecasting; peak/trough planning"],
    })

    fig = go.Figure()
    colours_ms = [COLOURS["reference"], COLOURS["prophet"], COLOURS["arima"], COLOURS["sarima"]]
    for i, row in milestones.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Months"]], y=[1],
            mode="markers+text",
            marker=dict(size=22, color=colours_ms[i], symbol="circle"),
            text=[f"Month {row['Months']}<br>{row['Milestone']}"],
            textposition="top center",
            name=row["Milestone"],
            hovertext=row["Action"],
            hoverinfo="text",
        ))

    # Connecting line
    fig.add_trace(go.Scatter(
        x=milestones["Months"], y=[1,1,1,1],
        mode="lines", line=dict(color="#CCCCCC", width=3), showlegend=False,
    ))
    
    fig.update_layout(
        height=220,
        xaxis=dict(
            title="Months of Monthly Recording",
            range=[-3, 40],
            gridcolor=COLOURS["grid"],
            title_font=dict(color="#1A1A1A", size=12),
            tickfont=dict(color="#1A1A1A", size=11)
        ),
        yaxis=dict(visible=False, range=[0.5, 1.8]),
        showlegend=True,
        legend=dict(
            font=dict(color="#1A1A1A", size=11),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#DDDDDD",
            borderwidth=1
        ),
        paper_bgcolor="#F7F5F0",
        plot_bgcolor="#FFFFFF",
        margin=dict(t=50, b=40, l=50, r=20),
    )
    
    st.plotly_chart(fig, width='stretch')
    #Action table
    section_heading("Recommended Actions for the Directorate of Livestock")
    actions = pd.DataFrame([
        {"Priority": "🔴 Immediate", "Action": "Record monthly milk production totals",
         "Detail": "Total litres collected at all major collection centres each calendar month",
         "Enables": "Data foundation for all future forecasting"},
        {"Priority": "🔴 Immediate", "Action": "Record monthly cattle population estimate",
         "Detail": "Even quarterly estimates (with monthly interpolation) improve SARIMAX models",
         "Enables": "Exogenous variable for regression-enhanced forecasting"},
        {"Priority": "🟡 Year 1",    "Action": "Record monthly AI (artificial insemination) uptake",
         "Detail": "Number of semen straws used per month per sub-county",
         "Enables": "Leading indicator for production 9–12 months ahead"},
        {"Priority": "🟡 Year 1",    "Action": "Record monthly rainfall or feed availability index",
         "Detail": "Coordinate with Kenya Met Service for ward-level rainfall data",
         "Enables": "Seasonal calibration for SARIMA; SARIMAX exogenous variable"},
        {"Priority": "🟢 Year 2–3",  "Action": "Deploy ARIMA on dashboard",
         "Detail": "24 months of monthly data makes ARIMA parameter estimation reliable",
         "Enables": "Trend forecasting with confidence intervals"},
        {"Priority": "🟢 Year 3+",   "Action": "Deploy SARIMA as primary forecasting model",
         "Detail": "36 months allows estimation of seasonal parameters at m=12",
         "Enables": "Full seasonal production forecasting; peak/trough planning"},
    ])
    st.dataframe(actions, width='stretch', hide_index=True)

    # What this project showed
    section_heading("What This Study Demonstrates")
    st.info("""
    **The central finding of this project** is that the quality of a forecast is determined
    more by the quality and frequency of data collection than by the sophistication of the
    forecasting algorithm.

    All three models — ARIMA, SARIMA, and Prophet — performed significantly better on the
    108-point simulated monthly dataset than on the 9-point observed annual dataset.
    The simulation shows exactly what those monthly records would look like, and what
    forecasting capability they would unlock.

    The county already collects the right information. The gap is collection frequency.
    Switching from annual to monthly recording costs nothing but a small change in
    administrative practice — yet it unlocks an entirely new category of planning capability.
    """)


# PAGE 7: METHODOLOGY
elif "Method" in page:
    page_header(
        "Methodology & Transparency",
        "CRISP-DM framework — full documentation of methods, assumptions, and limitations"
    )

    # CRISP-DM flow
    section_heading("CRISP-DM Phases")

    phases = [
        ("1. Business Understanding", "Nyeri County has 9 annual records insufficient for forecasting. Goal: demonstrate limitation, simulate monthly data, compare 3 models, provide policy roadmap.", "01_data_preparation.py"),
        ("2. Data Understanding",     "EDA on observed annual data (n=9), sub-county data (5yr×8), and simulated monthly data (n=108). ADF stationarity test. Correlation analysis.", "03_eda.py"),
        ("3. Data Preparation",       "Denton-Cholette temporal disaggregation of annual totals into monthly estimates. Bimodal East African seasonal index applied. 3% Gaussian noise. Validation: <1% annual error.", "02_simulate.py"),
        ("4. Modeling",               "ARIMA (auto_arima, non-seasonal), SARIMA (seasonal m=12), Prophet (multiplicative, yearly seasonality). Applied to both observed (n=9) and simulated (n=108) datasets.", "04_models.py"),
        ("5. Evaluation",             "80/20 chronological train/test split. MAE, RMSE, MAPE on test set. Qualitative tradeoff analysis. Walk-forward validation on annual data.", "05_evaluate.py"),
        ("6. Deployment",             "Streamlit interactive dashboard. 7 pages covering data, forecasts, model comparison, policy recommendations, and methodology documentation.", "app.py"),
    ]

    for phase, description, script in phases:
        with st.expander(f"**{phase}** — `{script}`"):
            st.markdown(description)

    #Simulation transparency
    section_heading("Simulation Methodology — Full Transparency")
    st.markdown("""
    **Method:** Denton-Cholette temporal disaggregation (Dagum & Cholette, 2006)

    **Step 1 — Proportional disaggregation:**
    Each annual total is distributed across 12 months using a normalised seasonal index.
    The proportions sum to 1.0, ensuring monthly values exactly reproduce the annual total.

    **Step 2 — Seasonal index:**
    A bimodal index reflecting Kenya's dairy production cycle driven by the long rains
    (March–May) and short rains (October–November). Values: Jan=0.82, Feb=0.80, Mar=0.95,
    Apr=1.10, May=1.15, Jun=1.05, Jul=0.88, Aug=0.85, Sep=0.92, Oct=1.12, Nov=1.18, Dec=1.00.
    Normalised so mean=1.0.

    **Step 3 — Noise injection:**
    Gaussian noise (CV=3%) added to prevent artificially smooth patterns that would
    inflate model accuracy metrics.

    **Step 4 — Rescaling:**
    Final values rescaled so sum(monthly) = official annual total. Validation threshold: <1% error.

    **What this is not:**
    The simulated data is not independently observed data. It is a principled mathematical
    representation of what monthly county records would look like under consistent collection.
    All charts and tables label it explicitly as simulated.
    """)

    # Limitations
    section_heading("⚠️ Limitations")
    st.warning("""
    1. **Observed data (n=9):** Too small for reliable ARIMA/SARIMA parameter estimation.
       All forecasts from the observed annual data should be interpreted as illustrative only.

    2. **Simulated data:** Not independently observed. Simulation accuracy depends on the
       seasonal index used. A different seasonal assumption would produce different monthly
       distributions (though annual totals would remain unchanged).

    3. **No causal inference:** This study identifies correlations and trends, not causal
       relationships. The impact of specific interventions (AI programmes, feed subsidies)
       cannot be isolated from this data.

    4. **Prototype dashboard:** This application is a proof-of-concept. It has not undergone
       formal user acceptance testing with county government staff.
    """)

    # References
    section_heading("Key References")
    st.markdown("""
    - Dagum, E. B., & Cholette, P. A. (2006). *Benchmarking, Temporal Distribution, and Reconciliation Methods for Time Series.* Springer.
    - Dogar, D., Cicek, A., & Ayyildiz, M. (2024). SARIMA model for milk production in Turkey. *Turkish Journal of Agricultural and Natural Science.* https://doi.org/10.30910/turkjans.1389143
    - Hansen, B. G., Li, Y., Sun, R., & Schei, I. (2024). Forecasting milk delivery to dairy. *Expert Systems With Applications.* https://doi.org/10.1016/j.eswa.2024.123475
    - Kashyap, A. et al. (2023). Cold-supply-chain-integrated production forecasting. *Sustainability, 15*(22). https://doi.org/10.3390/su152216102
    - Martinello, L. M. et al. ARIMA vs ETS for milk production in Brazil. *Rev. Inst. Laticínios.* https://doi.org/10.14295/2238-6416.v76i1.823
    - Peng, Z. (2023). Predictive analytics for raw milk production decisions. https://doi.org/10.11144/javeriana.10554.61819
    - Perez-Guerra, U. H. et al. (2023). SARIMA for dairy cows in Andean highlands. *PLOS ONE, 18.* https://doi.org/10.1371/journal.pone.0288849
    - Platonova, A., & Popov, V. S. (2025). Comparing ARIMA, Prophet, LSTM, GRU. *SIST, 5*(2). https://doi.org/10.47813/2782-2818-2025-5-2-3061-3070
    - Primawati, A. et al. (2023). LSTM and Prophet for limited goat milk data. *IEEE ICON-SONICS.* https://doi.org/10.1109/ICON-SONICS59898.2023.10435067
    - Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician, 72*(1). https://doi.org/10.1080/00031305.2017.1380080
    """)

# Footer
st.markdown("""
<div class="dashboard-footer">
    Nyeri County Milk Production Forecasting Dashboard &nbsp;|&nbsp;
    Faith Wambui Gichuru &nbsp;|&nbsp; SCT213-C002-0003/2022 &nbsp;|&nbsp; JKUAT &nbsp;|&nbsp;
    Supervisor: Dr Fanon Ananda &nbsp;|&nbsp; 2025/2026
</div>
""", unsafe_allow_html=True)
