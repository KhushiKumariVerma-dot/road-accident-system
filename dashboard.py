import datetime
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Import shared project utilities
from utils import (
    load_model_package,
    load_sensor_data,
    save_sensor_reading,
    load_dataset,
    predict_risk,
    explain_risk,
    RISK_MAP
)

# ==========================================
# Page Configuration & Modern Visual Theme
# ==========================================
st.set_page_config(
    page_title="AI Road Accident Risk & Traffic Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Cyberpunk / Glassmorphic CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Hero Gradient Title */
    .hero-title {
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 20px;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        margin-bottom: 16px;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 40px rgba(59, 130, 246, 0.15);
        border-color: rgba(96, 165, 250, 0.3);
    }

    /* Metric Stat Widget */
    .stat-widget {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px 18px;
        position: relative;
        overflow: hidden;
    }
    .stat-widget::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .stat-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .stat-value {
        font-size: 1.65rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        color: #f8fafc;
        line-height: 1.1;
    }
    .stat-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Pulsing Risk Badges */
    .risk-banner-high {
        background: radial-gradient(circle at 10% 20%, rgba(185, 28, 28, 0.85) 0%, rgba(127, 29, 29, 0.95) 90%);
        border: 1px solid #ef4444;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        animation: pulse-red 2.5s infinite;
    }

    .risk-banner-medium {
        background: radial-gradient(circle at 10% 20%, rgba(161, 98, 7, 0.9) 0%, rgba(113, 63, 18, 0.95) 90%);
        border: 1px solid #eab308;
        box-shadow: 0 0 25px rgba(234, 179, 8, 0.35);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .risk-banner-low {
        background: radial-gradient(circle at 10% 20%, rgba(4, 120, 87, 0.85) 0%, rgba(6, 78, 59, 0.95) 90%);
        border: 1px solid #10b981;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
        color: #ffffff;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.6); }
    }

    .tag-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .tag-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .tag-emerald { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 18px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%);
        border: 1px solid rgba(96, 165, 250, 0.5);
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Cached Data Loaders
# ==========================================
@st.cache_resource(show_spinner=False)
def get_model_package():
    return load_model_package()


@st.cache_data(show_spinner=False)
def get_road_dataset():
    df = load_dataset("road_accident")
    if "location_name" not in df.columns:
        from road_data_simulator import assign_landmark
        df["location_name"] = [
            assign_landmark(lat, lon, i)
            for i, (lat, lon) in enumerate(zip(df["latitude"], df["longitude"]))
        ]
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(end=datetime.datetime.now(), periods=len(df), freq="15min")
    return df


@st.cache_data(show_spinner=False)
def get_crash_report_dataset():
    try:
        return load_dataset("crash_report")
    except Exception:
        return None


# ==========================================
# Load Model & Dataset
# ==========================================
try:
    pkg = get_model_package()
    model = pkg["model"]
    weather_map = pkg["weather_map"]
    road_map = pkg["road_map"]
    accuracy = pkg.get("accuracy", 0.87)
    df_road = get_road_dataset()
    if "location_name" not in df_road.columns:
        from road_data_simulator import assign_landmark
        df_road["location_name"] = [
            assign_landmark(lat, lon, i)
            for i, (lat, lon) in enumerate(zip(df_road["latitude"], df_road["longitude"]))
        ]
except Exception as err:
    st.error(f"Critical error loading model or dataset: {err}")
    st.stop()


# ==========================================
# Sidebar: Controls & Live Operations
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=60)
    st.markdown("### Traffic Control Center")

    st.markdown('<span class="tag-chip tag-emerald">● SYSTEM ONLINE (ACTIVE MESH)</span>', unsafe_allow_html=True)
    st.markdown(f"**ML Engine:** XGBoost Classifier (**{accuracy * 100:.1f}% Acc**)")
    st.caption("Active Corridors: **500 Monitored Nodes**")
    st.caption(f"Telemetry Heartbeat: `{datetime.datetime.now().strftime('%H:%M:%S')}`")

    st.divider()

    st.subheader("⚡ Live Actions")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Refresh", use_container_width=True, help="Reload latest state"):
            st.cache_data.clear()
            st.rerun()
    with col_sb2:
        if st.button("📡 New IoT", use_container_width=True, help="Simulate fresh random telemetry packet"):
            save_sensor_reading()
            st.rerun()

    st.divider()

    st.subheader("🔍 Network Filters")
    min_date = pd.to_datetime(df_road["timestamp"]).min().date()
    max_date = pd.to_datetime(df_road["timestamp"]).max().date()

    date_filter = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    weather_options = sorted(list(df_road["weather_condition"].unique()))
    selected_weather = st.multiselect("Weather Condition", weather_options, default=weather_options)

    road_options = sorted(list(df_road["road_condition"].unique()))
    selected_road = st.multiselect("Road Surface", road_options, default=road_options)

    risk_options = ["Low", "Medium", "High"]
    selected_risk = st.multiselect("Risk Level", risk_options, default=risk_options)

    if isinstance(date_filter, (tuple, list)) and len(date_filter) == 2:
        start_d, end_d = date_filter
    else:
        start_d, end_d = min_date, max_date

    mask_date = (pd.to_datetime(df_road["timestamp"]).dt.date >= start_d) & (pd.to_datetime(df_road["timestamp"]).dt.date <= end_d)
    mask_weather = df_road["weather_condition"].isin(selected_weather)
    mask_road = df_road["road_condition"].isin(selected_road)
    mask_risk = df_road["risk"].isin(selected_risk)

    filtered_df = df_road[mask_date & mask_weather & mask_road & mask_risk]
    st.caption(f"Active Corridors Filtered: **{len(filtered_df)} / {len(df_road)}**")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Export Filtered CSV",
        data=csv_data,
        file_name="filtered_road_accident_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================
# Hero Header
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-title">🚦 Road Accident Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Real-time IoT Edge Telemetry &nbsp;•&nbsp; '
        'XGBoost ML Classification &nbsp;•&nbsp; Delhi-NCR Geospatial Black-Spots &nbsp;•&nbsp; '
        'Automated Safety Interventions</div>',
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown('<div style="text-align: right; padding-top: 10px;">'
                '<span class="tag-chip tag-blue">XGBOOST V3.4</span>&nbsp;'
                '<span class="tag-chip tag-emerald">87.0% TEST ACCURACY</span>'
                '</div>', unsafe_allow_html=True)

st.divider()

# ==========================================
# Navigation Tabs
# ==========================================
tab_live, tab_map, tab_analytics, tab_crashdb, tab_model = st.tabs([
    "📍 Location Search & Real-Time Monitor",
    "🗺️ Geospatial Hotspots & Map",
    "📊 Network Analytics & Correlations",
    "📁 Real-World Crash Database (58K)",
    "🧠 Model Diagnostics & Architecture"
])


# ==========================================
# TAB 1: Location Search & Real-Time Monitor
# ==========================================
with tab_live:
    # Location Search / Evaluation Mode Section
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            data_source = st.radio(
                "Telemetry Source:",
                ["📍 Search Monitored Road Corridor", "📡 Live IoT Edge Broadcast Node"],
                index=0,
                help="Select between evaluating a specific Delhi-NCR road location or pulling from the active edge gateway"
            )

        if data_source == "📍 Search Monitored Road Corridor":
            if "location_name" not in df_road.columns:
                from road_data_simulator import assign_landmark
                df_road["location_name"] = [
                    assign_landmark(lat, lon, i)
                    for i, (lat, lon) in enumerate(zip(df_road["latitude"], df_road["longitude"]))
                ]
            location_names = sorted(df_road["location_name"].unique())
            with col_m2:
                selected_location_name = st.selectbox(
                    "Select or Search Delhi-NCR Road Location:",
                    location_names,
                    index=0,
                    help="Type any keyword (e.g. Ring Road, Connaught, Noida, Cyber City, GT Karnal, Dwarka) to search"
                )

            chosen_row = df_road[df_road["location_name"] == selected_location_name].iloc[0]
            eval_sensor = {
                "location_name": chosen_row["location_name"],
                "timestamp": str(chosen_row.get("timestamp", datetime.datetime.now().isoformat()))[:19],
                "latitude": float(chosen_row["latitude"]),
                "longitude": float(chosen_row["longitude"]),
                "traffic_volume": int(chosen_row["traffic_volume"]),
                "average_speed": float(chosen_row["average_speed"]),
                "weather_condition": str(chosen_row["weather_condition"]),
                "visibility": float(chosen_row["visibility"]),
                "road_condition": str(chosen_row["road_condition"]),
                "historical_accidents": int(chosen_row["historical_accidents"])
            }
        else:
            raw_sensor = load_sensor_data()
            eval_sensor = {
                "location_name": "Active Edge Node (IoT Sensor Gateway #01)",
                "timestamp": raw_sensor.get("timestamp", "N/A")[:19],
                "latitude": float(raw_sensor["latitude"]),
                "longitude": float(raw_sensor["longitude"]),
                "traffic_volume": int(raw_sensor["traffic_volume"]),
                "average_speed": float(raw_sensor["average_speed"]),
                "weather_condition": str(raw_sensor["weather_condition"]),
                "visibility": float(raw_sensor["visibility"]),
                "road_condition": str(raw_sensor["road_condition"]),
                "historical_accidents": int(raw_sensor["historical_accidents"])
            }
            with col_m2:
                st.info(f"Connected to **Live Streaming Edge Node** at coordinates `({eval_sensor['latitude']}, {eval_sensor['longitude']})`.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Calculate model predictions for selected location
    current_risk, current_probs = predict_risk(eval_sensor, model, weather_map, road_map, return_proba=True)
    explanation_text = explain_risk(eval_sensor, current_risk)

    # Risk Score Calculation for Circular Gauge (0-100)
    risk_score_value = (
        current_probs["Low"] * 15.0
        + current_probs["Medium"] * 55.0
        + current_probs["High"] * 95.0
    )

    # 1. Pulsing Risk Banner
    if current_risk == "High":
        st.markdown(
            f'<div class="risk-banner-high">'
            f'<span>🚨 <b>CRITICAL ACCIDENT RISK DETECTED</b></span> &nbsp;|&nbsp; '
            f'<span>Probability: <b>{current_probs["High"] * 100:.1f}%</b></span> &nbsp;|&nbsp; '
            f'<span>Location: <b>{eval_sensor["location_name"]}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )
    elif current_risk == "Medium":
        st.markdown(
            f'<div class="risk-banner-medium">'
            f'<span>⚠️ <b>MODERATE ACCIDENT RISK HAZARD</b></span> &nbsp;|&nbsp; '
            f'<span>Probability: <b>{current_probs["Medium"] * 100:.1f}%</b></span> &nbsp;|&nbsp; '
            f'<span>Location: <b>{eval_sensor["location_name"]}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="risk-banner-low">'
            f'<span>✅ <b>NORMAL / SAFE TRAFFIC FLOW</b></span> &nbsp;|&nbsp; '
            f'<span>Probability: <b>{current_probs["Low"] * 100:.1f}%</b></span> &nbsp;|&nbsp; '
            f'<span>Location: <b>{eval_sensor["location_name"]}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # 2. Risk Meter Gauge + Probabilities + Explanation Card
    col_gauge, col_recs = st.columns([1, 2])

    with col_gauge:
        # Plotly Circular Speedometer Gauge Meter
        gauge_color = "#ef4444" if current_risk == "High" else "#eab308" if current_risk == "Medium" else "#10b981"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "/100", 'font': {'size': 28, 'family': 'Outfit', 'color': gauge_color}},
            title={'text': f"Risk Score Index<br><span style='font-size:0.9em;color:{gauge_color}'>{current_risk.upper()}</span>", 'font': {'size': 16, 'family': 'Outfit'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': gauge_color, 'thickness': 0.28},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [35, 70], 'color': "rgba(234, 179, 8, 0.25)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.8,
                    'value': risk_score_value
                }
            }
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_recs:
        st.subheader("💡 Intelligent Risk Analysis & Action Plan")
        st.info(explanation_text)

        # Calibrated Probability Horizontal Bar
        prob_df = pd.DataFrame({
            "Risk": ["Low", "Medium", "High"],
            "Probability": [current_probs["Low"], current_probs["Medium"], current_probs["High"]]
        })
        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Risk",
            orientation="h",
            color="Risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"},
            text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
            height=130
        )
        fig_prob.update_layout(
            margin=dict(l=0, r=0, t=5, b=5),
            xaxis=dict(range=[0, 1], showgrid=False, visible=False),
            yaxis=dict(title=""),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # 3. Four Modern Metric Stat Widgets
    st.subheader(f"📡 Real-Time Telemetry: {eval_sensor['location_name']}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-label">🚗 Traffic Volume</div>
            <div class="stat-value">{eval_sensor['traffic_volume']} <span style="font-size:0.9rem;font-weight:400;color:#94a3b8">veh/hr</span></div>
            <div class="stat-subtext">{'⚠️ High Congestion' if eval_sensor['traffic_volume'] > 800 else '✅ Normal Traffic Density'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-label">⚡ Velocity / Speed</div>
            <div class="stat-value">{eval_sensor['average_speed']} <span style="font-size:0.9rem;font-weight:400;color:#94a3b8">km/h</span></div>
            <div class="stat-subtext">{'⚠️ High Velocity Flow' if eval_sensor['average_speed'] > 55 else 'Slow / Bottlenecked'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-label">👁️ Sightline Visibility</div>
            <div class="stat-value">{eval_sensor['visibility']} <span style="font-size:0.9rem;font-weight:400;color:#94a3b8">km</span></div>
            <div class="stat-subtext">{'⚠️ Impaired Sightline' if eval_sensor['visibility'] < 3.0 else '✅ Clear Atmospheric View'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-widget">
            <div class="stat-label">⚠️ Black-Spot History</div>
            <div class="stat-value">{eval_sensor['historical_accidents']} <span style="font-size:0.9rem;font-weight:400;color:#94a3b8">crashes</span></div>
            <div class="stat-subtext">{'🚨 Critical Incident Zone' if eval_sensor['historical_accidents'] >= 10 else 'Low Historic Crashes'}</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Context & Location Mini-Map
    col_map_sub1, col_map_sub2 = st.columns([1, 1])
    with col_map_sub1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Weather Condition:** `{eval_sensor['weather_condition']}`")
        st.markdown(f"**Road Surface State:** `{eval_sensor['road_condition']}`")
        st.markdown(f"**Corridor Designation:** `{eval_sensor['location_name']}`")
        st.markdown(f"**Telemetry Heartbeat:** `{eval_sensor['timestamp']}`")
        st.markdown(f"**Coordinates:** `Lat: {eval_sensor['latitude']} | Lon: {eval_sensor['longitude']}`")
    with col_map_sub2:
        st.caption(f"📍 GPS Map Center: {eval_sensor['location_name']}")
        single_loc_df = pd.DataFrame([{
            "latitude": eval_sensor["latitude"],
            "longitude": eval_sensor["longitude"]
        }])
        st.map(single_loc_df, zoom=12)

    st.divider()

    # 5. Interactive "What-If" Scenario Simulator
    st.subheader("🎛️ Interactive 'What-If' Scenario Risk Simulator")
    st.markdown("Adjust parameters below to see how our AI model recalibrates risk dynamically in real-time:")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_traffic = st.slider("Traffic Volume (veh/hr)", 100, 1200, int(eval_sensor["traffic_volume"]), step=50)
        sim_speed = st.slider("Average Speed (km/h)", 15, 75, int(eval_sensor["average_speed"]), step=5)
    with sim_col2:
        sim_weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Fog"], index=["Clear", "Rain", "Fog"].index(eval_sensor.get("weather_condition", "Clear")))
        sim_visibility = st.slider("Atmospheric Visibility (km)", 1.0, 10.0, float(eval_sensor["visibility"]), step=0.5)
    with sim_col3:
        sim_road = st.selectbox("Road Surface Condition", ["Dry", "Wet", "Damaged"], index=["Dry", "Wet", "Damaged"].index(eval_sensor.get("road_condition", "Dry")))
        sim_accidents = st.slider("Historical Crashes", 0, 20, int(eval_sensor["historical_accidents"]), step=1)

    sim_payload = {
        "latitude": eval_sensor["latitude"],
        "longitude": eval_sensor["longitude"],
        "traffic_volume": sim_traffic,
        "average_speed": sim_speed,
        "weather_condition": sim_weather,
        "visibility": sim_visibility,
        "road_condition": sim_road,
        "historical_accidents": sim_accidents
    }

    sim_risk, sim_probs = predict_risk(sim_payload, model, weather_map, road_map, return_proba=True)
    sim_exp = explain_risk(sim_payload, sim_risk)

    badge_color = "#ef4444" if sim_risk == "High" else "#eab308" if sim_risk == "Medium" else "#10b981"
    st.markdown(
        f'<div style="border-left: 6px solid {badge_color}; background: rgba(30, 41, 59, 0.7); padding: 16px 20px; border-radius: 12px; margin-top: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">'
        f'<span style="font-size: 1.25rem; font-weight: 800; color: {badge_color}; font-family: Outfit;">SIMULATED OUTCOME: {sim_risk.upper()} ACCIDENT RISK</span>'
        f' &nbsp;•&nbsp; <span style="color:#94a3b8">(Low: <b>{sim_probs["Low"]*100:.1f}%</b>, Medium: <b>{sim_probs["Medium"]*100:.1f}%</b>, High: <b>{sim_probs["High"]*100:.1f}%</b>)</span><br>'
        f'<span style="font-size: 0.95rem; color: #cbd5e1; margin-top: 6px; display: inline-block;">{sim_exp}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ==========================================
# TAB 2: Geospatial Hotspots & Map Analysis
# ==========================================
with tab_map:
    st.subheader("🗺️ Delhi-NCR Accident Black-Spots & Corridor Radar")
    st.markdown("Delhi-NCR ke dangerous corridors aur blackspots ka clean, interactive spatial view.")

    # Top stats row
    tot_high = int((filtered_df["risk"] == "High").sum())
    tot_med = int((filtered_df["risk"] == "Medium").sum())
    tot_low = int((filtered_df["risk"] == "Low").sum())

    st_c1, st_c2, st_c3 = st.columns(3)
    st_c1.markdown(
        f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.85rem; color: #f87171; font-weight: 600;">🔴 HIGH RISK BLACKSPOTS</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #ef4444;">{tot_high} Corridors</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">High collision probability</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st_c2.markdown(
        f"""
        <div style="background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.4); border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.85rem; color: #facc15; font-weight: 600;">🟡 MEDIUM RISK WARNING</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #eab308;">{tot_med} Corridors</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Elevated traffic & weather alerts</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st_c3.markdown(
        f"""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 12px 16px; text-align: center;">
            <div style="font-size: 0.85rem; color: #34d399; font-weight: 600;">🟢 LOW RISK NORMAL FLOW</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #10b981;">{tot_low} Corridors</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Safe flow corridors</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Map Controls Toolbar
    map_ctrl1, map_ctrl2, map_ctrl3 = st.columns([5, 4, 3])

    with map_ctrl1:
        map_view_mode = st.radio(
            "🎨 Map View Style:",
            [
                "🔥 Smooth Risk Heatmap (Sabse Clean & Modern)",
                "🚨 High-Risk Blackspots Only (Clean Pins)",
                "📍 Micro-Dots Overview (All Corridors)"
            ],
            index=0
        )

    with map_ctrl2:
        risk_filter = st.multiselect(
            "Filter by Risk Category:",
            ["High", "Medium", "Low"],
            default=["High", "Medium"] if "Heatmap" not in map_view_mode else ["High", "Medium", "Low"]
        )

    # Prepare top high risk locations for spotlight selector
    top_high_places = filtered_df[filtered_df["risk"] == "High"].sort_values(
        by=["historical_accidents", "traffic_volume"],
        ascending=[False, False]
    )["location_name"].unique().tolist()

    with map_ctrl3:
        spotlight_loc = st.selectbox(
            "🎯 Zoom to Danger Spot:",
            ["-- All Delhi-NCR --"] + top_high_places[:20]
        )

    # Filter base dataset
    map_display_df = filtered_df.copy()
    if risk_filter:
        map_display_df = map_display_df[map_display_df["risk"].isin(risk_filter)]

    # If spotlight selected, filter or center
    focus_lat = 28.6139
    focus_lon = 77.2090
    focus_zoom = 10

    if spotlight_loc != "-- All Delhi-NCR --":
        spotlight_match = filtered_df[filtered_df["location_name"] == spotlight_loc]
        if not spotlight_match.empty:
            focus_lat = float(spotlight_match.iloc[0]["latitude"])
            focus_lon = float(spotlight_match.iloc[0]["longitude"])
            focus_zoom = 13

    # Add numeric weight for heatmap
    weight_mapping = {"High": 5, "Medium": 2, "Low": 1}
    map_display_df["heat_weight"] = map_display_df["risk"].map(weight_mapping) * (map_display_df["historical_accidents"] + 1)

    # RGB color coding: High=Red, Medium=Yellow, Low=Green
    rgb_map = {
        "High": [239, 68, 68, 220],
        "Medium": [234, 179, 8, 220],
        "Low": [16, 185, 129, 180]
    }
    map_display_df["color_rgb"] = map_display_df["risk"].map(rgb_map)

    # Two column layout: Left = Map, Right = Top Hazardous Corridors Leaderboard
    map_left_col, map_right_col = st.columns([7, 3])

    with map_left_col:
        deck_layers = []

        if "Smooth Risk Heatmap" in map_view_mode:
            # Heatmap layer: Smooth glowing density gradient without any messy dots
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=map_display_df,
                get_position=["longitude", "latitude"],
                get_weight="heat_weight",
                radius_pixels=45,
                intensity=1.8,
                threshold=0.08
            )
            # Subtle micro hover layer so user can still hover to see place names
            hover_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_display_df,
                get_position=["longitude", "latitude"],
                get_fill_color=[255, 255, 255, 40],
                get_radius=50,
                radius_min_pixels=2,
                radius_max_pixels=4,
                pickable=True
            )
            deck_layers = [heatmap_layer, hover_layer]

        elif "High-Risk Blackspots Only" in map_view_mode:
            # Show ONLY High Risk points with sleek, glowing rings
            danger_only_df = map_display_df[map_display_df["risk"] == "High"].copy()
            if danger_only_df.empty:
                danger_only_df = map_display_df.copy()

            blackspot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=danger_only_df,
                get_position=["longitude", "latitude"],
                get_fill_color=[239, 68, 68, 220],
                stroked=True,
                get_line_color=[255, 255, 255, 220],
                line_width_min_pixels=1.5,
                get_radius=100,
                radius_min_pixels=5,
                radius_max_pixels=9,
                pickable=True,
                auto_highlight=True
            )
            deck_layers = [blackspot_layer]

        else:
            # Micro-Dots: refined small dots, no giant overlapping blobs
            micro_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_display_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color_rgb",
                get_radius=60,
                radius_min_pixels=2.5,
                radius_max_pixels=5.5,
                pickable=True,
                auto_highlight=True
            )
            deck_layers = [micro_layer]

        # If a spotlight location was picked, add a prominent pulsating pin layer
        if spotlight_loc != "-- All Delhi-NCR --":
            spot_df = filtered_df[filtered_df["location_name"] == spotlight_loc].copy()
            if not spot_df.empty:
                spotlight_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=spot_df,
                    get_position=["longitude", "latitude"],
                    get_fill_color=[56, 189, 248, 255],
                    stroked=True,
                    get_line_color=[255, 255, 255, 255],
                    line_width_min_pixels=3,
                    get_radius=300,
                    radius_min_pixels=10,
                    radius_max_pixels=18,
                    pickable=True
                )
                deck_layers.append(spotlight_layer)

        view_state = pdk.ViewState(
            latitude=focus_lat,
            longitude=focus_lon,
            zoom=focus_zoom,
            pitch=15
        )

        deck = pdk.Deck(
            layers=deck_layers,
            initial_view_state=view_state,
            map_style="dark",
            tooltip={
                "html": """
                <div style='background-color:#0f172a; color:#ffffff; padding:12px 14px; border-radius:10px; border:1px solid #38bdf8; font-family: sans-serif; box-shadow: 0 10px 25px rgba(0,0,0,0.6); min-width: 220px;'>
                    <div style='font-weight:800; font-size:14px; color:#38bdf8; margin-bottom:4px;'>📍 {location_name}</div>
                    <div style='font-size:12px; margin-bottom:6px;'>🚨 <b>Risk Tier:</b> <span style='color:#f87171; font-weight:700;'>{risk}</span></div>
                    <div style='font-size:11px; color:#cbd5e1; line-height:1.6;'>
                        ⚠️ <b>Past Accidents:</b> {historical_accidents} crashes<br>
                        🚗 <b>Traffic Volume:</b> {traffic_volume} veh/hr<br>
                        ⚡ <b>Average Speed:</b> {average_speed} km/h<br>
                        🌧️ <b>Conditions:</b> {weather_condition} ({visibility} km) | {road_condition} Road
                    </div>
                </div>
                """,
                "style": {"color": "white"}
            }
        )
        st.pydeck_chart(deck, use_container_width=True)

    with map_right_col:
        st.markdown("#### 🚨 Sabse Jyada Khatra Yahan Hai:")
        st.caption("Top hazardous corridors sorted by collision count & congestion flow:")

        top_critical = filtered_df[filtered_df["risk"] == "High"].sort_values(
            by=["historical_accidents", "traffic_volume"],
            ascending=[False, False]
        ).head(6)

        for rank, (_, row) in enumerate(top_critical.iterrows(), 1):
            clean_name = row['location_name'].split('(')[0].strip()
            st.markdown(
                f"""
                <div style="background: rgba(185, 28, 28, 0.18); border-left: 4px solid #ef4444; padding: 10px 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(239, 68, 68, 0.25);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; color: #f87171; font-size: 0.88rem;">#{rank} 📍 {clean_name}</span>
                        <span style="background: #ef4444; color: #fff; font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">HIGH RISK</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #e2e8f0; margin-top: 4px;">
                        ⚠️ <b>{row['historical_accidents']}</b> Past Crashes &nbsp;|&nbsp; 🚗 <b>{row['traffic_volume']}</b> veh/hr
                    </div>
                    <div style="font-size: 0.74rem; color: #94a3b8; margin-top: 2px;">
                        ⚡ Speed: {row['average_speed']} km/h &nbsp;|&nbsp; 🌧️ {row['weather_condition']} ({row['road_condition']})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown("##### 📋 All Delhi-NCR Monitored Corridors (Data Table)")
    st.dataframe(
        map_display_df[[
            "location_name", "risk", "historical_accidents",
            "traffic_volume", "average_speed", "weather_condition",
            "road_condition", "visibility"
        ]].sort_values(by=["historical_accidents"], ascending=False),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 3: Network Analytics & Trends
# ==========================================
with tab_analytics:
    st.subheader("📊 Road Sensor Analytics & Pattern Intelligence")

    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        risk_dist = filtered_df["risk"].value_counts().reset_index()
        risk_dist.columns = ["Risk Level", "Count"]
        fig_risk = px.bar(
            risk_dist,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"},
            title="Risk Distribution Across Monitored Network",
            text="Count"
        )
        fig_risk.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with ch_col2:
        fig_scatter = px.scatter(
            filtered_df,
            x="average_speed",
            y="traffic_volume",
            color="risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"},
            size="historical_accidents",
            hover_name="location_name",
            title="Speed vs Traffic Volume Correlation",
            labels={"average_speed": "Average Speed (km/h)", "traffic_volume": "Traffic Volume (veh/hr)"}
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    ch_col3, ch_col4 = st.columns(2)

    with ch_col3:
        weather_risk = filtered_df.groupby(["weather_condition", "risk"]).size().reset_index(name="Count")
        fig_weather = px.bar(
            weather_risk,
            x="weather_condition",
            y="Count",
            color="risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"},
            barmode="group",
            title="Risk Breakdown by Weather Condition",
            labels={"weather_condition": "Weather"}
        )
        fig_weather.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    with ch_col4:
        road_risk = filtered_df.groupby(["road_condition", "risk"]).size().reset_index(name="Count")
        fig_road = px.bar(
            road_risk,
            x="road_condition",
            y="Count",
            color="risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#eab308", "High": "#ef4444"},
            barmode="group",
            title="Risk Breakdown by Road Surface State",
            labels={"road_condition": "Road Surface"}
        )
        fig_road.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_road, use_container_width=True)

    st.subheader("📑 Filtered Corridor Records Table")
    st.dataframe(filtered_df, use_container_width=True)


# ==========================================
# TAB 4: Real-World Crash Database (58K Records)
# ==========================================
with tab_crashdb:
    st.subheader("📁 Historical Crash Database (Tempe, Arizona - 58,963 Incidents)")
    st.markdown("Ground-truth empirical crash records benchmarked for road safety modeling.")

    df_crash = get_crash_report_dataset()

    if df_crash is None or len(df_crash) == 0:
        st.warning("Historical crash report dataset not loaded. Ensure `CrashDataReport.csv` exists in the Data folder.")
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_incidents = len(df_crash)
        total_injuries = int(df_crash["Totalinjuries"].sum()) if "Totalinjuries" in df_crash.columns else 0
        total_fatalities = int(df_crash["Totalfatalities"].sum()) if "Totalfatalities" in df_crash.columns else 0
        high_severity = (df_crash["accident_risk"] == "High").sum() if "accident_risk" in df_crash.columns else 0

        kpi1.metric("Total Crash Incidents", f"{total_incidents:,}")
        kpi2.metric("Total Injuries", f"{total_injuries:,}")
        kpi3.metric("Fatalities Recorded", f"{total_fatalities:,}")
        kpi4.metric("High-Severity Crashes", f"{high_severity:,}")

        st.divider()

        cd_col1, cd_col2 = st.columns(2)

        with cd_col1:
            if "Injuryseverity" in df_crash.columns:
                sev_counts = df_crash["Injuryseverity"].value_counts().head(8).reset_index()
                sev_counts.columns = ["Severity", "Count"]
                fig_sev = px.pie(
                    sev_counts,
                    names="Severity",
                    values="Count",
                    title="Accident Injury Severity Proportions",
                    hole=0.45
                )
                fig_sev.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0")
                )
                st.plotly_chart(fig_sev, use_container_width=True)

        with cd_col2:
            if "Collisionmanner" in df_crash.columns:
                coll_counts = df_crash["Collisionmanner"].value_counts().head(8).reset_index()
                coll_counts.columns = ["Collision Manner", "Count"]
                fig_coll = px.bar(
                    coll_counts,
                    x="Count",
                    y="Collision Manner",
                    orientation="h",
                    title="Top Collision Mechanisms",
                    color="Count",
                    color_continuous_scale="Reds"
                )
                fig_coll.update_layout(
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0")
                )
                st.plotly_chart(fig_coll, use_container_width=True)

        st.subheader("Recent Historical Incident Sample")
        sample_cols = [c for c in ["Year", "StreetName", "CrossStreet", "Injuryseverity", "Totalinjuries", "Totalfatalities", "accident_risk", "Weather", "SurfaceCondition"] if c in df_crash.columns]
        st.dataframe(df_crash[sample_cols].head(15), use_container_width=True)


# ==========================================
# TAB 5: Model Diagnostics & Explainability
# ==========================================
with tab_model:
    st.subheader("🧠 Machine Learning Model Architecture & Performance")

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("#### Model Specification")
        st.markdown("""
        - **Algorithm:** Extreme Gradient Boosting (XGBoost Classifier)
        - **Objective:** `multi:softprob` (Multiclass Calibrated Probabilities)
        - **Number of Classes:** 3 (Low, Medium, High)
        - **Test Set Accuracy:** **87.00%**
        - **Ensemble Estimators:** 100 Trees
        - **Tree Depth:** 4
        - **Learning Rate:** 0.10
        - **Evaluation Metric:** Multi-class Log Loss (`mlogloss`)
        """)

    with m_col2:
        st.markdown("#### Target Classification Schema")
        st.markdown("""
        - **Class 0 (Low):** Safe traffic density, clear visibility, benign weather.
        - **Class 1 (Medium):** Sub-optimal road grip, moderate speeds, emerging bottlenecks.
        - **Class 2 (High):** Critical congestion, poor sightlines, adverse precipitation, structural asphalt distress, hazardous black-spot history.
        """)

    st.divider()

    # Feature Importance Plot
    st.subheader("📈 Feature Importance Rankings (Gini / Gain)")
    importances = pkg.get("feature_importances", {})
    if importances:
        feat_df = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": [v * 100 for v in importances.values()]
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Purples",
            title="Relative Feature Predictive Power (%)",
            text=feat_df["Importance"].apply(lambda v: f"{v:.1f}%")
        )
        fig_imp.update_layout(
            margin=dict(l=0, r=0, t=30, b=10),
            coloraxis_showscale=False,
            xaxis=dict(title="Importance Weight (%)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# Footer
# ==========================================
st.divider()
st.caption(
    "Road Accident Risk Intelligence System | "
    "IoT Edge Simulation + XGBoost ML + FastAPI / Flask REST APIs + Streamlit Dashboard"
)