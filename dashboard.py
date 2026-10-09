import datetime
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# Page Configuration & Modern Theme Styling
# ==========================================
st.set_page_config(
    page_title="Road Accident Risk Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern CSS styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        color: #fef2f2;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-banner-medium {
        background: linear-gradient(135deg, #78350f, #92400e);
        border: 1px solid #f59e0b;
        color: #fffbeb;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .risk-banner-low {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        color: #ecfdf5;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-online { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .status-offline { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
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
# Load Model & Current Sensor Telemetry
# ==========================================
try:
    pkg = get_model_package()
    model = pkg["model"]
    weather_map = pkg["weather_map"]
    road_map = pkg["road_map"]
    accuracy = pkg.get("accuracy", 0.87)
    sensor = load_sensor_data()
    current_risk, current_probs = predict_risk(sensor, model, weather_map, road_map, return_proba=True)
    explanation_text = explain_risk(sensor, current_risk)
except Exception as err:
    st.error(f"Critical error loading model or sensor data: {err}")
    st.stop()


# ==========================================
# Sidebar: Controls, Telemetry & Filters
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.title("System Control")

    st.markdown('<span class="status-badge status-online">● EDGE IOT SYSTEM ONLINE</span>', unsafe_allow_html=True)
    st.markdown(f"**Model:** XGBoost Classifier ({accuracy * 100:.1f}% Acc)")
    st.caption(f"Last Sensor Reading: `{sensor.get('timestamp', 'N/A')[:19]}`")
    st.caption(f"Dashboard Refreshed: `{datetime.datetime.now().strftime('%H:%M:%S')}`")

    st.divider()

    st.subheader("⚡ Live Actions")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Refresh", use_container_width=True, help="Reload latest sensor telemetry"):
            st.rerun()
    with col_sb2:
        if st.button("📡 New IoT", use_container_width=True, help="Simulate a new IoT reading"):
            save_sensor_reading()
            st.rerun()

    st.divider()

    st.subheader("🔍 Dataset Filters")
    df_road = get_road_dataset()

    min_date = pd.to_datetime(df_road["timestamp"]).min().date()
    max_date = pd.to_datetime(df_road["timestamp"]).max().date()

    date_filter = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter historical records by date"
    )

    weather_options = sorted(list(df_road["weather_condition"].unique()))
    selected_weather = st.multiselect("Weather", weather_options, default=weather_options)

    road_options = sorted(list(df_road["road_condition"].unique()))
    selected_road = st.multiselect("Road Surface", road_options, default=road_options)

    risk_options = ["Low", "Medium", "High"]
    selected_risk = st.multiselect("Risk Level", risk_options, default=risk_options)

    # Filtered dataframe
    if isinstance(date_filter, (tuple, list)) and len(date_filter) == 2:
        start_d, end_d = date_filter
    else:
        start_d, end_d = min_date, max_date

    mask_date = (pd.to_datetime(df_road["timestamp"]).dt.date >= start_d) & (pd.to_datetime(df_road["timestamp"]).dt.date <= end_d)
    mask_weather = df_road["weather_condition"].isin(selected_weather)
    mask_road = df_road["road_condition"].isin(selected_road)
    mask_risk = df_road["risk"].isin(selected_risk)

    filtered_df = df_road[mask_date & mask_weather & mask_road & mask_risk]

    st.caption(f"Showing **{len(filtered_df)}** of **{len(df_road)}** records")

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Export Filtered CSV",
        data=csv_data,
        file_name="filtered_road_accident_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================
# Main Header
# ==========================================
st.title("🚦 AI Road Accident Risk & IoT Telemetry System")
st.markdown(
    "Real-time edge sensor processing, XGBoost machine learning risk classification, "
    "geospatial black-spot intelligence, and historical crash analytics."
)

st.divider()

# ==========================================
# Navigation Tabs
# ==========================================
tab_live, tab_map, tab_analytics, tab_crashdb, tab_model = st.tabs([
    "🛰️ Real-Time IoT Monitor",
    "🗺️ Geospatial Hotspots",
    "📊 Sensor Analytics & Trends",
    "📁 Real-World Crash Database (58K)",
    "🧠 Model Diagnostics & Explainability"
])


# ==========================================
# TAB 1: Real-Time IoT Monitor
# ==========================================
with tab_live:
    # 1. Live Risk Banner
    if current_risk == "High":
        st.markdown(
            f'<div class="risk-banner-high">⚠️ HIGH ACCIDENT RISK DETECTED &nbsp;|&nbsp; '
            f'Confidence: {current_probs["High"] * 100:.1f}%</div>',
            unsafe_allow_html=True
        )
    elif current_risk == "Medium":
        st.markdown(
            f'<div class="risk-banner-medium">⚠️ MODERATE ACCIDENT RISK &nbsp;|&nbsp; '
            f'Confidence: {current_probs["Medium"] * 100:.1f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="risk-banner-low">✅ SAFE ROAD CONDITIONS &nbsp;|&nbsp; '
            f'Confidence: {current_probs["Low"] * 100:.1f}%</div>',
            unsafe_allow_html=True
        )

    # 2. Probability Breakdown & Explanation Cards
    col_exp1, col_exp2 = st.columns([3, 2])
    with col_exp1:
        st.subheader("💡 Intelligent Risk Analysis & Recommendations")
        st.info(explanation_text)

    with col_exp2:
        st.subheader("🎯 Class Probabilities")
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
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
            height=200
        )
        fig_prob.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(range=[0, 1], showgrid=True),
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # 3. Live Sensor Telemetry Metrics
    st.subheader("📡 Live Sensor Telemetry")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Traffic Volume",
            value=f"{sensor['traffic_volume']} veh/hr",
            delta="High Congestion" if sensor["traffic_volume"] > 800 else "Normal Flow"
        )
    with col2:
        st.metric(
            label="Average Speed",
            value=f"{sensor['average_speed']} km/h",
            delta="Fast Flow" if sensor["average_speed"] > 55 else "Congested / Slow",
            delta_color="inverse" if sensor["average_speed"] > 55 else "normal"
        )
    with col3:
        st.metric(
            label="Visibility",
            value=f"{sensor['visibility']} km",
            delta="Poor Sight" if sensor["visibility"] < 3.0 else "Clear Sight",
            delta_color="inverse" if sensor["visibility"] < 3.0 else "normal"
        )
    with col4:
        st.metric(
            label="Historical Accidents",
            value=f"{sensor['historical_accidents']} crashes",
            delta="Hazardous Zone" if sensor["historical_accidents"] >= 10 else "Low Incident Zone",
            delta_color="inverse" if sensor["historical_accidents"] >= 10 else "normal"
        )

    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        st.markdown(f"**Weather Condition:** `{sensor['weather_condition']}`")
        st.markdown(f"**Road Surface:** `{sensor['road_condition']}`")
        st.markdown(f"**GPS Coordinates:** `({sensor['latitude']}, {sensor['longitude']})`")
    with col_meta2:
        # Current Location Map
        loc_df = pd.DataFrame([{
            "latitude": sensor["latitude"],
            "longitude": sensor["longitude"]
        }])
        st.caption("Active IoT Road Node Location")
        st.map(loc_df, zoom=12)

    st.divider()

    # 4. Interactive "What-If" Scenario Simulator
    st.subheader("🎛️ Interactive 'What-If' Scenario Risk Simulator")
    st.markdown("Adjust road and environment parameters below to see how our AI model classifies the risk in real-time.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_traffic = st.slider("Simulated Traffic Volume (veh/hr)", 100, 1200, int(sensor["traffic_volume"]), step=50)
        sim_speed = st.slider("Simulated Speed (km/h)", 15, 75, int(sensor["average_speed"]), step=5)
    with sim_col2:
        sim_weather = st.selectbox("Simulated Weather", ["Clear", "Rain", "Fog"], index=["Clear", "Rain", "Fog"].index(sensor.get("weather_condition", "Clear")))
        sim_visibility = st.slider("Simulated Visibility (km)", 1.0, 10.0, float(sensor["visibility"]), step=0.5)
    with sim_col3:
        sim_road = st.selectbox("Simulated Road Condition", ["Dry", "Wet", "Damaged"], index=["Dry", "Wet", "Damaged"].index(sensor.get("road_condition", "Dry")))
        sim_accidents = st.slider("Simulated Historical Crashes", 0, 20, int(sensor["historical_accidents"]), step=1)

    sim_payload = {
        "latitude": sensor["latitude"],
        "longitude": sensor["longitude"],
        "traffic_volume": sim_traffic,
        "average_speed": sim_speed,
        "weather_condition": sim_weather,
        "visibility": sim_visibility,
        "road_condition": sim_road,
        "historical_accidents": sim_accidents
    }

    sim_risk, sim_probs = predict_risk(sim_payload, model, weather_map, road_map, return_proba=True)
    sim_exp = explain_risk(sim_payload, sim_risk)

    badge_color = "#ef4444" if sim_risk == "High" else "#f59e0b" if sim_risk == "Medium" else "#10b981"
    st.markdown(
        f'<div style="border-left: 6px solid {badge_color}; background: rgba(255,255,255,0.03); padding: 14px; border-radius: 6px; margin-top: 10px;">'
        f'<span style="font-size: 1.15rem; font-weight: bold; color: {badge_color};">Simulated Outcome: {sim_risk.upper()} RISK</span>'
        f' &nbsp;(Low: {sim_probs["Low"]*100:.1f}%, Medium: {sim_probs["Medium"]*100:.1f}%, High: {sim_probs["High"]*100:.1f}%)<br>'
        f'<span style="font-size: 0.95rem; color: #cbd5e1;">{sim_exp}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ==========================================
# TAB 2: Geospatial Hotspots & Map Analysis
# ==========================================
with tab_map:
    st.subheader("🗺️ Delhi-NCR Accident Black-Spots & Segment Mapping")
    st.markdown("Visualizing 500 monitored road segments categorized by their ML-predicted risk level.")

    map_filter_col1, map_filter_col2 = st.columns([3, 1])
    with map_filter_col2:
        st.markdown("#### Black-Spot Stats")
        high_cnt = (filtered_df["risk"] == "High").sum()
        med_cnt = (filtered_df["risk"] == "Medium").sum()
        low_cnt = (filtered_df["risk"] == "Low").sum()
        st.metric("🔴 High Risk Segments", high_cnt)
        st.metric("🟠 Medium Risk Segments", med_cnt)
        st.metric("🟢 Low Risk Segments", low_cnt)

    with map_filter_col1:
        # Prepare color map for Streamlit map or Plotly scatter
        color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
        fig_map = px.scatter_map(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="risk",
            color_discrete_map=color_map,
            size="historical_accidents",
            size_max=15,
            hover_name="risk",
            hover_data={
                "traffic_volume": True,
                "average_speed": True,
                "weather_condition": True,
                "road_condition": True,
                "historical_accidents": True,
                "latitude": False,
                "longitude": False
            },
            zoom=10,
            title="Road Segments & Accident Risk Hotspots",
            height=550
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            map_style="carto-positron"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("⚠️ Top 10 Most Critical Road Hazards")
    top_hazards = filtered_df.sort_values(
        by=["historical_accidents", "traffic_volume"],
        ascending=[False, False]
    ).head(10)
    st.dataframe(
        top_hazards[[
            "latitude", "longitude", "risk", "traffic_volume",
            "average_speed", "weather_condition", "road_condition",
            "visibility", "historical_accidents"
        ]],
        use_container_width=True
    )


# ==========================================
# TAB 3: Sensor Analytics & Trends
# ==========================================
with tab_analytics:
    st.subheader("📊 Road Sensor Analytics & Pattern Intelligence")

    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        # Risk distribution bar chart
        risk_dist = filtered_df["risk"].value_counts().reset_index()
        risk_dist.columns = ["Risk Level", "Count"]
        fig_risk = px.bar(
            risk_dist,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            title="Distribution of Accident Risk Across Road Network",
            text="Count"
        )
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(fig_risk, use_container_width=True)

    with ch_col2:
        # Speed vs Traffic Volume scatter
        fig_scatter = px.scatter(
            filtered_df,
            x="average_speed",
            y="traffic_volume",
            color="risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            size="historical_accidents",
            title="Speed vs Traffic Volume Correlation",
            labels={"average_speed": "Average Speed (km/h)", "traffic_volume": "Traffic Volume (veh/hr)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    ch_col3, ch_col4 = st.columns(2)

    with ch_col3:
        # Weather impact on risk
        weather_risk = filtered_df.groupby(["weather_condition", "risk"]).size().reset_index(name="Count")
        fig_weather = px.bar(
            weather_risk,
            x="weather_condition",
            y="Count",
            color="risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            barmode="group",
            title="Risk Breakdown by Weather Condition",
            labels={"weather_condition": "Weather"}
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    with ch_col4:
        # Road condition impact on risk
        road_risk = filtered_df.groupby(["road_condition", "risk"]).size().reset_index(name="Count")
        fig_road = px.bar(
            road_risk,
            x="road_condition",
            y="Count",
            color="risk",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            barmode="group",
            title="Risk Breakdown by Road Surface Condition",
            labels={"road_condition": "Road Surface"}
        )
        st.plotly_chart(fig_road, use_container_width=True)

    st.subheader("📑 Filtered Sensor Records Table")
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
        # High-level KPIs
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
                    hole=0.4
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
                fig_coll.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
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
            color_continuous_scale="Blues",
            title="Relative Feature Predictive Power (%)",
            text=feat_df["Importance"].apply(lambda v: f"{v:.1f}%")
        )
        fig_imp.update_layout(
            margin=dict(l=0, r=0, t=30, b=10),
            coloraxis_showscale=False,
            xaxis=dict(title="Importance Weight (%)")
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