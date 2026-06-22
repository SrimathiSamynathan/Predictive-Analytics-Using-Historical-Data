import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1f6feb, #388bfd, #58a6ff);
    }
    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin: 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 20px 22px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #388bfd; }
    .kpi-label {
        font-size: 12px;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #8b949e;
    }
    .kpi-good { color: #3fb950; }
    .kpi-warn { color: #d29922; }
    .kpi-bad  { color: #f85149; }

    /* Section titles */
    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #e6edf3;
        margin: 8px 0 16px 0;
        padding-left: 10px;
        border-left: 3px solid #388bfd;
    }

    /* Info box */
    .info-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 3px solid #388bfd;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 13px;
        color: #8b949e;
        margin-bottom: 16px;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
    }

    /* Selectbox, slider */
    .stSelectbox > div > div,
    .stSlider > div {
        background-color: #21262d !important;
        border-color: #30363d !important;
        color: #e6edf3 !important;
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px;
    }

    footer { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Data Generation ─────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(category):
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", end="2025-12-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    if category == "Retail Sales":
        trend = 50000 + t * 800
        seasonal = 8000 * np.sin(2 * np.pi * t / 12) + 4000 * np.sin(4 * np.pi * t / 12)
        noise = np.random.normal(0, 3000, n)
    elif category == "Website Traffic":
        trend = 20000 + t * 300
        seasonal = 3000 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 1500, n)
    elif category == "Product Revenue":
        trend = 80000 + t * 1200
        seasonal = 15000 * np.sin(2 * np.pi * t / 12) + 5000 * np.cos(4 * np.pi * t / 12)
        noise = np.random.normal(0, 5000, n)
    else:  # Temperature
        trend = 20 + t * 0.02
        seasonal = 12 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 1.5, n)

    values = trend + seasonal + noise
    df = pd.DataFrame({"Date": dates, "Value": np.maximum(values, 0)})
    return df

@st.cache_data
def load_uploaded(file):
    df = pd.read_csv(file)
    return df


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data Source")
    data_source = st.radio("", ["Use Sample Dataset", "Upload CSV"], label_visibility="collapsed")

    if data_source == "Use Sample Dataset":
        category = st.selectbox("Dataset Category", [
            "Retail Sales", "Website Traffic", "Product Revenue", "Temperature"
        ])
        df = generate_dataset(category)
        value_col = "Value"
        date_col = "Date"
        unit = {"Retail Sales": "₹", "Website Traffic": "visits", "Product Revenue": "₹", "Temperature": "°C"}[category]
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = load_uploaded(uploaded)
            st.markdown("**Select Columns:**")
            date_col = st.selectbox("Date column", df.columns.tolist())
            value_col = st.selectbox("Value column", [c for c in df.columns if c != date_col])
            unit = ""
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[[date_col, value_col]].dropna().rename(columns={date_col: "Date", value_col: "Value"})
            date_col, value_col = "Date", "Value"
        else:
            st.info("Upload a CSV with a date column and a numeric value column.")
            st.stop()

    st.markdown("---")
    st.markdown("### Model Settings")
    model_type = st.selectbox("Model", ["Linear Regression", "Polynomial Regression (deg 2)", "Polynomial Regression (deg 3)"])
    forecast_months = st.slider("Forecast Months", 3, 24, 12)

    st.markdown("---")
    st.markdown("### Filter Historical Data")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#8b949e;text-align:center;'>Predictive Analytics Dashboard<br>Thiranex Data Science Internship</div>",
        unsafe_allow_html=True
    )


# ─── Filter & Prepare Data ────────────────────────────────────────────────────
if len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]

df = df.sort_values("Date").reset_index(drop=True)
df["t"] = np.arange(len(df))

X = df[["t"]].values
y = df["Value"].values

# ─── Train Model ──────────────────────────────────────────────────────────────
if model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "Polynomial Regression (deg 2)":
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
else:
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())

model.fit(X, y)
y_pred = model.predict(X)

# Metrics
mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_score(y, y_pred)

# Forecast
last_t = df["t"].max()
future_t = np.arange(last_t + 1, last_t + 1 + forecast_months).reshape(-1, 1)
future_dates = pd.date_range(
    start=df["Date"].max() + pd.DateOffset(months=1),
    periods=forecast_months, freq="MS"
)
future_pred = model.predict(future_t)
future_pred = np.maximum(future_pred, 0)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
    <div class="dashboard-title">Predictive Analytics Dashboard</div>
    <div class="dashboard-subtitle">Historical trend analysis and future forecasting using {model_type}</div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

r2_class = "kpi-good" if r2 >= 0.85 else ("kpi-warn" if r2 >= 0.6 else "kpi-bad")

with col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Data Points</div>
        <div class="kpi-value">{len(df)}</div>
        <div class="kpi-sub">historical records</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">R² Score</div>
        <div class="kpi-value {r2_class}">{r2:.3f}</div>
        <div class="kpi-sub">model accuracy</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">MAE</div>
        <div class="kpi-value">{unit}{mae:,.0f}</div>
        <div class="kpi-sub">mean absolute error</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Forecast Horizon</div>
        <div class="kpi-value">{forecast_months}</div>
        <div class="kpi-sub">months ahead</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Actual vs Predicted + Forecast</div>', unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Value"],
    name="Actual", mode="lines+markers",
    line=dict(color="#58a6ff", width=2),
    marker=dict(size=4, color="#58a6ff"),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"
))

fig.add_trace(go.Scatter(
    x=df["Date"], y=y_pred,
    name="Fitted Model", mode="lines",
    line=dict(color="#3fb950", width=2, dash="dot")
))

fig.add_trace(go.Scatter(
    x=future_dates, y=future_pred,
    name="Forecast", mode="lines+markers",
    line=dict(color="#d29922", width=2.5),
    marker=dict(size=5, color="#d29922", symbol="diamond"),
    fill="tozeroy", fillcolor="rgba(210,153,34,0.07)"
))

# Divider line
fig.add_vline(
    x=df["Date"].max(),
    line_dash="dash", line_color="#8b949e", line_width=1,
    annotation_text="Forecast Start", annotation_position="top right",
    annotation_font_color="#8b949e"
)

fig.update_layout(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#8b949e"),
    legend=dict(
        bgcolor="#161b22", bordercolor="#21262d",
        font=dict(color="#e6edf3"), orientation="h",
        yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", title="Date"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", title=f"Value ({unit})"),
    hovermode="x unified",
    height=420,
    margin=dict(t=40, b=40, l=60, r=20)
)

st.plotly_chart(fig, use_container_width=True)

# ─── Residuals Chart ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-title">Residuals (Actual - Predicted)</div>', unsafe_allow_html=True)
    residuals = y - y_pred
    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(
        x=df["Date"], y=residuals,
        marker_color=["#f85149" if r < 0 else "#3fb950" for r in residuals],
        name="Residual"
    ))
    fig_res.add_hline(y=0, line_color="#8b949e", line_width=1)
    fig_res.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font=dict(family="Inter", color="#8b949e"),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        height=280, margin=dict(t=20, b=40, l=60, r=20),
        showlegend=False
    )
    st.plotly_chart(fig_res, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title">Forecast Values</div>', unsafe_allow_html=True)
    forecast_df = pd.DataFrame({
        "Month": future_dates.strftime("%b %Y"),
        "Predicted Value": [f"{unit}{v:,.0f}" for v in future_pred]
    })
    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True,
        height=280
    )

# ─── Model Summary ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Performance Summary</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("R² Score", f"{r2:.4f}", help="Closer to 1.0 = better fit")
with m2:
    st.metric("MAE", f"{unit}{mae:,.1f}", help="Mean Absolute Error")
with m3:
    st.metric("RMSE", f"{unit}{rmse:,.1f}", help="Root Mean Squared Error")
with m4:
    growth = ((future_pred[-1] - y[-1]) / y[-1]) * 100
    st.metric("Projected Growth", f"{growth:+.1f}%", help=f"Over next {forecast_months} months")

# ─── Download ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

full_export = pd.DataFrame({
    "Date": list(df["Date"].dt.strftime("%Y-%m-%d")) + list(future_dates.strftime("%Y-%m-%d")),
    "Actual": list(df["Value"].round(2)) + [None] * forecast_months,
    "Fitted": list(y_pred.round(2)) + [None] * forecast_months,
    "Forecast": [None] * len(df) + list(future_pred.round(2)),
    "Type": ["Historical"] * len(df) + ["Forecast"] * forecast_months
})

st.download_button(
    label="Download Full Predictions CSV",
    data=full_export.to_csv(index=False),
    file_name="predictive_analytics_results.csv",
    mime="text/csv"
)   