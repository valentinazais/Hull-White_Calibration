"""
Hull-White One-Factor Model Calibration — Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from hull_white import build_zero_curve, calibrate, theta, hw_swaption_vol_analytic

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Hull-White Calibration", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; margin-top: 1rem !important; }
    .stMetric { background: #0e1117; border: 1px solid #333; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("Hull-White One-Factor Model Calibration")
st.markdown("$dr(t) = [\\theta(t) - a\\, r(t)]\\, dt + \\sigma\\, dW(t)$")

# ─────────────────────────────────────────────
# Market Data
# ─────────────────────────────────────────────

SWAPTION_VOL_DATA = {
    '1Yr': [117.7,116.85,118.99,119.39,120.03,110.13,104.04,99.59,97.17,95.37,93.58,92.51,91.62,90.72,89.47,87.7,84.93,81.95,78.44],
    '2Yr': [128.38,126.07,122.72,118.66,116.46,107.38,101.75,97.86,95.65,93.82,92.63,91.72,91.27,90.44,89.02,86.98,84.78,82.28,80.59],
    '3Yr': [121.84,118.02,114.82,112.09,110.61,103.36,98.92,96.22,94.22,92.65,91.74,90.93,90.32,89.76,88.25,86.04,83.19,80.83,78.63],
    '4Yr': [113.12,113.21,111.43,108.52,106.64,101.36,97.62,94.96,93.26,91.99,91.09,90.38,89.76,89.09,87.5,85.17,82.16,79.66,77.27],
    '5Yr': [107.64,108.89,107.45,105.0,103.59,99.03,95.82,93.67,92.32,91.51,90.52,89.85,89.21,88.5,86.83,84.39,81.12,78.49,75.91],
    '7Yr': [97.11,99.57,98.99,97.9,97.27,94.61,92.66,91.39,90.49,89.79,89.03,88.38,87.73,87.02,85.31,82.78,79.73,77.21,74.01],
    '10Yr': [89.11,90.38,91.13,91.35,91.27,90.64,89.76,89.16,88.61,87.97,87.28,86.69,86.03,85.33,83.58,81.01,78.05,75.29,71.39],
    '12Yr': [85.86,88.58,89.72,89.97,90.04,89.45,88.59,87.95,87.35,86.63,85.89,85.27,84.43,83.8,82.07,79.5,76.64,74.08,70.32],
    '15Yr': [80.98,85.98,87.63,87.99,88.25,87.73,86.89,86.19,85.53,84.69,83.88,83.21,82.15,81.6,79.89,77.32,74.6,72.34,68.78],
    '20Yr': [77.62,82.19,84.97,85.71,85.93,85.91,85.29,84.8,84.22,83.38,82.54,81.84,80.59,80.1,78.39,75.79,73.4,71.2,68.07],
    '25Yr': [76.79,80.81,84.02,84.73,84.93,84.98,84.39,83.95,83.36,82.5,81.72,80.89,79.89,79.3,77.51,75.07,72.85,70.72,67.58],
    '30Yr': [75.86,79.77,83.34,84.29,84.45,84.6,84.04,83.69,83.05,82.24,81.5,80.64,79.88,78.97,77.26,74.82,72.65,70.76,68.13],
}

EXPIRY_LABELS = ['1Mo','3Mo','6Mo','9Mo','1Yr','2Yr','3Yr','4Yr','5Yr',
                 '6Yr','7Yr','8Yr','9Yr','10Yr','12Yr','15Yr','20Yr','25Yr','30Yr']

TENOR_LABELS = ['1Yr','2Yr','3Yr','4Yr','5Yr','7Yr','10Yr','12Yr','15Yr','20Yr','25Yr','30Yr']

EXPIRY_MAP = {
    '1Mo':1/12,'3Mo':3/12,'6Mo':6/12,'9Mo':9/12,
    '1Yr':1,'2Yr':2,'3Yr':3,'4Yr':4,'5Yr':5,'6Yr':6,'7Yr':7,
    '8Yr':8,'9Yr':9,'10Yr':10,'12Yr':12,'15Yr':15,'20Yr':20,'25Yr':25,'30Yr':30
}

TENOR_MAP = {
    '1Yr':1,'2Yr':2,'3Yr':3,'4Yr':4,'5Yr':5,'7Yr':7,'10Yr':10,
    '12Yr':12,'15Yr':15,'20Yr':20,'25Yr':25,'30Yr':30
}

YIELD_CURVE = {
    '1M': 3.677, '6W': 3.672, '2M': 3.709, '3M': 3.715,
    '4M': 3.733, '6M': 3.757, '1Y': 3.776, '2Y': 3.859,
    '3Y': 3.877, '5Y': 3.974, '7Y': 4.166, '10Y': 4.351,
    '20Y': 4.946, '30Y': 4.911
}

YIELD_TENOR_MAP = {
    '1M':1/12,'6W':6/52,'2M':2/12,'3M':3/12,'4M':4/12,
    '6M':6/12,'1Y':1,'2Y':2,'3Y':3,'5Y':5,'7Y':7,
    '10Y':10,'20Y':20,'30Y':30
}

def get_market_data():
    vol_df = pd.DataFrame(SWAPTION_VOL_DATA, index=EXPIRY_LABELS)
    vol_df.columns = TENOR_LABELS
    vol_df.index.name = 'Expiry'

    yc_tenors = np.array([YIELD_TENOR_MAP[k] for k in YIELD_CURVE])
    yc_yields = np.array([v / 100.0 for v in YIELD_CURVE.values()])
    sort_idx = np.argsort(yc_tenors)

    yc_tenors = yc_tenors[sort_idx]
    yc_yields = yc_yields[sort_idx]

    return vol_df, yc_tenors, yc_yields

vol_df, yc_tenors, yc_yields = get_market_data()

# ─────────────────────────────────────────────
# Calibration Controls
# ─────────────────────────────────────────────
st.header("Calibration")

with st.sidebar:
    st.subheader("Initial Parameters")
    a_init = st.slider("Mean-reversion speed (a₀)", 0.001, 0.50, 0.05, 0.001, format="%.3f")
    sigma_init = st.slider("Volatility (σ₀)", 0.001, 0.05, 0.01, 0.001, format="%.4f")

run_btn = st.button("Run Calibration", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Market Data Display
# ─────────────────────────────────────────────
st.header("Market Data")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Swaption Normal Vol Surface (bps)")
    fig_vol = go.Figure(data=go.Heatmap(
        z=vol_df.values,
        x=TENOR_LABELS,
        y=EXPIRY_LABELS,
        colorscale='RdYlBu_r'
    ))
    fig_vol.update_layout(
        xaxis_title='Swap Tenor',
        yaxis_title='Option Expiry',
        height=550,
        margin=dict(l=60, r=20, t=30, b=50),
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col2:
    st.subheader("Yield Curve")
    yc_labels = list(YIELD_CURVE.keys())
    yc_labels_sorted = [yc_labels[i] for i in np.argsort([YIELD_TENOR_MAP[k] for k in yc_labels])]
    yc_vals_sorted = [YIELD_CURVE[k] for k in yc_labels_sorted]

    fig_yc = go.Figure()
    fig_yc.add_trace(go.Scatter(
        x=yc_labels_sorted,
        y=yc_vals_sorted,
        mode='lines+markers'
    ))
    fig_yc.update_layout(
        yaxis_title='Yield (%)',
        height=550
    )
    st.plotly_chart(fig_yc, use_container_width=True)

# ─────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────
if run_btn:
    zero_spline = build_zero_curve(yc_tenors, yc_yields)

    result = calibrate(
        vol_df, zero_spline, EXPIRY_MAP, TENOR_MAP,
        a0=a_init, sigma0=sigma_init
    )

    st.header("Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean-Reversion (a*)", f"{result['a']:.6f}")
    c2.metric("Volatility (σ*)", f"{result['sigma']:.6f}")
    c3.metric("Final RMSE", f"{result['history'][-1]['rmse']:.2f} bps")

    hist_df = pd.DataFrame(result['history'])

    col_a, col_b = st.columns(2)

    with col_a:
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=hist_df['iteration'],
            y=hist_df['rmse'],
            mode='lines',
            name='RMSE'
        ))
        fig_conv.update_layout(
            title='RMSE vs Iteration',
            xaxis_title='Iteration',
            yaxis_title='RMSE (bps)',
            height=350
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    with col_b:
        fig_params = go.Figure()
        fig_params.add_trace(go.Scatter(
            x=hist_df['iteration'],
            y=hist_df['a'],
            mode='lines',
            name='a'
        ))
        fig_params.add_trace(go.Scatter(
            x=hist_df['iteration'],
            y=hist_df['sigma'],
            mode='lines',
            name='σ',
            yaxis='y2'
        ))

        # FIX PLOTLY
        fig_params.update_layout(
            title='Parameter Trajectory',
            xaxis_title='Iteration',
            yaxis=dict(
                title=dict(text='a')
            ),
            yaxis2=dict(
                title=dict(text='σ'),
                overlaying='y',
                side='right'
            ),
            height=350,
            margin=dict(l=60, r=60, t=40, b=40)
        )

        st.plotly_chart(fig_params, use_container_width=True)
