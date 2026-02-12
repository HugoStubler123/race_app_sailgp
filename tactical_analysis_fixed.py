"""
Tactical Analysis Module for Sailing Race App - FIXED VERSION
Contains functions for analyzing upwind/downwind leg performance
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
import streamlit as st
from io import BytesIO

COLOR_MAPPING = {
    "AUS": "#009A00",
    "CAN": "#F86767",
    "DEN": "#c50a07",
    "ESP": "#F58700",
    "FRA": "#0E8DFB",
    "GBR": "#bd9c67",
    "GER": "#5F8593",
    "NZL": "#030014",
    "SUI": "#b33b82",
    "USA": "#001FAB",
    "BRA": "#23B5D3",
    "ITA": "mediumseagreen"
}

def create_rotated_legs_plot(legs_df, leg_summary_df, leg_type='uw'):
    """
    Create rotated legs plot EXACTLY as in notebook:
    - Rotate tracks by mean TWD
    - Color by instantaneous TWD deviation from mean
    - Line width per leg scaled by inverse time (faster legs = thicker)
    - Uses simple lat/lon offset (no meter conversion)
    """
    summary = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    legs = legs_df[legs_df["leg_id"].isin(summary["leg_id"])].copy()

    if summary.empty or legs.empty:
        return None

    # --- rotation angle (deg -> rad)
    rotation_angle = np.deg2rad(summary["avg_TWD"].mean())
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)

    # --- line width scaling (inverse of time)
    time_values = summary.set_index("leg_id")["time_seconds"].astype(float)
    q75 = time_values.quantile(0.75)
    q25 = time_values.quantile(0.25)
    den = (q75 - q25) if (q75 - q25) != 0 else 1.0  # avoid divide-by-zero

    # --- global color normalization across ALL points
    twd_mean = summary["avg_TWD"].mean()
    twd_dev_all = (legs["TWD_MHU_SGP_deg"].astype(float) - twd_mean).to_numpy()
    
    # If all deviations are same value, TwoSlopeNorm breaks -> fall back to symmetric range
    if np.nanmax(twd_dev_all) == np.nanmin(twd_dev_all):
        vmax = max(1.0, abs(np.nanmax(twd_dev_all)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(
            vmin=np.nanmin(twd_dev_all),
            vcenter=0.0,
            vmax=np.nanmax(twd_dev_all)
        )

    cmap = cm.RdYlGn

    fig, ax = plt.subplots(figsize=(10, 8))

    for legID in summary.sort_values("time_seconds")["leg_id"].unique():
        leg_data = legs[legs["leg_id"] == legID].copy()
        if len(leg_data) < 2:
            continue

        lon = leg_data["LONGITUDE_GPS_unk"].to_numpy(dtype=float)
        lat = leg_data["LATITUDE_GPS_unk"].to_numpy(dtype=float)

        # rotate around the leg centroid for nicer visuals
        lon0, lat0 = np.nanmean(lon), np.nanmean(lat)
        x = lon - lon0
        y = lat - lat0

        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle

        leg_time = float(time_values.loc[legID])
        linewidth = 1 + 9 * (q75 - leg_time) / den
        linewidth = float(np.clip(linewidth, 1, 10))

        twd_dev = (leg_data["TWD_MHU_SGP_deg"].to_numpy(dtype=float) - twd_mean)
        seg_colors = cmap(norm(twd_dev))  # Nx4 RGBA array

        for i in range(len(x_rot) - 1):
            ax.plot(
                x_rot[i:i+2], y_rot[i:i+2],
                color=tuple(seg_colors[i]),   # ensure RGBA tuple
                linewidth=linewidth,
                alpha=0.8
            )

    ax.set_xlabel("Rotated X (lon)")
    ax.set_ylabel("Rotated Y (lat)")
    ax.set_title(f"Detected {'UW' if leg_type == 'uw' else 'DW'} Legs (Rotated by {np.rad2deg(rotation_angle):.1f}°)")
    ax.set_aspect("equal", adjustable="box")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=f"TWD deviation from {'UW' if leg_type == 'uw' else 'DW'} mean (deg)")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def get_gradient_color_time(value, min_val, max_val):
    """
    Generate RGB color for time gradient: GREEN (fast/low) to RED (slow/high)
    """
    if max_val == min_val:
        return "background-color: #ffffff"
    
    # Normalize: 0 = min (fast), 1 = max (slow)
    normalized = (value - min_val) / (max_val - min_val)
    
    # Invert so low time = green (0), high time = red (1)
    # Green to Yellow to Red
    if normalized < 0.5:
        r = int(normalized * 2 * 255)
        g = 255
        b = 0
    else:
        r = 255
        g = int((1 - (normalized - 0.5) * 2) * 255)
        b = 0
    
    return f"background-color: rgb({r}, {g}, {b})"


def create_gate_summary_table(leg_summary_df, leg_type='uw'):
    """
    Create summary table of average times by gate with COLOR GRADIENT
    GREEN = fastest (lowest time), RED = slowest (highest time)
    
    Args:
        leg_summary_df: DataFrame with leg summaries
        leg_type: 'uw' for upwind or 'dw' for downwind
    
    Returns:
        Styled DataFrame with color gradients
    """
    filtered = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    
    if len(filtered) == 0:
        return None
    
    # Group by gate and calculate mean time
    summary = filtered.groupby("gate")["time_seconds"].agg(['mean', 'count']).round(1)
    summary.columns = ['Avg Time (s)', 'Count']
    
    # Apply color gradient to "Avg Time (s)" column
    min_time = summary['Avg Time (s)'].min()
    max_time = summary['Avg Time (s)'].max()
    
    def apply_time_gradient(row):
        styles = []
        # Color the Avg Time column
        styles.append(get_gradient_color_time(row['Avg Time (s)'], min_time, max_time))
        # No color for Count column
        styles.append("")
        return styles
    
    styled_df = summary.style.apply(apply_time_gradient, axis=1)
    
    return styled_df


def render_tactical_tab(legs_df, leg_summary_df):
    """
    Render the Tactical Analysis tab in Streamlit
    
    Args:
        legs_df: DataFrame with all leg data points
        leg_summary_df: DataFrame with leg summary statistics
    """
    st.markdown("## ⚓ Tactical Analysis")
    st.markdown("Analyze upwind and downwind leg performance with detailed gate crossing statistics.")
    
    # Create two tabs for upwind and downwind
    uw_tab, dw_tab = st.tabs(["📈 Upwind Legs", "📉 Downwind Legs"])
    
    with uw_tab:
        st.markdown("### Upwind Legs Analysis")
        
        # Check if we have upwind data
        uw_summary = leg_summary_df[leg_summary_df["leg_type"] == "uw"]
        if len(uw_summary) == 0:
            st.warning("No upwind legs detected in this race.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Rotated Leg Tracks**")
                st.caption("Line width = speed (thicker = faster), Color = TWD deviation from mean")
                
                # Create plot
                uw_plot = create_rotated_legs_plot(legs_df, leg_summary_df, leg_type='uw')
                if uw_plot:
                    st.image(uw_plot, use_container_width=True)
                else:
                    st.error("Could not generate upwind plot")
            
            with col2:
                st.markdown("**Gate Performance**")
                st.caption("Average time by gate choice (🟢 = faster, 🔴 = slower)")
                
                # Gate summary table with color gradient
                gate_summary = create_gate_summary_table(leg_summary_df, leg_type='uw')
                if gate_summary is not None:
                    st.dataframe(gate_summary, use_container_width=True)
                else:
                    st.info("No gate data available")
                
                # Additional stats
                st.markdown("**Statistics**")
                st.metric("Total UW Legs", len(uw_summary))
                st.metric("Avg Time", f"{uw_summary['time_seconds'].mean():.1f}s")
                st.metric("Avg BSP", f"{uw_summary['avg_BSP'].mean():.1f} km/h")
    
    with dw_tab:
        st.markdown("### Downwind Legs Analysis")
        
        # Check if we have downwind data
        dw_summary = leg_summary_df[leg_summary_df["leg_type"] == "dw"]
        if len(dw_summary) == 0:
            st.warning("No downwind legs detected in this race.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Rotated Leg Tracks**")
                st.caption("Line width = speed (thicker = faster), Color = TWD deviation from mean")
                
                # Create plot
                dw_plot = create_rotated_legs_plot(legs_df, leg_summary_df, leg_type='dw')
                if dw_plot:
                    st.image(dw_plot, use_container_width=True)
                else:
                    st.error("Could not generate downwind plot")
            
            with col2:
                st.markdown("**Gate Performance**")
                st.caption("Average time by gate choice (🟢 = faster, 🔴 = slower)")
                
                # Gate summary table with color gradient
                gate_summary = create_gate_summary_table(leg_summary_df, leg_type='dw')
                if gate_summary is not None:
                    st.dataframe(gate_summary, use_container_width=True)
                else:
                    st.info("No gate data available")
                
                # Additional stats
                st.markdown("**Statistics**")
                st.metric("Total DW Legs", len(dw_summary))
                st.metric("Avg Time", f"{dw_summary['time_seconds'].mean():.1f}s")
                st.metric("Avg BSP", f"{dw_summary['avg_BSP'].mean():.1f} km/h")
