"""
SailGP Race Analysis Dashboard - COMPLETE UPDATED VERSION
Professional UI for analyzing race performance data with:
- Color-coded tables (green=good, red=bad)
- Tactical analysis tab with upwind/downwind visualizations
- Enhanced PDF generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from pathlib import Path
import io
import geopy
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

# Tactical analysis module (tab 4) - FIXED VERSION
import sys
sys.path.append('/mnt/user-data/outputs')
try:
    from tactical_analysis_fixed import render_tactical_tab
    TACTICAL_AVAILABLE = True
except ImportError:
    TACTICAL_AVAILABLE = False
    print("Warning: tactical_analysis_fixed.py not found")

# ============================================================================
# COLOR MAPPING
# ============================================================================

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

# ============================================================================
# HELPER FUNCTIONS - GEOMETRY & WIND
# ============================================================================

def mean_bearing(bearing1, bearing2):
    """Calculate mean of two bearings accounting for circular nature"""
    bearing1_rad = math.radians(bearing1)
    bearing2_rad = math.radians(bearing2)
    
    sin1, cos1 = math.sin(bearing1_rad), math.cos(bearing1_rad)
    sin2, cos2 = math.sin(bearing2_rad), math.cos(bearing2_rad)
    
    sin_mean = (sin1 + sin2) / 2
    cos_mean = (cos1 + cos2) / 2
    
    mean_bearing_rad = math.atan2(sin_mean, cos_mean)
    mean_bearing_deg = math.degrees(mean_bearing_rad)
    
    if mean_bearing_deg < 0:
        mean_bearing_deg += 360
        
    return mean_bearing_deg

def subtract_angles(angle1, angle2):
    """Subtract two angles and wrap to -180 to +180"""
    result = angle1 - angle2
    while result > 180:
        result -= 360
    while result <= -180:
        result += 360
    return result

def get_gradient_color(value, min_val, max_val, reverse=False):
    """
    Generate RGB color for gradient
    reverse=False: green (high) to red (low) - for performance metrics
    reverse=True: green (low) to red (high) - for stability metrics
    """
    if max_val == min_val:
        return "background-color: #ffffff"
    
    normalized = (value - min_val) / (max_val - min_val)
    
    if reverse:
        normalized = 1 - normalized
    
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

def get_blue_gradient(value, min_val, max_val):
    """Generate SUBTLE blue gradient: very light blue, keeps text readable"""
    if max_val == min_val:
        return "background-color: #E3F2FD"
    
    normalized = (value - min_val) / (max_val - min_val)
    
    # Very light blue gradient: #E3F2FD (light) to #90CAF9 (medium light)
    # Keeping it light so dark text remains readable
    r = int(227 - normalized * (227 - 144))
    g = int(242 - normalized * (242 - 202))
    b = int(253 - normalized * (253 - 249))
    
    return f"background-color: rgb({r}, {g}, {b}); color: #000000"

# ============================================================================
# DATA PROCESSING
# ============================================================================

@st.cache_data
def load_race_data(csv_path):
    """Load and return race data with caching"""
    return pd.read_csv(csv_path)

def get_unique_race_ids(df):
    """Extract unique valid race IDs from dataframe"""
    race_ids = df['TRK_RACE_NUM_unk'].dropna().unique()
    race_ids = [rid for rid in race_ids if rid > 0]
    return sorted(race_ids)

def filter_valid_legs(df, race_id, boats):
    """Filter out invalid legs"""
    race_df = df[df['TRK_RACE_NUM_unk'] == race_id].copy()
    race_df = race_df[race_df['TRK_LEG_NUM_unk'] > 0]
    
    leg_stats = []
    for leg_num in sorted(race_df['TRK_LEG_NUM_unk'].unique()):
        leg_data = race_df[race_df['TRK_LEG_NUM_unk'] == leg_num]
        
        valid_boats = 0
        for boat in boats:
            boat_leg = leg_data[leg_data['BOAT'] == boat]
            if len(boat_leg) > 0:
                duration = boat_leg['DATETIME'].iloc[-1] if len(boat_leg) > 1 else 0
                if duration != 0:
                    valid_boats += 1
        
        if valid_boats >= len(boats):
            leg_stats.append(leg_num)
    
    return sorted(leg_stats)

def calculate_leg_summary(leg_data, boat):
    """Calculate summary statistics for a single boat-leg"""
    boat_data = leg_data[leg_data['BOAT'] == boat].copy()
    mean_TWD_leg = leg_data["TWD_MHU_SGP_deg"].mean()
    if len(boat_data) == 0:
        return None
    
    time_seconds = len(boat_data)
    
    coords = boat_data[['LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk']].values
    total_distance = 0
    for i in range(1, len(coords)):
        total_distance += geodesic(coords[i-1], coords[i]).meters
    
    avg_BSP = boat_data['BOAT_SPEED_km_h_1'].mean()
    avg_VMG = abs(boat_data['VMG_km_h_1'].mean())
    avg_TWD = boat_data['TWD_MHU_SGP_deg'].mean()
    avg_TWS = boat_data['TWS_MHU_SGP_km_h_1'].mean()
    # in-phase %
    delta_twd = mean_TWD_leg - boat_data["TWD_MHU_SGP_deg"]
    stbd = boat_data["TWA_MHU_SGP_deg"] > 0
    port = boat_data["TWA_MHU_SGP_deg"] < 0
    in_phase = (stbd & (delta_twd < 0)) | (port & (delta_twd > 0))
    pct_time_in_phase = in_phase.mean() * 100.0
    
    heel_stability = 0
    rh_stability = 0
    if 'HEEL_deg' in boat_data.columns:
        heel_stability = boat_data['HEEL_deg'].rolling(10).std().mean()
    if 'RH_LEE' in boat_data.columns:
        rh_stability = boat_data['RH_LEE'].rolling(10).std().mean()
    
    return {
        'time_seconds': round(time_seconds, 1),
        'total_distance_m': round(total_distance, 1),
        'avg_BSP': round(avg_BSP, 1),
        'avg_VMG': round(avg_VMG, 1),
        'avg_TWD': round(avg_TWD, 1),
        'avg_TWS': round(avg_TWS, 1),
        'pct_time_in_phase': round(pct_time_in_phase, 1),
        'heel_stability': round(heel_stability, 2) if heel_stability else 0,
        'rh_stability': round(rh_stability, 2) if rh_stability else 0
    }


def compute_tactical_dfs(race_data: pd.DataFrame, boat: str, min_points: int = 30):
    """Build legs_df + leg_summary_df for the Tactical tab.

    This app doesn't always ship with explicit mark/gate positions.
    We therefore infer a simple L/R 'gate' label from the rotated track:
    - rotate points by the leg's mean TWD
    - take the median rotated X; x>=0 => 'R', else 'L'

    Expected columns in race_data:
      - TRK_LEG_NUM_unk, BOAT
      - LATITUDE_GPS_unk, LONGITUDE_GPS_unk
      - TWD_MHU_SGP_deg
      - BOAT_SPEED_km_h_1, VMG_km_h_1 (optional but recommended)
      - DATETIME (optional, used to compute duration)
    """
    required = ["TRK_LEG_NUM_unk", "BOAT", "LATITUDE_GPS_unk", "LONGITUDE_GPS_unk", "TWD_MHU_SGP_deg"]
    missing = [c for c in required if c not in race_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for Tactical Analysis: {missing}")

    boat_df = race_data[race_data["BOAT"] == boat].copy()
    boat_df = boat_df[boat_df["TRK_LEG_NUM_unk"] > 0].copy()
    if boat_df.empty:
        return None, None

    legs_rows = []
    summary_rows = []

    for leg_num in sorted(boat_df["TRK_LEG_NUM_unk"].dropna().unique()):
        leg_df = boat_df[boat_df["TRK_LEG_NUM_unk"] == leg_num].copy()
        leg_df = leg_df.dropna(subset=["LATITUDE_GPS_unk", "LONGITUDE_GPS_unk", "TWD_MHU_SGP_deg"])
        if len(leg_df) < min_points:
            continue

        leg_id = int(leg_num)
        leg_df["leg_id"] = leg_id

        # Duration
        if "DATETIME" in leg_df.columns:
            try:
                t0 = float(leg_df["DATETIME"].iloc[0])
                t1 = float(leg_df["DATETIME"].iloc[-1])
                time_seconds = max(1.0, t1 - t0)
            except Exception:
                time_seconds = float(len(leg_df))
        else:
            time_seconds = float(len(leg_df))

        # Averages
        avg_twd = float(leg_df["TWD_MHU_SGP_deg"].astype(float).mean())
        avg_bsp = float(leg_df["BOAT_SPEED_km_h_1"].astype(float).mean()) if "BOAT_SPEED_km_h_1" in leg_df.columns else np.nan

        # Leg type from VMG sign if available, else alternating (odd=uw, even=dw)
        if "VMG_km_h_1" in leg_df.columns:
            try:
                vmg_mean = float(leg_df["VMG_km_h_1"].astype(float).mean())
                leg_type = "uw" if vmg_mean >= 0 else "dw"
            except Exception:
                leg_type = "uw" if (leg_id % 2 == 1) else "dw"
        else:
            leg_type = "uw" if (leg_id % 2 == 1) else "dw"

        # Infer gate side (L/R) from rotated track
        ang = np.deg2rad(avg_twd)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        lon = leg_df["LONGITUDE_GPS_unk"].to_numpy(dtype=float)
        lat = leg_df["LATITUDE_GPS_unk"].to_numpy(dtype=float)
        lon0, lat0 = np.nanmean(lon), np.nanmean(lat)
        x = lon - lon0
        y = lat - lat0
        x_rot = x * cos_a - y * sin_a
        gate = "R" if np.nanmedian(x_rot) >= 0 else "L"

        legs_rows.append(leg_df)

        summary_rows.append({
            "leg_id": leg_id,
            "leg_type": leg_type,
            "gate": gate,
            "time_seconds": time_seconds,
            "avg_BSP": avg_bsp,
            "avg_TWD": avg_twd
        })

    if not legs_rows or not summary_rows:
        return None, None

    legs_df = pd.concat(legs_rows, ignore_index=True)
    leg_summary_df = pd.DataFrame(summary_rows)

    return legs_df, leg_summary_df

def rotate_coordinates(x, y, angle_deg):
    """Rotate coordinates by angle in degrees"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    return x_rot, y_rot

def prepare_track_data(leg_data, boats):
    """Prepare rotated track data aligned to mean TWD"""
    mean_twd = leg_data['TWD_MHU_SGP_deg'].mean()
    
    tracks = {}
    for boat in boats:
        boat_data = leg_data[leg_data['BOAT'] == boat].copy()
        if len(boat_data) == 0:
            continue
        
        lat = boat_data['LATITUDE_GPS_unk'].values
        lon = boat_data['LONGITUDE_GPS_unk'].values
        
        lat_mean = lat.mean()
        x = (lon - lon[0]) * 111320 * math.cos(math.radians(lat_mean))
        y = (lat - lat[0]) * 111320
        
        tracks[boat] = {
            'x': x,
            'y': y,
            'bsp': boat_data['BOAT_SPEED_km_h_1'].values,
            'boat': boat
        }
    
    return tracks, mean_twd

# ============================================================================
# IDW INTERPOLATION FOR WIND MAPS
# ============================================================================

def idw_interpolation(points, values, xi, yi, power=2.8):
    """IDW interpolation for spatial wind data"""
    points = np.asarray(points)
    values = np.asarray(values)
    xi = np.asarray(xi)
    yi = np.asarray(yi)

    dist = np.sqrt((points[:, 0, np.newaxis] - xi[np.newaxis, :])**2 +
                   (points[:, 1, np.newaxis] - yi[np.newaxis, :])**2)

    dist[dist == 0] = 1e-10
    weights = 1 / dist**power
    
    zi = np.sum(np.nan_to_num(weights * values[:, np.newaxis], nan=0.0), axis=0) / np.sum(weights, axis=0)

    return zi

def interpolate_wind_idw(value_col, df, lat_grid, lon_grid):
    """Interpolate wind values using IDW"""
    points = np.array(df[['LONGITUDE_GPS_unk', 'LATITUDE_GPS_unk']])
    values = np.array(df[value_col])
    
    grid_z = idw_interpolation(points, values, lon_grid.ravel(), lat_grid.ravel())
    
    return grid_z.reshape(lat_grid.shape)

def create_wind_map(race_data, value_col, title, colorscale="RdYlGn"):
    """Create wind heatmap using IDW interpolation"""
    log_ = race_data.copy()
    
    lat_min, lat_max = log_['LATITUDE_GPS_unk'].quantile(.15), log_['LATITUDE_GPS_unk'].quantile(.9)
    lon_min, lon_max = log_['LONGITUDE_GPS_unk'].quantile(.15), log_['LONGITUDE_GPS_unk'].quantile(.9)
    
    lat_grid, lon_grid = np.meshgrid(np.linspace(lat_min, lat_max, 100), 
                                     np.linspace(lon_min, lon_max, 100))
    
    grid_values = interpolate_wind_idw(value_col, log_, lat_grid, lon_grid)
    
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    values_flat = grid_values.ravel()
    
    df_interpolated = pd.DataFrame({
        'LATITUDE_GPS_unk': lat_flat, 
        'LONGITUDE_GPS_unk': lon_flat, 
        value_col: values_flat
    })
    
    fig = px.scatter_mapbox(
        df_interpolated, 
        lat="LATITUDE_GPS_unk", 
        lon="LONGITUDE_GPS_unk",
        color=value_col, 
        color_continuous_scale=colorscale,
        title=title,
        zoom=14.5
    )
    
    mean_twd = log_['TWD_MHU_SGP_deg'].mean()
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(bearing=mean_twd),
        margin={"r":0, "t":40, "l":0, "b":0},
        height=500
    )
    
    return fig

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_leg_tracks(leg_data, boats, leg_num):
    """Create track plot colored by boat only"""
    tracks, mean_twd = prepare_track_data(leg_data, boats)
    
    if not tracks:
        return None
    
    fig = go.Figure()
    
    for boat in boats:
        if boat not in tracks:
            continue
        
        track = tracks[boat]
        boat_color = COLOR_MAPPING.get(boat, '#888888')
        
        fig.add_trace(go.Scatter(
            x=track['x'],
            y=track['y'],
            mode='lines',
            name=boat,
            line=dict(color=boat_color, width=3),
            hovertemplate=f'<b>{boat}</b><br>X: %{{x:.1f}} m<br>Y: %{{y:.1f}} m<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Leg {leg_num} - Rotated Tracks (Aligned to mean TWD = {mean_twd:.1f}°)",
        xaxis_title="X (m, rotated)",
        yaxis_title="Y (m, rotated - wind ↑)",
        template="plotly_white",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    return fig

def plot_wind_analysis(race_data, boat='AUS'):
    """Create TWS/TWD time series plot with leg markers"""
    boat_data = race_data[race_data['BOAT'] == boat].copy()
    boat_data = boat_data[boat_data['TRK_LEG_NUM_unk'] > 0].sort_index()
    
    if len(boat_data) == 0:
        return None
    
    tws_min, tws_max = boat_data['TWS_MHU_SGP_km_h_1'].min(), boat_data['TWS_MHU_SGP_km_h_1'].max()
    twd_min, twd_max = boat_data['TWD_MHU_SGP_deg'].min(), boat_data['TWD_MHU_SGP_deg'].max()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=boat_data.index,
        y=boat_data['TWS_MHU_SGP_km_h_1'].rolling(5).mean(),
        mode='lines',
        name='TWS (kt)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=boat_data.index,
        y=boat_data['TWD_MHU_SGP_deg'].rolling(5).mean(),
        mode='lines',
        name='TWD (°)',
        line=dict(color='#d62728', width=2),
        yaxis='y2'
    ))
    
    transition_label = {
        (0, 1): "Start", (1, 2): "M1", (2, 3): "LG", (3, 4): "WG",
        (4, 5): "LG", (5, 6): "WG", (6, 7): "M1", (7, 8): "Finish"
    }
    
    label_color = {
        "Start": "#2ca02c", "M1": "#9467bd", "LG": "#ff7f0e",
        "WG": "#8c564b", "Finish": "#000000"
    }
    
    leg_series = boat_data['TRK_LEG_NUM_unk']
    added_labels = set()
    prev_leg = leg_series.iloc[0]
    
    for idx, leg in leg_series.iloc[1:].items():
        if leg != prev_leg:
            key = (prev_leg, leg)
            if key in transition_label:
                label = transition_label[key]
                color = label_color[label]
                
                fig.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[tws_min - 1, tws_max + 1],
                    mode='lines',
                    line=dict(color=color, width=1.5, dash='dash'),
                    name=label,
                    showlegend=(label not in added_labels)
                ))
                added_labels.add(label)
            
            prev_leg = leg
    
    fig.update_layout(
        title=f"True Wind Speed (TWS) and True Wind Direction (TWD) – {boat}",
        xaxis=dict(title="Time / Frame Index"),
        yaxis=dict(
            title="TWS (kt)",
            tickfont=dict(color="#1f77b4"),
            side="left",
            range=[tws_min - 1, tws_max + 1]
        ),
        yaxis2=dict(
            title="TWD (°)",
            tickfont=dict(color="#d62728"),
            overlaying="y",
            side="right",
            range=[twd_min - 5, twd_max + 5]
        ),
        legend=dict(x=0.01, y=0.99),
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_summary_table_styled(leg_data, boats, leg_num):
    """Create styled summary table with COLOR GRADIENTS"""
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary:
            summaries[boat] = summary
    
    if not summaries:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(summaries)
    df.index = ['Time (s)', 'Distance (m)', 'Avg BSP', 'Avg VMG', 'Avg TWD', 'Avg TWS', '% In Phase', 'Num of man', 'Distance Made Good']
    
    # Define which metrics should have which gradient
    higher_is_better = {'Avg BSP', 'Avg VMG', '% In Phase'}
    lower_is_better = {'Time (s)', 'Distance (m)'}
    
    # Create styler function
    def apply_gradient(row):
        styles = []
        row_name = row.name
        
        for col in df.columns:
            value = row[col]
            min_val = row.min()
            max_val = row.max()
            
            if row_name in higher_is_better:
                # High = green, low = red
                styles.append(get_gradient_color(value, min_val, max_val, reverse=False))
            elif row_name in lower_is_better:
                # Low = green, high = red
                styles.append(get_gradient_color(value, min_val, max_val, reverse=True))
            else:
                # Neutral - no color
                styles.append("")
        
        return styles
    
    styled_df = df.style.apply(apply_gradient, axis=1)
    
    return styled_df

def create_stability_table_styled(leg_data, boats):
    """Create stability table with COLOR GRADIENTS (lower is better)"""
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary and (summary['heel_stability'] != 0 or summary['rh_stability'] != 0):
            summaries[boat] = {
                'Heel Stab': summary['heel_stability'],
                'RH Stab': summary['rh_stability']
            }
    
    if not summaries:
        return None
    
    df = pd.DataFrame(summaries)
    
    # For stability: lower is better (more stable)
    def apply_stability_gradient(row):
        styles = []
        min_val = row.min()
        max_val = row.max()
        
        for value in row:
            # Reverse=True: low values = green, high values = red
            styles.append(get_gradient_color(value, min_val, max_val, reverse=True))
        
        return styles
    
    styled_df = df.style.apply(apply_stability_gradient, axis=1)
    
    return styled_df

# ============================================================================
# TACTICAL ANALYSIS FUNCTIONS
# ============================================================================

def create_rotated_legs_plot(legs_df, leg_summary_df, leg_type='uw'):
    """Create rotated legs plot with TWD deviation coloring"""
    summary = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    legs = legs_df[legs_df["leg_id"].isin(summary["leg_id"])].copy()
    
    if len(summary) == 0 or len(legs) == 0:
        return None
    
    # Calculate rotation angle
    rotation_angle = np.deg2rad(summary["avg_TWD"].mean())
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    
    # Line width scaling
    time_values = summary.set_index("leg_id")["time_seconds"].astype(float)
    q75 = time_values.quantile(0.75)
    q25 = time_values.quantile(0.25)
    den = (q75 - q25) if (q75 - q25) != 0 else 1.0
    
    # Global color normalization
    twd_mean = summary["avg_TWD"].mean()
    twd_dev_all = (legs["TWD_MHU_SGP_deg"].astype(float) - twd_mean).to_numpy()
    
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each leg
    for legID in summary.sort_values("time_seconds")["leg_id"].unique():
        leg_data = legs[legs["leg_id"] == legID].copy()
        if len(leg_data) < 2:
            continue
        
        lon = leg_data["LONGITUDE_GPS_unk"].to_numpy(dtype=float)
        lat = leg_data["LATITUDE_GPS_unk"].to_numpy(dtype=float)
        
        # Rotate around centroid
        lon0, lat0 = np.nanmean(lon), np.nanmean(lat)
        x = lon - lon0
        y = lat - lat0
        
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        
        # Line width
        leg_time = float(time_values.loc[legID])
        linewidth = 1 + 9 * (q75 - leg_time) / den
        linewidth = float(np.clip(linewidth, 1, 10))
        
        # Color by TWD deviation
        twd_dev = (leg_data["TWD_MHU_SGP_deg"].to_numpy(dtype=float) - twd_mean)
        seg_colors = cmap(norm(twd_dev))
        
        # Plot segments
        for i in range(len(x_rot) - 1):
            ax.plot(
                x_rot[i:i+2], y_rot[i:i+2],
                color=tuple(seg_colors[i]),
                linewidth=linewidth,
                alpha=0.8
            )
    
    ax.set_xlabel("Rotated X (lon)", fontsize=12)
    ax.set_ylabel("Rotated Y (lat)", fontsize=12)
    ax.set_title(
        f"Detected {'Upwind' if leg_type == 'uw' else 'Downwind'} Legs\n" +
        f"(Rotated by {np.rad2deg(rotation_angle):.1f}°)",
        fontsize=14,
        fontweight='bold'
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="TWD deviation from mean (deg)")
    
    plt.tight_layout()
    
    return fig

def create_gate_summary_table(leg_summary_df, leg_type='uw'):
    """Create gate summary table"""
    filtered = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    
    if len(filtered) == 0 or 'gate' not in filtered.columns:
        return None
    
    summary = filtered.groupby("gate")["time_seconds"].agg(['mean', 'count']).round(1)
    summary.columns = ['Avg Time (s)', 'Count']
    
    return summary

# ============================================================================
# STREAMLIT UI
# ============================================================================

def apply_custom_css():
    """Apply custom CSS for professional styling"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        }
        
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            color: #1a2332 !important;
        }
        
        h1 {
            font-size: 2.8rem !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #0066cc 0%, #003d7a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: white;
            border-radius: 12px;
            padding: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e8f0fe;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0066cc 0%, #003d7a 100%);
            color: white !important;
        }
        
        .dataframe {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.9rem !important;
            border-radius: 8px !important;
        }
        
        .stButton button {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            background: linear-gradient(135deg, #0066cc 0%, #003d7a 100%);
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,102,204,0.3);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,102,204,0.4);
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="SailGP Race Analysis",
        page_icon="⛵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Header
    st.markdown("# ⛵ SailGP Race Analysis Dashboard")
    st.markdown("### Professional performance analysis for competitive sailing")
    st.markdown("---")
    
    # Sidebar for data selection
    with st.sidebar:
        st.markdown("## 📊 Data Selection")
        
        race_date = st.text_input(
            "Race Date (YYYYMMDD)",
            value="20260118",
            help="Enter the race date in format YYYYMMDD"
        )
        
        csv_path = f"/Users/hugostubler/Documents/SailGP /Report_Pipeline/data/logs/log_{race_date}.csv"
        
        if Path(csv_path).exists():
            df = load_race_data(csv_path)
            st.success(f"✅ Data loaded: {len(df):,} rows")
            df['RH_LEE'] = np.where(df['TWA_MHU_SGP_deg'] > 0, df.LENGTH_RH_P_mm, df.LENGTH_RH_S_mm)
            
            race_ids = get_unique_race_ids(df)
            if race_ids:
                selected_race_id = st.selectbox(
                    "Race ID",
                    options=race_ids,
                    format_func=lambda x: f"Race {int(x)}"
                )
                
                all_boats = sorted(df['BOAT'].dropna().unique())
                default_boats = ['AUS', 'FRA', 'GBR']
                default = default_boats if all(b in all_boats for b in default_boats) else all_boats[:3]

                selected_boats = st.multiselect(
                    "Select Boats",
                    options=all_boats,
                    default=default
                )
                                
                if selected_boats:
                    race_data = df[df['TRK_RACE_NUM_unk'] == selected_race_id]
                    race_data = race_data[race_data['BOAT'].isin(selected_boats)]
                    
                    valid_legs = filter_valid_legs(df, selected_race_id, selected_boats)
                    
                    selected_legs = st.multiselect(
                        "Select Legs",
                        options=valid_legs,
                        default=valid_legs
                    )
                    
                    st.markdown("---")
                    show_stability = st.checkbox("📊 Add Stability Metrics", value=False)
                    
                    st.markdown("---")
                    st.markdown("### 📈 Quick Stats")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Boats", len(selected_boats))
                    with col2:
                        st.metric("Legs", len(selected_legs))
                else:
                    st.warning("⚠️ Please select at least one boat")
                    selected_race_id = None
                    selected_legs = []
            else:
                st.error("❌ No valid race IDs found")
                selected_race_id = None
                selected_legs = []
        else:
            st.error(f"❌ File not found: {csv_path}")
            st.info("💡 Tip: Upload your CSV file or check the date format")
            selected_race_id = None
            selected_legs = []
    
    # Main content area
    if selected_race_id and selected_legs and selected_boats:
        
        # Tabs - UPDATED WITH TACTICAL TAB
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚀 Start", 
            "🏁 Legs", 
            "🌊 Wind", 
            "⚓ Tactical",  # NEW TAB
            "📄 Download PDF"
        ])
        
        with tab1:
            st.markdown("## Start Analysis")
            
            # Overview metrics
            st.markdown("### Race Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Race ID", f"{int(selected_race_id)}")
            with col2:
                st.metric("Total Legs", len(selected_legs))
            with col3:
                st.metric("Boats Racing", len(selected_boats))
            with col4:
                avg_tws = race_data['TWS_MHU_SGP_km_h_1'].mean()
                st.metric("Avg TWS", f"{avg_tws:.1f} kt")
            
            # Start metrics with BLUE GRADIENT
            st.markdown("### Start Metrics")
            
            try:
                df_start = race_data[(race_data['TTS_s'] < 90) & (race_data['TTS_s'] > -10)].copy()
                
                pc_ttk_values = {}
                pc_tts_values = {}
                pc_ratio_values = {}
                pc_dtl_values = {}
                
                race_num = selected_race_id
                pc_ttk_values[race_num] = {}
                pc_tts_values[race_num] = {}
                pc_ratio_values[race_num] = {}
                pc_dtl_values[race_num] = {}
                
                for boat in selected_boats:
                    df_subset = df_start[df_start['BOAT'] == boat]
                    df_subset_sorted = df_subset.sort_values(by="TTS_s")
                    
                    if 'LENGTH_DB_H_P_mm' in df_subset_sorted.columns:
                        board_values = df_subset_sorted["LENGTH_DB_H_P_mm"]
                        
                        if len(board_values[board_values.diff().abs() > 1]) > 0:
                            first_change_idx = board_values[board_values.diff().abs() > 1].idxmin()
                            first_change_row = df_start.loc[first_change_idx]
                            
                            if 'PC_TTK_s' in first_change_row and 'TTS_s' in first_change_row:
                                pc_ttk_values[race_num][boat] = first_change_row["PC_TTK_s"]
                                pc_tts_values[race_num][boat] = first_change_row["TTS_s"]
                                pc_ratio_values[race_num][boat] = first_change_row["TTS_s"] / first_change_row["PC_TTK_s"]
                                pc_dtl_values[race_num][boat] = first_change_row.get('PC_DTL_m', None)
                
                # Convert to DataFrames and apply BLUE GRADIENT
                df_pc_ttk = pd.DataFrame(pc_ttk_values).T
                df_pc_tts = pd.DataFrame(pc_tts_values).T
                df_pc_ratio = pd.DataFrame(pc_ratio_values).T
                df_pc_dtl = pd.DataFrame(pc_dtl_values).T
                
                # Apply blue gradient styling
                def apply_blue_gradient(row):
                    min_val = row.min()
                    max_val = row.max()
                    return [get_blue_gradient(val, min_val, max_val) for val in row]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**PC Time to Kill (TTK)**")
                    st.dataframe(df_pc_ttk.style.apply(apply_blue_gradient, axis=1), use_container_width=True)
                    
                    st.markdown("**PC Time to Start (TTS)**")
                    st.dataframe(df_pc_tts.style.apply(apply_blue_gradient, axis=1), use_container_width=True)
                
                with col2:
                    st.markdown("**PC Ratio (TTS/TTK)**")
                    st.dataframe(df_pc_ratio.style.apply(apply_blue_gradient, axis=1), use_container_width=True)
                    
                    st.markdown("**PC Distance to Line (DTL)**")
                    if not df_pc_dtl.empty:
                        st.dataframe(df_pc_dtl.style.apply(apply_blue_gradient, axis=1), use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating start metrics: {str(e)}")
                st.info("Start metrics require columns: TTS_s, PC_TTK_s, LENGTH_DB_H_P_mm, PC_DTL_m")
        
        with tab2:
            st.markdown("## 🏁 Leg-by-Leg Analysis")
            
            for leg_num in selected_legs:
                st.markdown(f"### Leg {int(leg_num)}")
                
                leg_data = race_data[race_data['TRK_LEG_NUM_unk'] == leg_num]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = plot_leg_tracks(leg_data, selected_boats, int(leg_num))
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Summary table with COLOR GRADIENTS
                    summary_styled = create_summary_table_styled(leg_data, selected_boats, int(leg_num))
                    if summary_styled is not None:
                        st.markdown(f"**Leg {int(leg_num)} Summary**")
                        st.dataframe(summary_styled, use_container_width=True)
                        
                        # Stability metrics with COLOR GRADIENTS
                        if show_stability:
                            st.markdown("**Stability Metrics**")
                            stability_styled = create_stability_table_styled(leg_data, selected_boats)
                            if stability_styled is not None:
                                st.dataframe(stability_styled, use_container_width=True)
                
                st.markdown("---")
        
        with tab3:
            st.markdown("## 🌊 Wind Tactical Analysis")
            
            wind_boat = st.selectbox(
                "Select boat for wind analysis",
                options=selected_boats,
                index=0
            )
            
            fig = plot_wind_analysis(race_data, wind_boat)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No wind data available for selected boat")
            
            st.markdown("### Wind Statistics")
            
            wind_stats_data = {}
            for boat in selected_boats:
                boat_data = race_data[race_data['BOAT'] == boat]
                twd_rolling = boat_data['TWD_MHU_SGP_deg'].rolling(60)
                
                wind_stats_data[boat] = {
                    'TWD Avg': boat_data['TWD_MHU_SGP_deg'].mean(),
                    'TWD Min (60s)': twd_rolling.min().min(),
                    'TWD Max (60s)': twd_rolling.max().max(),
                    'TWS Avg': boat_data['TWS_MHU_SGP_km_h_1'].mean()
                }
            
            wind_stats_df = pd.DataFrame(wind_stats_data).T.round(1)
            st.dataframe(wind_stats_df, use_container_width=True)
            
            st.markdown("### Wind Maps (IDW Interpolation)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**True Wind Speed (TWS)**")
                try:
                    tws_fig = create_wind_map(
                        race_data, 
                        'TWS_MHU_SGP_km_h_1', 
                        'TWS Distribution',
                        colorscale='Viridis'
                    )
                    st.plotly_chart(tws_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating TWS map: {str(e)}")
            
            with col2:
                st.markdown("**True Wind Direction (TWD)**")
                try:
                    twd_fig = create_wind_map(
                        race_data, 
                        'TWD_MHU_SGP_deg', 
                        'TWD Distribution',
                        colorscale='RdYlGn'
                    )
                    st.plotly_chart(twd_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating TWD map: {str(e)}")
        
        # NEW TACTICAL TAB
        with tab4:
            st.markdown("## ⚓ Tactical Analysis")

            # Pick a boat for tactical analysis (default: first selected boat)
            tactical_boat = st.selectbox(
                "Select boat for tactical analysis",
                options=selected_boats,
                index=0,
                key="tactical_boat_select"
            )

            # Build (or reuse) tactical data
            cache_key = f"tactical::{int(selected_race_id)}::{tactical_boat}"
            if st.session_state.get("tactical_cache_key") != cache_key:
                st.session_state.pop("legs_df", None)
                st.session_state.pop("leg_summary_df", None)
                st.session_state["tactical_cache_key"] = cache_key

            if "legs_df" not in st.session_state or "leg_summary_df" not in st.session_state:
                with st.spinner("Building tactical legs data..."):
                    try:
                        legs_df, leg_summary_df = compute_tactical_dfs(race_data, tactical_boat)
                        if legs_df is None or leg_summary_df is None:
                            st.warning("Not enough data to build tactical legs (try another boat or race).")
                        else:
                            st.session_state["legs_df"] = legs_df
                            st.session_state["leg_summary_df"] = leg_summary_df
                    except Exception as e:
                        st.error(f"Tactical analysis setup failed: {e}")

            legs_df = st.session_state.get("legs_df")
            leg_summary_df = st.session_state.get("leg_summary_df")

            if legs_df is not None and leg_summary_df is not None:
                if TACTICAL_AVAILABLE:
                    render_tactical_tab(legs_df, leg_summary_df)
                else:
                    st.error("Tactical analysis module not found. Please ensure tactical_analysis_fixed.py is in /mnt/user-data/outputs/")
            else:
                st.info(
                    "Tactical analysis needs GPS + TWD (and ideally VMG/BSP) per leg. "
                    "If your dataset doesn't include enough points per leg, it will show this message."
                )

        with tab5:
            st.markdown("## 📄 Download Race Report")
            st.markdown("Generate a professional PDF report with all race analysis.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        try:
                            # Import improved PDF generator
                            import sys
                            sys.path.append('/mnt/user-data/outputs')
                            from sailing_race_pdf_improved import generate_race_report_pdf
                            
                            pdf_buffer = generate_race_report_pdf(
                                df, selected_race_id, selected_boats, selected_legs
                            )
                            
                            st.success("✅ PDF generated successfully!")
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"race_report_{int(selected_race_id)}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            st.info("Make sure sailing_race_pdf_improved.py is available")
    else:
        st.info("👈 Please select race data from the sidebar to begin analysis")

if __name__ == "__main__":
    main()
