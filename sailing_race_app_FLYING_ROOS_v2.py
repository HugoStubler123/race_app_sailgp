"""
BONDS FLYING ROOS - SailGP Race Analysis Dashboard
Professional sailing performance analysis with corporate branding
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
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from gate_crossing_analyzer import analyze_gate_crossings

# ============================================================================
# FLYING ROOS COLOR PALETTE
# ============================================================================

FLYING_ROOS_COLORS = {
    "primary_teal": "#006B5E",      # Dark teal for headers
    "deep_green": "#004D43",         # Secondary elements
    "gold": "#FFD700",               # Accents, AUS team highlight
    "light_teal": "#00A896",         # Backgrounds
    "white": "#FFFFFF",
    "light_gray": "#F5F5F5",
    "dark_gray": "#333333"
}

# PRESERVE boat color mapping (DO NOT TOUCH - as requested)
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
# HELPER FUNCTIONS
# ============================================================================

def format_number(value, decimals=1):
    """Format numbers to max 1 decimal place"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def mean_bearing(bearing1, bearing2):
    """Calculate mean of two bearings"""
    bearing1_rad = math.radians(bearing1)
    bearing2_rad = math.radians(bearing2)
    sin_mean = (math.sin(bearing1_rad) + math.sin(bearing2_rad)) / 2
    cos_mean = (math.cos(bearing1_rad) + math.cos(bearing2_rad)) / 2
    mean_bearing_rad = math.atan2(sin_mean, cos_mean)
    mean_bearing_deg = math.degrees(mean_bearing_rad)
    if mean_bearing_deg < 0:
        mean_bearing_deg += 360
    return mean_bearing_deg

def get_gradient_color(value, min_val, max_val, reverse=False):
    """Generate RGB color for gradient"""
    if max_val == min_val:
        return "background-color: #ffffff"
    normalized = (value - min_val) / (max_val - min_val)
    if reverse:
        normalized = 1 - normalized
    if normalized < 0.5:
        r = int(normalized * 2 * 255)
        g = 255
        b = 0
    else:
        r = 255
        g = int((1 - (normalized - 0.5) * 2) * 255)
        b = 0
    return f"background-color: rgb({r}, {g}, {b})"

def get_gradient_color(value, min_val, max_val, reverse=False):
    """Generate RGB color for green-yellow-red gradient"""

    if max_val == min_val:
        return "background-color: #ffffff"

    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))  # clamp

    if not reverse:
        # Green → Yellow → Red
        if normalized < 0.5:
            r = int(normalized * 2 * 255)
            g = 255
            b = 0
        else:
            r = 255
            g = int((1 - (normalized - 0.5) * 2) * 255)
            b = 0
    else:
        # Red → Yellow → Green
        if normalized < 0.5:
            r = 255
            g = int(normalized * 2 * 255)
            b = 0
        else:
            r = int((1 - (normalized - 0.5) * 2) * 255)
            g = 255
            b = 0

    return f"background-color: rgb({r}, {g}, {b})"


def get_subtle_gradient(value, min_val, max_val):
    """Subtle green gradient using primary_teal as max color - readable text"""
    
    primary_teal = (0, 107, 94)          # #006B5E
    light_green = (230, 245, 240)        # very soft light green background
    
    if max_val == min_val:
        return f"background-color: rgb{light_green}; color: #000000"
    
    normalized = (value - min_val) / (max_val - min_val)
    
    r = int(light_green[0] + normalized * (primary_teal[0] - light_green[0]))
    g = int(light_green[1] + normalized * (primary_teal[1] - light_green[1]))
    b = int(light_green[2] + normalized * (primary_teal[2] - light_green[2]))
    
    return f"background-color: rgb({r}, {g}, {b}); color: #000000"


# ============================================================================
# DATA PROCESSING
# ============================================================================

@st.cache_data
def load_race_data(csv_path):
    """Load and return race data with caching"""
    return pd.read_csv(csv_path)

def get_unique_race_ids(df):
    """Extract unique valid race IDs"""
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
    """Calculate summary statistics for a boat-leg"""
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
    heel_stability = boat_data['HEEL_deg'].rolling(10).std().mean() if 'HEEL_deg' in boat_data.columns else 0
    rh_stability = boat_data['RH_LEE'].rolling(10).std().mean() if 'RH_LEE' in boat_data.columns else 0
    delta_twd = mean_TWD_leg - boat_data["TWD_MHU_SGP_deg"]
    stbd = boat_data["TWA_MHU_SGP_deg"] > 0
    port = boat_data["TWA_MHU_SGP_deg"] < 0
    in_phase = (stbd & (delta_twd < 0)) | (port & (delta_twd > 0))
    pct_time_in_phase = in_phase.mean() * 100.0

    return {
        'time_seconds': time_seconds,
        'total_distance_m': total_distance,
        'avg_BSP': avg_BSP,
        'avg_VMG': avg_VMG,
        'avg_TWD': avg_TWD,
        'avg_TWS': avg_TWS,
        'pct_time_in_phase': pct_time_in_phase,
        'heel_stability': heel_stability,
        'rh_stability': rh_stability
    }

def rotate_coordinates(x, y, angle_deg):
    """Rotate coordinates by angle"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    return x_rot, y_rot

def prepare_track_data(leg_data, boats):
    """Prepare rotated track data"""
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
        tracks[boat] = {'x': x, 'y': y, 'boat': boat}
    return tracks, mean_twd
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
    """Create track plot"""
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
            x=track['x'], y=track['y'],
            mode='lines', name=boat,
            line=dict(color=boat_color, width=3),
            hovertemplate=f'<b>{boat}</b><br>X: %{{x:.1f}} m<br>Y: %{{y:.1f}} m<extra></extra>'
        ))
    fig.update_layout(
        title=f"Leg {leg_num} - Tracks (Aligned to TWD = {mean_twd:.1f}°)",
        xaxis_title="X (m, rotated)",
        yaxis_title="Y (m, rotated - wind ↑)",
        template="plotly_white",
        height=600,
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    return fig


def plot_leg_tracks(leg_data, boats, leg_num):
    """Create track plot oriented with mean wind coming from top (upward)."""

    tracks, mean_twd = prepare_track_data(leg_data, boats)
    if not tracks:
        return None

    fig = go.Figure()

    # Convert wind direction to radians
    # TWD is direction wind COMES FROM (meteorological)
    # We want that direction aligned with +Y axis (upward)
    theta = np.radians(mean_twd)

    # Rotation angle:
    # To align wind-from direction with +Y axis,
    # we rotate coordinates by (90° - mean_twd)
    rot_angle = np.radians(90 - mean_twd)

    cos_a = np.cos(rot_angle)
    sin_a = np.sin(rot_angle)

    for boat in boats:
        if boat not in tracks:
            continue

        track = tracks[boat]
        boat_color = COLOR_MAPPING.get(boat, '#888888')

        # Original coordinates
        x = np.array(track['x'])
        y = np.array(track['y'])

        # Rotate coordinates
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        fig.add_trace(go.Scatter(
            x=x_rot,
            y=y_rot,
            mode='lines',
            name=boat,
            line=dict(color=boat_color, width=3),
            hovertemplate=f'<b>{boat}</b><br>X: %{{x:.1f}} m<br>Y: %{{y:.1f}} m<extra></extra>'
        ))

    fig.update_layout(
        title=f"Leg {leg_num} - Tracks (Wind from {mean_twd:.1f}° ↑)",
        xaxis_title="Cross-wind (m)",
        yaxis_title="Upwind (m)",
        template="plotly_white",
        height=600,
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig


def plot_wind_analysis(race_data, boat='AUS'):
    """Create TWS/TWD time series plot"""
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
        mode='lines', name='TWS (kt)',
        line=dict(color=FLYING_ROOS_COLORS['primary_teal'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=boat_data.index,
        y=boat_data['TWD_MHU_SGP_deg'].rolling(5).mean(),
        mode='lines', name='TWD (°)',
        line=dict(color=FLYING_ROOS_COLORS['gold'], width=2),
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
        title=f"Wind Analysis - {boat}",
        xaxis=dict(title="Time"),
        yaxis=dict(title="TWS (kt)", side="left", range=[tws_min - 1, tws_max + 1]),
        yaxis2=dict(title="TWD (°)", overlaying="y", side="right", range=[twd_min - 5, twd_max + 5]),
        height=500,
        template="plotly_white"
    )
    return fig

def create_summary_table_styled(leg_data, boats, leg_num):
    """Create styled summary table with gradients"""
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary:
            summaries[boat] = summary
    if not summaries:
        return None
    df = pd.DataFrame(summaries)
    
    # Format to 1 decimal max
    for idx in df.index:
        for col in df.columns:
            df.at[idx, col] = round(df.at[idx, col], 1)
    
    df.index = ['Time (s)', 'Distance (m)', 'Avg BSP', 'Avg VMG', 'Avg TWD', 'Avg TWS', '% In Phase', 'Num of man', 'Dist Made Good']
    higher_is_better = {'Avg BSP', 'Avg VMG', '% In Phase','Dist Made Good','Avg TWD', 'Avg TWS'}
    lower_is_better = {'Time (s)', 'Distance (m)', 'Num of man'}
    
    def apply_gradient(row):
        styles = []
        row_name = row.name
        for col in df.columns:
            value = row[col]
            min_val = row.min()
            max_val = row.max()
            if row_name in higher_is_better:
                styles.append(get_gradient_color(value, min_val, max_val, reverse=False))
            elif row_name in lower_is_better:
                styles.append(get_gradient_color(value, min_val, max_val, reverse=True))
            else:
                styles.append("")
        return styles
    
    styled_df = df.style.apply(apply_gradient, axis=1).format("{:.1f}")
    return styled_df

# ============================================================================
# TACTICAL ANALYSIS - MULTI-BOAT (YOUR EXACT CODE)
# ============================================================================

def create_multi_boat_tactical_plot(legs_df, leg_summary_df, leg_type='uw'):
    """
    Create tactical plot with ALL boats using your exact notebook code
    """
    summary = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    legs = legs_df[legs_df["leg_id"].isin(summary["leg_id"])].copy()

    if summary.empty or legs.empty:
        return None

    # Your exact code
    rotation_angle = np.deg2rad(summary["avg_TWD"].mean())
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)

    time_values = summary.set_index("leg_id")["time_seconds"].astype(float)
    q75 = time_values.quantile(0.75)
    q25 = time_values.quantile(0.25)
    den = (q75 - q25) if (q75 - q25) != 0 else 1.0

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
    fig, ax = plt.subplots(figsize=(12, 10))

    for legID in summary.sort_values("time_seconds")["leg_id"].unique():
        leg_data = legs[legs["leg_id"] == legID].copy()
        if len(leg_data) < 2:
            continue

        lon = leg_data["LONGITUDE_GPS_unk"].to_numpy(dtype=float)
        lat = leg_data["LATITUDE_GPS_unk"].to_numpy(dtype=float)

        lon0, lat0 = np.nanmean(lon), np.nanmean(lat)
        x = lon - lon0
        y = lat - lat0

        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle

        leg_time = float(time_values.loc[legID])
        linewidth = 1 + 9 * (q75 - leg_time) / den
        linewidth = float(np.clip(linewidth, 1, 10))

        twd_dev = (leg_data["TWD_MHU_SGP_deg"].to_numpy(dtype=float) - twd_mean)
        seg_colors = cmap(norm(twd_dev))

        for i in range(len(x_rot) - 1):
            ax.plot(
                x_rot[i:i+2], y_rot[i:i+2],
                color=tuple(seg_colors[i]),
                linewidth=linewidth,
                alpha=0.8
            )

    ax.set_xlabel("Rotated X (lon)", fontsize=12)
    ax.set_ylabel("Rotated Y (lat)", fontsize=12)
    ax.set_title(f"ALL BOATS - {'Upwind' if leg_type == 'uw' else 'Downwind'} Legs (Rotated {np.rad2deg(rotation_angle):.1f}°)", 
                 fontsize=14, fontweight='bold')
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=f"TWD deviation from {'UW' if leg_type == 'uw' else 'DW'} mean (deg)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def get_gradient_color_time(value, min_val, max_val):
    """Green (fast) to Red (slow) gradient"""
    if max_val == min_val:
        return "background-color: #ffffff"
    normalized = (value - min_val) / (max_val - min_val)
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
    """Gate summary with color gradient:
       - Avg Time (s) = mean of top-3 fastest legs per gate
       - If a gate has <3 legs, fallback to overall mean for that gate
    """
    filtered = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    if len(filtered) == 0:
        return None

    def top3_or_mean(series, k=5):
        s = series.dropna()
        if len(s) >= k:
            return s.nsmallest(k).median()
        return s.median()

    summary = (
        filtered.groupby("gate")["time_seconds"]
        .agg(**{
            "Avg Time (s)": lambda s: top3_or_mean(s, k=5),
            "Count": "count"
        })
    ).round(1)

    min_time = summary["Avg Time (s)"].min()
    max_time = summary["Avg Time (s)"].max()

    def apply_time_gradient(row):
        return [
            get_gradient_color_time(row["Avg Time (s)"], min_time, max_time),
            ""
        ]

    styled_df = summary.style.apply(apply_time_gradient, axis=1).format("{:.1f}")
    return styled_df


# ============================================================================
# STREAMLIT UI WITH FLYING ROOS BRANDING
# ============================================================================

def apply_flying_roos_css():
    """Apply Flying Roos corporate styling"""
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;600;700&display=swap');
        
        .main {{
            background: {FLYING_ROOS_COLORS['light_gray']};
            font-family: 'Roboto', sans-serif;
        }}
        
        h1, h2, h3 {{
            font-family: 'Roboto', sans-serif !important;
            color: {FLYING_ROOS_COLORS['primary_teal']} !important;
            font-weight: 700 !important;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: white;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Roboto', sans-serif;
            font-weight: 600;
            padding: 12px 24px;
            border-radius: 6px;
            border: none;
            color: {FLYING_ROOS_COLORS['dark_gray']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {FLYING_ROOS_COLORS['primary_teal']};
            color: white !important;
        }}
        
        .stButton button {{
            background: {FLYING_ROOS_COLORS['primary_teal']};
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 6px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,107,94,0.3);
        }}
        
        .stButton button:hover {{
            background: {FLYING_ROOS_COLORS['deep_green']};
        }}
        
        .dataframe {{
            font-family: 'Roboto', monospace !important;
            font-size: 0.9rem !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {FLYING_ROOS_COLORS['primary_teal']};
            font-weight: 600;
        }}
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Flying Roos - Race Analysis",
        page_icon="🦘",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_flying_roos_css()
    
    # LOGO HEADER (icon size, clickable)
    col1, col2 = st.columns([1, 8])
    with col1:
        try:
            st.image("FLYING ROOS.png", width=380)
        except:
            st.markdown("### 🦘 FLYING ROOS")
    with col2:
        st.markdown("# Race Analysis Dashboard")
        st.markdown("**BONDS Flying Roos** | Racing Analysis")
    
    # Make logo clickable
    st.markdown("""
        <a href="https://drive.google.com/drive/folders/11SslMi7EELFd-DkpFutCXrKCGpfi9Q_3" target="_blank">
            <medium>📁 Access Team Drive</small>
        </a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Data Selection")
        race_date = st.text_input("Race Date (YYYYMMDD)", value="20260118")
        csv_path = f"/Users/hugostubler/Documents/SailGP /Report_Pipeline/data/logs/log_{race_date}.csv"
        
        if Path(csv_path).exists():
            df = load_race_data(csv_path)
            st.success(f"✅ {len(df):,} rows loaded")
            df['RH_LEE'] = np.where(df['TWA_MHU_SGP_deg'] > 0, df.LENGTH_RH_P_mm, df.LENGTH_RH_S_mm)
            
            race_ids = get_unique_race_ids(df)
            if race_ids:
                selected_race_id = st.selectbox("Race ID", options=race_ids, format_func=lambda x: f"Race {int(x)}")
                all_boats = sorted(df['BOAT'].dropna().unique())
                default_boats = ['AUS', 'FRA', 'GBR']
                default = default_boats if all(b in all_boats for b in default_boats) else all_boats[:3]
                selected_boats = st.multiselect("Select Boats", options=all_boats, default=default)
                
                if selected_boats:
                    race_data = df[df['TRK_RACE_NUM_unk'] == selected_race_id]
                    race_data_ = race_data.copy()
                    race_data = race_data[race_data['BOAT'].isin(selected_boats)]
                    valid_legs = filter_valid_legs(df, selected_race_id, selected_boats)
                    selected_legs = st.multiselect("Select Legs", options=valid_legs, default=valid_legs)
                    st.markdown("---")
                    show_stability = st.checkbox("📊 Add Stability Metrics", value=False)
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Boats", len(selected_boats))
                    with col2:
                        st.metric("Legs", len(selected_legs))
                else:
                    selected_race_id = None
                    selected_legs = []
            else:
                selected_race_id = None
                selected_legs = []
        else:
            st.error(f"❌ File not found")
            selected_race_id = None
            selected_legs = []
    
    # Main content
    if selected_race_id and selected_legs and selected_boats:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚀 Start", "🏁 Legs", "🌊 Wind", "⚓ Tactical (ALL BOATS)", "📄 PDF"
        ])
        
        with tab1:
            st.markdown("## Start Analysis")
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
                st.metric("Avg TWS", f"{avg_tws:.1f} km/h")
            
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
                                pc_ratio_values[race_num][boat] = first_change_row["TTS_s"] / (first_change_row["TTS_s"] - first_change_row["PC_TTK_s"])
                                pc_dtl_values[race_num][boat] = first_change_row.get('PC_DTL_m', None)
                
                # Convert to DataFrames and apply BLUE GRADIENT
                df_pc_ttk = pd.DataFrame(pc_ttk_values).T
                df_pc_tts = pd.DataFrame(pc_tts_values).T
                df_pc_ratio = pd.DataFrame(pc_ratio_values).T
                df_pc_dtl = pd.DataFrame(pc_dtl_values).T
                
                # Apply blue gradient styling
                
                    
                def apply_subtle_gradient_func(row):
                    min_val = row.min()
                    max_val = row.max()
                    return [get_subtle_gradient(val, min_val, max_val) for val in row]
                    
                col1, col2 = st.columns(2)
                
                with col1:
                        st.markdown("**PC Time to Kill (TTK)**")
                        st.dataframe(df_pc_ttk.style.apply(apply_subtle_gradient_func, axis=1).format("{:.0f}"), use_container_width=True)
                        
                        st.markdown("**PC Time to Start (TTS)**")
                        st.dataframe(df_pc_tts.style.apply(apply_subtle_gradient_func, axis=1).format("{:.0f}"), use_container_width=True)
                
                with col2:
                        st.markdown("**PC Ratio (TTS/TTK)**")
                        st.dataframe(df_pc_ratio.style.apply(apply_subtle_gradient_func, axis=1).format("{:.1f}"), use_container_width=True)
                        
                        st.markdown("**PC Distance to Line (DTL)**")
                        if not df_pc_dtl.empty:
                            st.dataframe(df_pc_dtl.style.apply(apply_subtle_gradient_func, axis=1).format("{:.0f}"), use_container_width=True)

                # Add plot below the table
                st.markdown("### Start Track Plot")
                try:
                        from start_track_plot import plot_course_and_tracks
                        
                        # Get course_id from race data if available
                        course_id = selected_race_id  # Default, adjust if needed
                        
                        
                        fig, meta = plot_course_and_tracks(
                            course_id=course_id,
                            boats=race_data,
                            twd_col="TWD_MHU_SGP_deg",
                            tts_col="TTS_s",
                            boat_col="BOAT",
                            lat_col="LATITUDE_GPS_unk",
                            lon_col="LONGITUDE_GPS_unk",
                            board_col="LENGTH_DB_H_P_mm"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                        st.warning("start_track_plot.py not found. Skipping track plot.")
                except Exception as e:
                        st.error(f"Error creating track plot: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
            
        
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
                    summary_styled = create_summary_table_styled(leg_data, selected_boats, int(leg_num))
                    if summary_styled is not None:
                        st.markdown(f"**Leg {int(leg_num)} Summary**")
                        st.dataframe(summary_styled, use_container_width=True)
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
                        race_data_, 
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
                        race_data_, 
                        'TWD_MHU_SGP_deg', 
                        'TWD Distribution',
                        colorscale='RdYlGn'
                    )
                    st.plotly_chart(twd_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating TWD map: {str(e)}")
        
        # NEW TACTICAL TAB
        
        with tab4:
            st.markdown("## ⚓ Tactical Analysis - ALL BOATS")
            st.info("🔄 Loading tactical data...")
            
            try:
                # Load marks data
                marks_path = f"wind_data.csv"
                if Path(marks_path).exists():
                    marks_df = pd.read_csv(marks_path)
                    
                    # Prepare boat data
                    boat_data = race_data_[['BOAT', 'DATETIME', 'LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk',
                                          'TWD_MHU_SGP_deg', 'BOAT_SPEED_km_h_1', 'TWA_BOW_SGP_deg']].copy()
                    boat_data = boat_data.rename(columns={'BOAT': 'boat', 'DATETIME': '_time'})
                    
                    # Run gate crossing analysis
                    legs_df, leg_summary_df = analyze_gate_crossings(boat_data, marks_df)
                    
                    if not legs_df.empty:
                        uw_tab, dw_tab = st.tabs(["📈 Upwind (ALL BOATS)", "📉 Downwind (ALL BOATS)"])
                        
                        with uw_tab:
                            st.markdown("### All Boats - Upwind Legs")
                            uw_summary = leg_summary_df[leg_summary_df["leg_type"] == "uw"]
                            if len(uw_summary) > 0:
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.caption("Line width = speed | Color = TWD deviation | Shows ALL boats")
                                    plot = create_multi_boat_tactical_plot(legs_df, leg_summary_df, 'uw')
                                    if plot:
                                        st.image(plot, use_container_width=True)
                                with col2:
                                    st.markdown("**Gate Performance**")
                                    st.write("Median time (s) for each gate based on top-5 fastest legs for each turn. Green = faster, Red = slower.")
                                    st.caption("🟢 = faster | 🔴 = slower")
                                    gate_summary = create_gate_summary_table(leg_summary_df, 'uw')
                                    if gate_summary is not None:
                                        st.dataframe(gate_summary, use_container_width=True)
                                    st.metric("Total UW Legs", len(uw_summary))
                                    st.metric("Avg Time", f"{uw_summary['time_seconds'].mean():.1f}s")
                            else:
                                st.warning("No upwind legs detected")
                        
                        with dw_tab:
                            st.markdown("### All Boats - Downwind Legs")
                            dw_summary = leg_summary_df[leg_summary_df["leg_type"] == "dw"]
                            if len(dw_summary) > 0:
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.caption("Line width = speed | Color = TWD deviation | Shows ALL boats")
                                    plot = create_multi_boat_tactical_plot(legs_df, leg_summary_df, 'dw')
                                    if plot:
                                        st.image(plot, use_container_width=True)
                                with col2:
                                    st.markdown("**Gate Performance**")
                                    st.caption("🟢 = faster | 🔴 = slower")
                                    gate_summary = create_gate_summary_table(leg_summary_df, 'dw')
                                    if gate_summary is not None:
                                        st.dataframe(gate_summary, use_container_width=True)
                                    st.metric("Total DW Legs", len(dw_summary))
                                    st.metric("Avg Time", f"{dw_summary['time_seconds'].mean():.1f}s")
                            else:
                                st.warning("No downwind legs detected")
                    else:
                        st.warning("No legs detected - check data quality")
                else:
                    st.error(f"Marks file not found: {marks_path}")
            except Exception as e:
                st.error(f"Tactical analysis error: {str(e)}")
                st.info("Ensure marks.csv is available in the data directory")
        
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
        st.info("👈 Please select race data from the sidebar")

if __name__ == "__main__":
    main()
