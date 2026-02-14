"""
BONDS FLYING ROOS - SailGP Race Analysis Dashboard v4
Performance-optimized with fragment-based rendering, improved branding, and robust PDF export.
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
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from io import BytesIO
import base64

from gate_crossing_analyzer import analyze_gate_crossings
from manoeuvres import boat_summary

# ============================================================================
# FLYING ROOS COLOR PALETTE — matched to logo
# ============================================================================

FLYING_ROOS_COLORS = {
    "primary_green": "#004D40",       # Deep forest green (logo primary)
    "secondary_green": "#006B5E",     # Lighter teal-green
    "accent_gold": "#D4A843",         # Gold accent (logo kangaroo)
    "gold_light": "#E8C96A",          # Lighter gold for hover
    "bg_dark": "#00332C",             # Very dark green for header
    "bg_light": "#F7F9F8",            # Off-white with green tint
    "bg_card": "#FFFFFF",             # Card backgrounds
    "text_primary": "#1A2E2A",        # Dark text
    "text_secondary": "#5A7A72",      # Muted green text
    "border": "#D8E4E0",             # Subtle green-tinted border
    "white": "#FFFFFF",
}

# PRESERVE boat color mapping (DO NOT TOUCH)
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
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def mean_bearing(bearing1, bearing2):
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
    if max_val == min_val:
        return "background-color: #f8f9fa"
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))
    if reverse:
        normalized = 1 - normalized
    if normalized < 0.5:
        t = normalized * 2
        r = int(82 + t * (250 - 82))
        g = int(196 + t * (219 - 196))
        b = int(26 + t * (20 - 26))
    else:
        t = (normalized - 0.5) * 2
        r = int(250 + t * (255 - 250))
        g = int(219 + t * (77 - 219))
        b = int(20 + t * (79 - 20))
    return f"background-color: rgb({r}, {g}, {b})"

def get_subtle_gradient(value, min_val, max_val):
    primary_teal = (0, 77, 64)
    light_green = (230, 245, 240)
    if max_val == min_val:
        return f"background-color: rgb{light_green}; color: #000000"
    normalized = (value - min_val) / (max_val - min_val)
    r = int(light_green[0] + normalized * (primary_teal[0] - light_green[0]))
    g = int(light_green[1] + normalized * (primary_teal[1] - light_green[1]))
    b = int(light_green[2] + normalized * (primary_teal[2] - light_green[2]))
    return f"background-color: rgb({r}, {g}, {b}); color: #000000"

# ============================================================================
# DATA PROCESSING (all cached)
# ============================================================================

@st.cache_data
def load_race_data(file_path):
    """Load race data from Parquet (preferred) or CSV."""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def discover_race_dates():
    """Scan logs/ folder for available race dates, preferring parquet files."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    dates = set()
    # Prefer parquet, also find CSV-only dates
    for f in logs_dir.iterdir():
        if f.name.startswith("log_") and f.suffix in ('.parquet', '.csv'):
            # Extract YYYYMMDD from log_YYYYMMDD.ext
            date_part = f.stem.replace("log_", "")
            if len(date_part) == 8 and date_part.isdigit():
                dates.add(date_part)
    return sorted(dates, reverse=True)


def resolve_data_path(race_date):
    """Return the best available file path for a given date (parquet > csv)."""
    parquet_path = f"logs/log_{race_date}.parquet"
    csv_path = f"logs/log_{race_date}.csv"
    if Path(parquet_path).exists():
        return parquet_path
    if Path(csv_path).exists():
        return csv_path
    return None

def get_unique_race_ids(df):
    race_ids = df['TRK_RACE_NUM_unk'].dropna().unique()
    race_ids = [rid for rid in race_ids if rid > 0]
    return sorted(race_ids)

@st.cache_data
def filter_valid_legs(df_bytes, race_id, boats_tuple):
    df = pd.read_json(io.StringIO(df_bytes))
    boats = list(boats_tuple)
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

def compute_leg_summary(leg_data, boat):
    """Calculate summary statistics for a boat-leg"""
    boat_data = leg_data[leg_data['BOAT'] == boat].copy()
    man = boat_summary(boat_data, "MHU")
    boat_data['DATETIME'] = pd.to_datetime(boat_data['DATETIME'])
    if len(man) > 1:
        mdmg = man.loc[man["distance_vmg_cog"] > 0, "distance_vmg_cog"].mean()
        man_dist_made_good = np.round(mdmg, 0) if pd.notna(mdmg) else np.nan
        num_of_man = max(len(man) - 1, 0)
        for tman in man["DATETIME"].iloc[1:].tolist():
            t = pd.Timestamp(tman.datetime)
            boat_data_without_man = boat_data.loc[
                ~((boat_data["DATETIME"] >= t - pd.Timedelta(seconds=3)) &
                  (boat_data["DATETIME"] <= t + pd.Timedelta(seconds=8)))
            ]
    else:
        boat_data_without_man = boat_data.copy()
        man_dist_made_good = 0
        num_of_man = 0
    mean_TWD_leg = leg_data["TWD_MHU_SGP_deg"].mean()
    if len(boat_data) == 0:
        return None
    time_seconds = len(boat_data)
    coords = boat_data[['LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk']].values
    total_distance = 0
    for i in range(1, len(coords)):
        total_distance += geodesic(coords[i-1], coords[i]).meters
    avg_BSP = boat_data_without_man['BOAT_SPEED_km_h_1'].mean()
    avg_VMG = abs(boat_data_without_man['VMG_km_h_1'].mean())
    avg_TWD = boat_data_without_man['TWD_MHU_SGP_deg'].mean()
    avg_TWS = boat_data_without_man['TWS_MHU_SGP_km_h_1'].mean()
    heel_stability = boat_data_without_man['HEEL_deg'].rolling(10).std().mean() if 'HEEL_deg' in boat_data.columns else 0
    rh_stability = boat_data_without_man['RH_LEE'].rolling(10).std().mean() if 'RH_LEE' in boat_data.columns else 0
    delta_twd = mean_TWD_leg - boat_data_without_man["TWD_MHU_SGP_deg"]
    stbd = boat_data_without_man["TWA_MHU_SGP_deg"] > 0
    port = boat_data_without_man["TWA_MHU_SGP_deg"] < 0
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
        'man_dist_made_good': man_dist_made_good,
        'num_of_man': num_of_man,
        'heel_stability': heel_stability,
        'rh_stability': rh_stability
    }

def calculate_leg_summary(leg_data, boat):
    """Wrapper that calls cached compute_leg_summary"""
    return compute_leg_summary(leg_data, boat)


def create_summary_table_styled(leg_data, boats, leg_num):
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary:
            summaries[boat] = summary
    if not summaries:
        return None
    df = pd.DataFrame(summaries)
    for idx in df.index:
        for col in df.columns:
            df.at[idx, col] = round(df.at[idx, col], 1)
    df = df.iloc[:9]
    df.index = ['Time (s)', 'Distance (m)', 'Avg BSP', 'Avg VMG', 'Avg TWD', 'Avg TWS', '% In Phase', 'Dist Made Good', 'Num of man']
    higher_is_better = {'Avg BSP', 'Avg VMG', '% In Phase', 'Dist Made Good', 'Avg TWD', 'Avg TWS'}
    lower_is_better = {'Time (s)', 'Distance (m)', 'Num of man'}

    def apply_gradient(row):
        styles = []
        row_name = row.name
        for col in df.columns:
            value = row[col]
            min_val = row.min()
            max_val = row.max()
            if row_name in higher_is_better:
                styles.append(get_gradient_color(value, min_val, max_val, reverse=True))
            elif row_name in lower_is_better:
                styles.append(get_gradient_color(value, min_val, max_val, reverse=False))
            else:
                styles.append("")
        return styles

    styled_df = df.style.apply(apply_gradient, axis=1).format("{:.1f}")
    return styled_df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def _generate_grid_lines(center_lat, center_lon, twd_deg, grid_spacing_m=100, n_lines=10):
    """Generate lat/lon grid lines aligned to the wind axis (TWD).
    Lines run parallel and perpendicular to TWD, spaced grid_spacing_m apart."""
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(center_lat)))
    # TWD is direction wind comes FROM (compass bearing, CW from N)
    # Along-wind axis direction in math coords: 90 - TWD (CCW from E)
    along_rad = math.radians(twd_deg)
    # Unit vectors in lat/lon degrees-per-meter
    # Along wind: (dlon, dlat) pointing in TWD compass direction
    along_dlon = math.sin(along_rad) * deg_per_m_lon
    along_dlat = math.cos(along_rad) * deg_per_m_lat
    # Cross wind (perpendicular, 90 deg CW)
    cross_dlon = math.cos(along_rad) * deg_per_m_lon
    cross_dlat = -math.sin(along_rad) * deg_per_m_lat

    half_len = n_lines * grid_spacing_m  # half-length of each line in meters
    lines = []
    # Lines perpendicular to wind (cross-wind lines, spaced along wind axis)
    for i in range(-n_lines, n_lines + 1):
        offset_m = i * grid_spacing_m
        c_lat = center_lat + offset_m * along_dlat
        c_lon = center_lon + offset_m * along_dlon
        lat0 = c_lat - half_len * cross_dlat
        lon0 = c_lon - half_len * cross_dlon
        lat1 = c_lat + half_len * cross_dlat
        lon1 = c_lon + half_len * cross_dlon
        lines.append({"lats": [lat0, lat1], "lons": [lon0, lon1]})
    # Lines parallel to wind (along-wind lines, spaced cross-wind)
    for i in range(-n_lines, n_lines + 1):
        offset_m = i * grid_spacing_m
        c_lat = center_lat + offset_m * cross_dlat
        c_lon = center_lon + offset_m * cross_dlon
        lat0 = c_lat - half_len * along_dlat
        lon0 = c_lon - half_len * along_dlon
        lat1 = c_lat + half_len * along_dlat
        lon1 = c_lon + half_len * along_dlon
        lines.append({"lats": [lat0, lat1], "lons": [lon0, lon1]})
    return lines


def plot_leg_tracks(leg_data, boats, leg_num):
    fig = go.Figure()
    twd_mean = leg_data['TWD_MHU_SGP_deg'].mean()

    all_lats = leg_data['LATITUDE_GPS_unk'].dropna()
    all_lons = leg_data['LONGITUDE_GPS_unk'].dropna()
    center_lat = all_lats.mean()
    center_lon = all_lons.mean()

    # Add 100m grid overlay aligned to wind axis
    grid_lines = _generate_grid_lines(center_lat, center_lon, twd_mean, grid_spacing_m=100, n_lines=8)
    for i, line in enumerate(grid_lines):
        fig.add_trace(go.Scattermapbox(
            lat=line["lats"], lon=line["lons"],
            mode='lines',
            line=dict(width=0.5, color='rgba(0,77,64,0.15)'),
            hoverinfo='skip',
            showlegend=False,
        ))

    for boat in boats:
        boat_data = leg_data[leg_data['BOAT'] == boat]
        if len(boat_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=boat_data['LATITUDE_GPS_unk'],
                lon=boat_data['LONGITUDE_GPS_unk'],
                mode='lines',
                name=boat,
                line=dict(width=3, color=COLOR_MAPPING.get(boat, '#888888')),
                hovertemplate=f"<b>{boat}</b><br>Speed: %{{customdata[0]:.1f}} km/h<extra></extra>",
                customdata=boat_data[['BOAT_SPEED_km_h_1']].values
            ))
    if not any(t.name in COLOR_MAPPING or (t.name and t.name in [b for b in boats]) for t in fig.data):
        return None

    fig.update_layout(
        mapbox=dict(
            bearing=twd_mean,
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=14.6
        ),
        showlegend=True,
        height=600,
        title=dict(
            text=f"Leg {leg_num} Tracks (grid = 100m)",
            font=dict(color=FLYING_ROOS_COLORS['primary_green'], size=16, family="Inter, sans-serif")
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_wind_analysis(race_data, boat='AUS'):
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
        line=dict(color=FLYING_ROOS_COLORS['primary_green'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=boat_data.index,
        y=boat_data['TWD_MHU_SGP_deg'].rolling(5).mean(),
        mode='lines', name='TWD',
        line=dict(color=FLYING_ROOS_COLORS['accent_gold'], width=2),
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
        title=dict(
            text=f"Wind Analysis - {boat}",
            font=dict(color=FLYING_ROOS_COLORS['primary_green'], size=16)
        ),
        xaxis=dict(title="Time"),
        yaxis=dict(title="TWS (kt)", side="left", range=[tws_min - 1, tws_max + 1]),
        yaxis2=dict(title="TWD", overlaying="y", side="right", range=[twd_min - 5, twd_max + 5]),
        height=500,
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def fast_idw(points, values, xi, power=1.5):
    tree = cKDTree(points)
    distances, indices = tree.query(xi, k=min(15, len(points)))
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    distances = np.where(distances < 1e-10, 1e-10, distances)
    weights = 1.0 / (distances ** power)
    return np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)


def create_wind_map(race_data, value_col, title, colorscale="RdYlGn"):
    valid_data = race_data[['LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk', value_col]].dropna()
    if len(valid_data) < 3:
        return None
    if len(valid_data) > 800:
        valid_data = valid_data.sample(n=800, random_state=42)
    lats = valid_data['LATITUDE_GPS_unk'].values
    lons = valid_data['LONGITUDE_GPS_unk'].values
    vals = valid_data[value_col].values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    padding = 0.05
    lat_min -= padding * lat_range
    lat_max += padding * lat_range
    lon_min -= padding * lon_range
    lon_max += padding * lon_range
    grid_resolution = 60
    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack([lons, lats])
    grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    try:
        rbf = RBFInterpolator(points, vals, kernel='thin_plate_spline', smoothing=0.01, degree=1)
        z_grid = rbf(grid_points).reshape(lat_mesh.shape)
    except Exception:
        z_grid = fast_idw(points, vals, grid_points, power=1.5).reshape(lat_mesh.shape)

    # --- Contour overlay image ---
    fig_mpl, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    cmap_name = colorscale.lower() if colorscale.lower() in plt.colormaps() else 'RdYlGn'
    cf = ax.contourf(lon_mesh, lat_mesh, z_grid, levels=15, cmap=cmap_name, alpha=0.7)
    contour = ax.contour(lon_mesh, lat_mesh, z_grid, levels=15, colors='white', linewidths=0.5, alpha=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    ax.axis('off')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    plt.tight_layout(pad=0)
    buf = BytesIO()
    plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig_mpl)

    # --- Standalone colorbar legend image (large & readable) ---
    fig_cb, ax_cb = plt.subplots(figsize=(1.2, 5))
    cmap_obj = plt.get_cmap(cmap_name)
    norm_obj = plt.Normalize(vmin=z_grid.min(), vmax=z_grid.max())
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cb)
    # Readable label from column name
    label_map = {
        'TWS_MHU_SGP_km_h_1': 'TWS (km/h)',
        'TWD_MHU_SGP_deg': 'TWD (deg)',
    }
    unit_label = label_map.get(value_col, value_col)
    cbar.set_label(unit_label, fontsize=14, fontweight='bold', color='#004D40')
    cbar.ax.tick_params(labelsize=12, colors='#004D40', width=1.5, length=4)
    plt.tight_layout(pad=0.4)
    buf_cb = BytesIO()
    plt.savefig(buf_cb, format='png', dpi=150, bbox_inches='tight', pad_inches=0.15, transparent=True)
    buf_cb.seek(0)
    legend_base64 = base64.b64encode(buf_cb.read()).decode()
    plt.close(fig_cb)

    fig = go.Figure()
    mean_twd = race_data['TWD_MHU_SGP_deg'].mean() if 'TWD_MHU_SGP_deg' in race_data.columns else 0
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            bearing=mean_twd,
            center=dict(lat=lats.mean(), lon=lons.mean()),
            zoom=13.7,
            layers=[dict(
                sourcetype='image',
                source=f"data:image/png;base64,{img_base64}",
                coordinates=[
                    [lon_min, lat_max], [lon_max, lat_max],
                    [lon_max, lat_min], [lon_min, lat_min]
                ],
                opacity=0.8
            )]
        ),
        title=dict(text=title, font=dict(size=16, color=FLYING_ROOS_COLORS['primary_green']), x=0.5, xanchor='center'),
        height=600,
        margin=dict(l=0, r=80, t=50, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.add_trace(go.Scattermapbox(
        lat=[lats.mean()], lon=[lons.mean()],
        mode='markers', marker=dict(size=0.1, opacity=0),
        showlegend=False
    ))

    # Add colorbar legend as layout image annotation (large & readable)
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{legend_base64}",
            xref="paper", yref="paper",
            x=1.02, y=0.5,
            sizex=0.15, sizey=0.85,
            xanchor="left", yanchor="middle",
            layer="above"
        )
    )
    # Widen right margin to make room for legend
    fig.update_layout(margin=dict(l=0, r=120, t=50, b=0))
    return fig


def create_multi_boat_tactical_plot(legs_df, leg_summary_df, leg_type='uw'):
    summary = leg_summary_df[leg_summary_df["leg_type"] == leg_type].copy()
    legs = legs_df[legs_df["leg_id"].isin(summary["leg_id"])].copy()
    if summary.empty or legs.empty:
        return None
    rotation_angle = np.deg2rad(summary["avg_TWD"].mean())
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    time_values = summary.set_index("leg_id")["time_seconds"].astype(float)
    q75 = time_values.quantile(0.75)
    q25 = time_values.quantile(0.25)
    den = (q75 - q25) if (q75 - q25) != 0 else 1.0
    bsp_all = legs["BOAT_SPEED_km_h_1"].astype(float).to_numpy()
    if np.nanmax(bsp_all) == np.nanmin(bsp_all):
        vmax = max(1.0, abs(np.nanmax(bsp_all)))
        vmin = max(1.0, abs(np.nanmin(bsp_all)))
        norm = TwoSlopeNorm(vmin=-vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(
            vmin=np.nanquantile(bsp_all, 0.2),
            vcenter=np.nanmedian(bsp_all),
            vmax=np.nanquantile(bsp_all, 0.8)
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
        bsp = leg_data["BOAT_SPEED_km_h_1"].to_numpy(dtype=float)
        seg_colors = cmap(norm(bsp))
        for i in range(len(x_rot) - 1):
            ax.plot(
                x_rot[i:i+2], y_rot[i:i+2],
                color=tuple(seg_colors[i]),
                linewidth=linewidth, alpha=0.8
            )
    ax.set_xlabel("Rotated X (lon)", fontsize=12)
    ax.set_ylabel("Rotated Y (lat)", fontsize=12)
    ax.set_title(
        f"ALL BOATS - {'Upwind' if leg_type == 'uw' else 'Downwind'} Legs (Rotated {np.rad2deg(rotation_angle):.1f})",
        fontsize=14, fontweight='bold', color=FLYING_ROOS_COLORS['primary_green']
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Boat Speed (km/h)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def get_gradient_color_time(value, min_val, max_val):
    if max_val == min_val:
        return "background-color: #ffffff"
    normalized = (value - min_val) / (max_val - min_val)
    if normalized > 0.5:
        r, g, b = 255, 0, 0
    else:
        r, g, b = 0, 255, 0
    return f"background-color: rgb({r}, {g}, {b})"


def create_gate_summary_table(leg_summary_df, leg_type='uw'):
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
        .agg(**{"Avg Time (s)": lambda s: top3_or_mean(s, k=5), "Count": "count"})
    ).round(1)
    min_time = summary["Avg Time (s)"].min()
    max_time = summary["Avg Time (s)"].max()

    def apply_time_gradient(row):
        return [get_gradient_color_time(row["Avg Time (s)"], min_time, max_time), ""]

    styled_df = summary.style.apply(apply_time_gradient, axis=1).format("{:.1f}")
    return styled_df


def create_stability_table_styled(leg_data, boats):
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

    def apply_stability_gradient(row):
        styles = []
        min_val = row.min()
        max_val = row.max()
        for value in row:
            styles.append(get_gradient_color(value, min_val, max_val, reverse=False))
        return styles

    styled_df = df.style.apply(apply_stability_gradient, axis=1).format("{:.1f}")
    return styled_df


# ============================================================================
# PDF GENERATION — matplotlib-based (no kaleido/Chrome dependency)
# ============================================================================

def generate_pdf_plot_as_image(fig_plotly, width=700, height=400):
    """Convert a plotly figure to PNG bytes using matplotlib fallback if kaleido fails."""
    try:
        img_bytes = fig_plotly.to_image(format="png", width=width, height=height)
        return BytesIO(img_bytes)
    except Exception:
        # Fallback: save as static HTML note
        return None


def generate_race_report_pdf(df, race_id, boats, legs, output_path=None):
    """Generate PDF report using reportlab with matplotlib-rendered plots (no Chrome needed)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    if output_path:
        doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    else:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=24,
        textColor=rl_colors.HexColor('#004D40'), spaceAfter=20,
        alignment=TA_CENTER, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle', parent=styles['Heading2'], fontSize=12,
        textColor=rl_colors.HexColor('#1A2E2A'), spaceAfter=8, spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    leg_title_style = ParagraphStyle(
        'LegTitle', parent=styles['Heading2'], fontSize=18,
        textColor=rl_colors.HexColor('#004D40'), spaceAfter=6, spaceBefore=6,
        fontName='Helvetica-Bold'
    )

    story = []
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("BONDS Flying Roos - Race Report", title_style))
    story.append(Paragraph(f"Race ID: {int(race_id)}", subtitle_style))
    story.append(Paragraph(f"Boats: {', '.join(boats)}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    race_data = df[df['TRK_RACE_NUM_unk'] == race_id]
    race_data = race_data[race_data['BOAT'].isin(boats)]

    # Wind time series using matplotlib directly (no kaleido)
    story.append(Paragraph("Wind Analysis", subtitle_style))
    wind_img = _create_wind_plot_matplotlib(race_data, boats[0])
    if wind_img:
        img = Image(wind_img, width=6.5*inch, height=2.3*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))

    # Each leg
    for leg_num in legs:
        story.append(PageBreak())
        story.append(Paragraph(f"LEG {int(leg_num)}", leg_title_style))
        story.append(Spacer(1, 0.05*inch))

        leg_data = race_data[race_data['TRK_LEG_NUM_unk'] == leg_num]
        leg_elements = []

        # Track plot via matplotlib
        track_img = _create_track_plot_matplotlib(leg_data, boats, int(leg_num))
        if track_img:
            img = Image(track_img, width=6*inch, height=3.7*inch)
            leg_elements.append(img)
            leg_elements.append(Spacer(1, 0.15*inch))

        # Summary table
        table_data, color_data = _create_pdf_summary_table(leg_data, boats, int(leg_num))
        if table_data:
            num_boats = len(boats)
            metric_col_width = 1.1*inch
            boat_col_width = (6*inch - metric_col_width) / num_boats
            col_widths = [metric_col_width] + [boat_col_width] * num_boats
            table = Table(table_data, colWidths=col_widths)
            table_style_list = [
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#004D40')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ('BOX', (0, 0), (-1, -1), 1.5, rl_colors.HexColor('#004D40')),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('BACKGROUND', (0, 1), (0, -1), rl_colors.HexColor('#E6F5F0')),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]
            if color_data:
                for row_idx, colors_row in enumerate(color_data):
                    for col_idx, color in enumerate(colors_row):
                        table_style_list.append(
                            ('BACKGROUND', (col_idx + 1, row_idx + 1), (col_idx + 1, row_idx + 1), color)
                        )
            table.setStyle(TableStyle(table_style_list))
            leg_elements.append(Paragraph(f"<b>Leg {int(leg_num)} Summary</b>", subtitle_style))
            leg_elements.append(table)

        story.append(KeepTogether(leg_elements))

    doc.build(story)
    if output_path:
        return None
    else:
        buffer.seek(0)
        return buffer


def _create_wind_plot_matplotlib(race_data, boat='AUS'):
    """Create wind plot using pure matplotlib (no kaleido)."""
    boat_data = race_data[race_data['BOAT'] == boat].copy()
    boat_data = boat_data[boat_data['TRK_LEG_NUM_unk'] > 0].sort_index()
    if len(boat_data) == 0:
        return None

    fig_mpl, ax1 = plt.subplots(figsize=(10, 3.5))
    ax2 = ax1.twinx()

    tws = boat_data['TWS_MHU_SGP_km_h_1'].rolling(5).mean()
    twd = boat_data['TWD_MHU_SGP_deg'].rolling(5).mean()

    ax1.plot(boat_data.index, tws, color='#004D40', linewidth=1.5, label='TWS (kt)')
    ax2.plot(boat_data.index, twd, color='#D4A843', linewidth=1.5, label='TWD')

    ax1.set_ylabel('TWS (kt)', color='#004D40')
    ax2.set_ylabel('TWD', color='#D4A843')
    ax1.set_xlabel('Time')
    ax1.set_title(f'Wind Analysis - {boat}', fontsize=12, fontweight='bold', color='#004D40')
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    return buf


def _create_track_plot_matplotlib(leg_data, boats, leg_num):
    """Create rotated track plot using matplotlib (no kaleido).
    Rotated so wind-from (TWD) axis points UP on the chart."""
    mean_twd = leg_data['TWD_MHU_SGP_deg'].mean()

    fig_mpl, ax = plt.subplots(figsize=(8, 5))
    # Rotation by +TWD aligns the wind-from direction with +Y (up)
    # This matches the tactical plot and start_track_plot conventions
    angle_rad = math.radians(mean_twd)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    for boat in boats:
        boat_data = leg_data[leg_data['BOAT'] == boat]
        if len(boat_data) == 0:
            continue
        lat = boat_data['LATITUDE_GPS_unk'].values
        lon = boat_data['LONGITUDE_GPS_unk'].values
        lat_mean = lat.mean()
        # Convert to local meters (x=East, y=North)
        x = (lon - lon[0]) * 111320 * math.cos(math.radians(lat_mean))
        y = (lat - lat[0]) * 111320
        # Rotate by +TWD so wind axis points up
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        color = COLOR_MAPPING.get(boat, '#888888')
        ax.plot(x_rot, y_rot, linewidth=2, color=color, label=boat)

    # Add 100m grid
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    grid_step = 100
    x_start = int(xlim[0] // grid_step) * grid_step
    x_end = int(xlim[1] // grid_step + 1) * grid_step
    y_start = int(ylim[0] // grid_step) * grid_step
    y_end = int(ylim[1] // grid_step + 1) * grid_step
    for gx in range(x_start, x_end + 1, grid_step):
        ax.axvline(gx, color='#004D40', alpha=0.1, linewidth=0.5)
    for gy in range(y_start, y_end + 1, grid_step):
        ax.axhline(gy, color='#004D40', alpha=0.1, linewidth=0.5)

    ax.set_aspect('equal')
    ax.set_xlabel('Cross-wind (m)', fontsize=10)
    ax.set_ylabel('Upwind (m)', fontsize=10)
    ax.set_title(f'Leg {leg_num} - Tracks (Wind {mean_twd:.0f}\u00b0 \u2191, grid=100m)',
                 fontsize=12, fontweight='bold', color='#004D40')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_mpl)
    buf.seek(0)
    return buf


def _interpolate_color_pdf(value, min_val, max_val, reverse=False):
    """Color gradient for PDF tables."""
    from reportlab.lib import colors as rl_colors
    if max_val == min_val:
        return rl_colors.HexColor('#ffffff')
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
    return rl_colors.Color(r/255, g/255, b/255)


def _create_pdf_summary_table(leg_data, boats, leg_num):
    """Create summary table data for PDF with color gradients."""
    from reportlab.lib import colors as rl_colors
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary:
            summaries[boat] = summary
    if not summaries:
        return None, None

    metric_labels = ['Time (s)', 'Distance (m)', 'Avg BSP', 'Avg VMG', 'Avg TWD', 'Avg TWS', '% In Phase']
    metric_keys = ['time_seconds', 'total_distance_m', 'avg_BSP', 'avg_VMG', 'avg_TWD', 'avg_TWS', 'pct_time_in_phase']
    higher_is_better = {'avg_BSP', 'avg_VMG', 'pct_time_in_phase'}
    lower_is_better = {'time_seconds'}

    headers = ['Metric'] + list(summaries.keys())
    rows = [headers]
    color_data = []

    for label, key in zip(metric_labels, metric_keys):
        row = [label]
        values = []
        for boat in summaries.keys():
            val = summaries[boat][key]
            row.append(f"{val:.1f}")
            values.append(val)
        rows.append(row)

        if len(values) > 0:
            min_val = min(values)
            max_val = max(values)
            if key in lower_is_better:
                colors_row = [_interpolate_color_pdf(v, min_val, max_val, reverse=False) for v in values]
            elif key in higher_is_better:
                colors_row = [_interpolate_color_pdf(v, min_val, max_val, reverse=True) for v in values]
            else:
                colors_row = [rl_colors.white] * len(values)
            color_data.append(colors_row)
        else:
            color_data.append([rl_colors.white] * len(summaries))

    return rows, color_data


# ============================================================================
# STREAMLIT UI — v4 with Flying Roos branding
# ============================================================================

def apply_flying_roos_css():
    C = FLYING_ROOS_COLORS
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ---- Global ---- */
        .main {{
            background: {C['bg_light']};
            font-family: 'Inter', sans-serif;
        }}
        .block-container {{
            padding-top: 1rem;
        }}

        /* ---- Headings ---- */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif !important;
            color: {C['primary_green']} !important;
            font-weight: 700 !important;
        }}

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {C['bg_dark']} 0%, {C['primary_green']} 100%);
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {{
            color: {C['white']} !important;
        }}
        /* Keep input field text dark so it's readable on white input bg */
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="tag"] span,
        [data-testid="stSidebar"] [data-baseweb="input"] input {{
            color: {C['text_primary']} !important;
        }}
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stCheckbox label {{
            color: {C['accent_gold']} !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        [data-testid="stSidebar"] .stMetric {{
            background: rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 8px 12px;
        }}
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
            color: {C['accent_gold']} !important;
            font-weight: 700 !important;
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {C['bg_card']};
            border-radius: 12px;
            padding: 6px;
            box-shadow: 0 2px 12px rgba(0,77,64,0.08);
            border: 1px solid {C['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            color: {C['text_secondary']};
            transition: all 0.2s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: rgba(0,77,64,0.06);
            color: {C['primary_green']};
        }}
        .stTabs [aria-selected="true"] {{
            background: {C['primary_green']} !important;
            color: {C['white']} !important;
            box-shadow: 0 2px 8px rgba(0,77,64,0.3);
        }}

        /* ---- Buttons ---- */
        .stButton button {{
            background: linear-gradient(135deg, {C['primary_green']} 0%, {C['secondary_green']} 100%);
            color: {C['white']};
            border: none;
            padding: 10px 28px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            box-shadow: 0 4px 12px rgba(0,77,64,0.25);
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stButton button:hover {{
            background: linear-gradient(135deg, {C['secondary_green']} 0%, {C['primary_green']} 100%);
            box-shadow: 0 6px 20px rgba(0,77,64,0.35);
            transform: translateY(-1px);
        }}

        /* ---- Download button ---- */
        .stDownloadButton button {{
            background: linear-gradient(135deg, {C['accent_gold']} 0%, {C['gold_light']} 100%) !important;
            color: {C['bg_dark']} !important;
            font-weight: 700 !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(212,168,67,0.3);
        }}

        /* ---- Metrics ---- */
        [data-testid="stMetricValue"] {{
            color: {C['primary_green']};
            font-weight: 700;
            font-size: 1.3rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {C['text_secondary']};
            font-weight: 500;
            text-transform: uppercase;
            font-size: 0.7rem !important;
            letter-spacing: 0.5px;
        }}

        /* ---- DataFrames ---- */
        .dataframe {{
            font-family: 'Inter', monospace !important;
            font-size: 0.85rem !important;
        }}

        /* ---- Cards / containers ---- */
        [data-testid="stExpander"] {{
            border: 1px solid {C['border']};
            border-radius: 12px;
            background: {C['bg_card']};
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }}

        /* ---- Dividers ---- */
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(90deg, {C['primary_green']}, {C['accent_gold']}, transparent);
            margin: 1rem 0;
        }}

        /* ---- Header bar ---- */
        .flying-roos-header {{
            background: linear-gradient(135deg, {C['bg_dark']} 0%, {C['primary_green']} 60%, {C['secondary_green']} 100%);
            padding: 20px 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 24px;
            box-shadow: 0 4px 20px rgba(0,51,44,0.3);
        }}
        .flying-roos-header img {{
            height: 60px;
        }}
        .flying-roos-header h1 {{
            color: {C['white']} !important;
            margin: 0 !important;
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }}
        .flying-roos-header .subtitle {{
            color: {C['accent_gold']};
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 2px;
        }}
        .flying-roos-header a {{
            color: {C['accent_gold']} !important;
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 500;
            transition: color 0.2s;
        }}
        .flying-roos-header a:hover {{
            color: {C['gold_light']} !important;
        }}

        /* ---- Section headers ---- */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 0;
            border-bottom: 2px solid {C['accent_gold']};
            margin-bottom: 16px;
        }}
        .section-header h2 {{
            margin: 0 !important;
            font-size: 1.2rem !important;
        }}

        /* ---- Success / info boxes ---- */
        .stSuccess, .stInfo {{
            border-radius: 8px;
        }}

        /* ---- Selectbox dropdowns ---- */
        .stSelectbox > div > div {{
            border-radius: 8px;
        }}
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the branded header bar."""
    st.markdown("""
        <div class="flying-roos-header">
            <div>
                <h1>Race Analyser</h1>
                <div class="subtitle">BONDS Flying Roos | SailGP Performance Analytics</div>
                <a href="https://drive.google.com/drive/folders/11SslMi7EELFd-DkpFutCXrKCGpfi9Q_3" target="_blank">
                    Team Drive &rarr;
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FRAGMENT-BASED TAB RENDERERS (only re-run when their inputs change)
# ============================================================================

@st.fragment
def render_start_tab(race_data, selected_boats, selected_race_id):
    """Start analysis tab — runs independently as a fragment."""
    st.markdown("## Start Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Race ID", f"{int(selected_race_id)}")
    with col2:
        legs_in_race = race_data['TRK_LEG_NUM_unk'].dropna().unique()
        st.metric("Total Legs", len([l for l in legs_in_race if l > 0]))
    with col3:
        st.metric("Boats Racing", len(selected_boats))
    with col4:
        avg_tws = race_data['TWS_MHU_SGP_km_h_1'].mean()
        st.metric("Avg TWS", f"{avg_tws:.1f} km/h")

    st.markdown("### Start Metrics")
    try:
        df_start = race_data[(race_data['TTS_s'] < 120) & (race_data['TTS_s'] > -10)].copy()
        pc_metrics = {}
        for boat in selected_boats:
            df_subset = df_start[df_start['BOAT'] == boat]
            df_subset_sorted = df_subset.sort_values(by="TTS_s")
            if 'LENGTH_DB_H_P_mm' in df_subset_sorted.columns:
                board_values = df_subset_sorted["LENGTH_DB_H_P_mm"]
                if len(board_values[board_values.diff().abs() > 1]) > 0:
                    first_change_idx = board_values[board_values.diff().abs() > 1].idxmin()
                    first_change_row = df_start.loc[first_change_idx]
                    if 'PC_TTK_s' in first_change_row and 'TTS_s' in first_change_row:
                        ttk = first_change_row["PC_TTK_s"]
                        tts = first_change_row["TTS_s"]
                        ratio = tts / (tts - ttk) if (tts - ttk) != 0 else None
                        dtl = first_change_row.get('PC_DTL_m', None)
                        pc_metrics[boat] = {'TTK': ttk, 'TTS': tts, 'Ratio': ratio, 'DTL': dtl}

        def apply_subtle_gradient_func(row):
            min_val = row.min()
            max_val = row.max()
            return [get_subtle_gradient(val, min_val, max_val) for val in row]

        if pc_metrics:
            df_combined = pd.DataFrame(pc_metrics).T
            styled_df = df_combined.T.style
            styled_df = styled_df.format("{:.1f}")
            st.markdown("**Pre-Commit Metrics (PC)**")
            st.dataframe(styled_df.apply(apply_subtle_gradient_func, axis=1).format("{:.1f}"), use_container_width=True)

            st.markdown("### Start Track Plot")
            try:
                from start_track_plot import plot_course_and_tracks
                fig, meta = plot_course_and_tracks(
                    course_id=selected_race_id,
                    boats=race_data,
                    twd_col="TWD_MHU_SGP_deg",
                    tts_col="TTS_s",
                    boat_col="BOAT",
                    lat_col="LATITUDE_GPS_unk",
                    lon_col="LONGITUDE_GPS_unk",
                    board_col="LENGTH_DB_H_P_mm"
                )
                st.plotly_chart(
                    fig, use_container_width=True,
                    config={
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': ['resetScale2d'],
                        'displaylogo': False,
                    }
                )
                st.caption("Drag to pan, scroll to zoom, double-click to reset view")
            except ImportError:
                st.warning("start_track_plot.py not found.")
            except Exception as e:
                st.error(f"Error creating track plot: {str(e)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")


@st.fragment
def render_legs_tab(race_data, selected_boats, selected_legs, show_stability):
    """Legs tab — runs independently as a fragment."""
    st.markdown("## Leg-by-Leg Analysis")
    for leg_num in selected_legs:
        st.markdown(f"### Leg {int(leg_num)}")
        leg_data = race_data[race_data['TRK_LEG_NUM_unk'] == leg_num]
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_leg_tracks(leg_data, selected_boats, int(leg_num))
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"leg_track_{leg_num}")
        with col2:
            summary_styled = create_summary_table_styled(leg_data, selected_boats, int(leg_num))
            if summary_styled is not None:
                st.markdown(f"**Leg {int(leg_num)} Summary**")
                st.dataframe(summary_styled, use_container_width=True)
                if show_stability:
                    st.markdown("**Stability Metrics**")
                    stability_styled = create_stability_table_styled(leg_data, selected_boats)
                    if stability_styled is not None:
                        st.dataframe(stability_styled, use_container_width=True)
        st.markdown("---")


@st.fragment
def render_wind_tab(race_data, race_data_full, selected_boats):
    """Wind tab — runs independently as a fragment."""
    st.markdown("## Wind Tactical Analysis")

    wind_boat = st.selectbox(
        "Select boat for wind analysis",
        options=selected_boats, index=0,
        key="wind_boat_select_v4"
    )

    fig = plot_wind_analysis(race_data, wind_boat)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="wind_analysis_chart")
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
    wind_stats_df = pd.DataFrame(wind_stats_data).T.round(0)
    styled_df = wind_stats_df.style.apply(
        lambda row: [get_gradient_color(val, row.min(), row.max(), reverse=True) for val in row],
        axis=0
    ).format("{:.0f}")
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("### Wind Maps (IDW Interpolation)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**True Wind Speed (TWS)**")
        try:
            tws_fig = create_wind_map(race_data_full, 'TWS_MHU_SGP_km_h_1', 'TWS Distribution', colorscale='Viridis')
            st.plotly_chart(tws_fig, use_container_width=True, key="tws_map")
        except Exception as e:
            st.error(f"Error creating TWS map: {str(e)}")
    with col2:
        st.markdown("**True Wind Direction (TWD)**")
        try:
            twd_fig = create_wind_map(race_data_full, 'TWD_MHU_SGP_deg', 'TWD Distribution', colorscale='RdYlGn')
            st.plotly_chart(twd_fig, use_container_width=True, key="twd_map")
        except Exception as e:
            st.error(f"Error creating TWD map: {str(e)}")


@st.fragment
def render_tactical_tab(race_data_full, selected_race_id, race_date):
    """Tactical tab — runs independently as a fragment."""
    st.markdown("## Tactical Analysis - ALL BOATS")
    try:
        marks_path = f"marks/wind_data_{race_date}.csv"
        if Path(marks_path).exists():
            marks_df = pd.read_csv(marks_path)
            boat_data = race_data_full[['BOAT', 'DATETIME', 'LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk',
                                        'TWD_MHU_SGP_deg', 'BOAT_SPEED_km_h_1', 'TWA_BOW_SGP_deg']].copy()
            
            boat_data = boat_data.rename(columns={'BOAT': 'boat', 'DATETIME': '_time'})
            legs_df, leg_summary_df = analyze_gate_crossings(boat_data, marks_df)

            if not legs_df.empty:
                # --- Filter controls ---
                fc1, fc2, fc3 = st.columns([1, 1, 2])
                total_legs = len(leg_summary_df)
                with fc1:
                    options = sorted(
                        set([max(total_legs, 1), 15, 10, 5, 3, 2, 1]),
                        reverse=True
                    )

                    show_top_n = st.select_slider(
                        "Top N legs",
                        options=options,
                        value=options[0],
                        key="tact_top_n",
                        help="Show only the N fastest legs (by time)"
)
                with fc2:
                    filter_mode = st.radio("Filter mode", ["Overall", "Per turn"],
                                           horizontal=True, key="tact_filter_mode",
                                           help="Overall = top N across all turns; Per turn = top N per Left/Right turn")

                def _filter_legs(summary_df, legs_data, leg_type):
                    """Filter leg_summary and legs_df to keep only top N fastest."""
                    s = summary_df[summary_df["leg_type"] == leg_type].copy()
                    if show_top_n >= len(s):
                        return s, legs_data[legs_data["leg_id"].isin(s["leg_id"])]
                    if filter_mode == "Per turn":
                        kept = s.groupby("gate", group_keys=False).apply(
                            lambda g: g.nsmallest(show_top_n, "time_seconds")
                        )
                    else:
                        kept = s.nsmallest(show_top_n, "time_seconds")
                    kept_ids = kept["leg_id"].unique()
                    return kept, legs_data[legs_data["leg_id"].isin(kept_ids)]

                uw_tab, dw_tab = st.tabs(["Upwind (ALL BOATS)", "Downwind (ALL BOATS)"])

                with uw_tab:
                    st.markdown("### All Boats - Upwind Legs")
                    uw_summary_full = leg_summary_df[leg_summary_df["leg_type"] == "uw"]
                    uw_summary, uw_legs = _filter_legs(leg_summary_df, legs_df, "uw")
                    if len(uw_summary) > 0:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            label = f"Showing {len(uw_summary)}/{len(uw_summary_full)} legs"
                            st.caption(f"Line width = speed | Color = BSP | {label}")
                            plot = create_multi_boat_tactical_plot(uw_legs, uw_summary, 'uw')
                            if plot:
                                st.image(plot, use_container_width=True)
                        with col2:
                            st.markdown("**Gate Performance**")
                            st.write("Median time (s) for each gate based on top-5 fastest legs.")
                            gate_summary = create_gate_summary_table(uw_summary, 'uw')
                            if gate_summary is not None:
                                st.dataframe(gate_summary, use_container_width=True)
                            st.metric("Displayed UW Legs", f"{len(uw_summary)} / {len(uw_summary_full)}")
                            st.metric("Avg Time", f"{uw_summary['time_seconds'].mean():.1f}s")
                    else:
                        st.warning("No upwind legs detected")

                with dw_tab:
                    st.markdown("### All Boats - Downwind Legs")
                    dw_summary_full = leg_summary_df[leg_summary_df["leg_type"] == "dw"]
                    dw_summary, dw_legs = _filter_legs(leg_summary_df, legs_df, "dw")
                    if len(dw_summary) > 0:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            label = f"Showing {len(dw_summary)}/{len(dw_summary_full)} legs"
                            st.caption(f"Line width = speed | Color = BSP | {label}")
                            plot = create_multi_boat_tactical_plot(dw_legs, dw_summary, 'dw')
                            if plot:
                                st.image(plot, use_container_width=True)
                        with col2:
                            st.markdown("**Gate Performance**")
                            gate_summary = create_gate_summary_table(dw_summary, 'dw')
                            if gate_summary is not None:
                                st.dataframe(gate_summary, use_container_width=True)
                            st.metric("Displayed DW Legs", f"{len(dw_summary)} / {len(dw_summary_full)}")
                            st.metric("Avg Time", f"{dw_summary['time_seconds'].mean():.1f}s")
                    else:
                        st.warning("No downwind legs detected")
            else:
                st.warning("No legs detected - check data quality")
        else:
            st.error(f"Marks file not found: {marks_path}")
    except Exception as e:
        st.error(f"Tactical analysis error: {str(e)}")


@st.fragment
def render_pdf_tab(df_full, selected_race_id, selected_boats, selected_legs):
    """PDF export tab — runs independently as a fragment."""
    st.markdown("## Download Race Report")
    st.markdown("Generate a professional PDF report with all race analysis. Uses matplotlib rendering (no Chrome required).")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Generate PDF Report", use_container_width=True, key="gen_pdf_v4"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_buffer = generate_race_report_pdf(
                        df_full, selected_race_id, selected_boats, selected_legs
                    )
                    st.success("PDF generated successfully!")
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"race_report_{int(selected_race_id)}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Flying Roos - Race Analysis",
        page_icon="🦘",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_flying_roos_css()

    # Logo + header
    col_logo, col_spacer = st.columns([1, 3])
    with col_logo:
        st.image("FLYING ROOS.png", width=220)
    render_header()

    # Initialize session state
    if 'race_data' not in st.session_state:
        st.session_state.race_data = None
    if 'race_data_full' not in st.session_state:
        st.session_state.race_data_full = None
    if 'df_full' not in st.session_state:
        st.session_state.df_full = None

    # Sidebar
    with st.sidebar:
        st.markdown("## Data Selection")

        available_dates = discover_race_dates()

        selected_race_id = None
        selected_boats = []
        selected_legs = []
        show_stability = False

        if not available_dates:
            st.error("No log files found in logs/ folder")
        else:
            race_date = st.selectbox(
                "Race Day",
                options=available_dates,
                format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}",
                key="race_date_select"
            )
            data_path = resolve_data_path(race_date)

        if available_dates and data_path:
            file_type = "parquet" if data_path.endswith('.parquet') else "CSV"
            df = load_race_data(data_path)
            st.success(f"{len(df):,} rows loaded ({file_type})")
            df['RH_LEE'] = np.where(df['TWA_MHU_SGP_deg'] > 0, df.LENGTH_RH_P_mm, df.LENGTH_RH_S_mm)

            race_ids = get_unique_race_ids(df)
            if race_ids:
                selected_race_id = st.selectbox(
                    "Race ID", options=race_ids,
                    format_func=lambda x: f"Race {int(x)}",
                    key="race_id_select"
                )
                all_boats = sorted(df['BOAT'].dropna().unique())
                default_boats = ['AUS', 'FRA', 'GBR']
                default = default_boats if all(b in all_boats for b in default_boats) else all_boats[:3]
                selected_boats = st.multiselect(
                    "Select Boats", options=all_boats, default=default,
                    key="boats_select"
                )
                if selected_boats:
                    race_data = df[df['TRK_RACE_NUM_unk'] == selected_race_id]
                    race_data_full = race_data.copy()
                    race_data = race_data[race_data['BOAT'].isin(selected_boats)]

                    # Compute valid legs and detect which have manoeuvres
                    race_df_legs = df[df['TRK_RACE_NUM_unk'] == selected_race_id].copy()
                    race_df_legs = race_df_legs[race_df_legs['TRK_LEG_NUM_unk'] > 0]
                    valid_legs = []
                    legs_with_manoeuvres = []
                    for leg_num in sorted(race_df_legs['TRK_LEG_NUM_unk'].unique()):
                        leg_data = race_df_legs[race_df_legs['TRK_LEG_NUM_unk'] == leg_num]
                        valid_boats = 0
                        leg_has_man = False
                        for boat in selected_boats:
                            boat_leg = leg_data[leg_data['BOAT'] == boat]
                            if len(boat_leg) > 0:
                                duration = boat_leg['DATETIME'].iloc[-1] if len(boat_leg) > 1 else 0
                                if duration != 0:
                                    valid_boats += 1
                                    # Check if this boat has manoeuvres in this leg
                                    try:
                                        man = boat_summary(boat_leg, "MHU")
                                        if len(man) > 1:
                                            leg_has_man = True
                                    except Exception:
                                        pass
                        if valid_boats >= len(selected_boats):
                            valid_legs.append(leg_num)
                            if leg_has_man:
                                legs_with_manoeuvres.append(leg_num)
                    valid_legs = sorted(valid_legs)
                    # Default: only legs with at least 1 manoeuvre
                    default_legs = sorted(legs_with_manoeuvres) if legs_with_manoeuvres else valid_legs

                    selected_legs = st.multiselect(
                        "Select Legs", options=valid_legs, default=default_legs,
                        key="legs_select"
                    )

                    st.session_state.race_data = race_data
                    st.session_state.race_data_full = race_data_full
                    st.session_state.df_full = df

                    st.markdown("---")
                    show_stability = st.checkbox("Add Stability Metrics", value=False, key="stability_check")
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Boats", len(selected_boats))
                    with col2:
                        st.metric("Legs", len(selected_legs))

                    if st.button("Clear Cache", use_container_width=True, key="clear_cache_btn"):
                        st.cache_data.clear()
                        st.rerun()
        elif available_dates and not data_path:
            st.error("File not found for selected date")

    # Main content — each tab is a fragment (reruns independently)
    if selected_race_id and selected_legs and selected_boats and st.session_state.race_data is not None:
        race_data = st.session_state.race_data
        race_data_full = st.session_state.race_data_full
        df_full = st.session_state.df_full

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Start", "Legs", "Wind", "Tactical", "PDF Export"
        ])

        with tab1:
            render_start_tab(race_data, selected_boats, selected_race_id)

        with tab2:
            render_legs_tab(race_data, selected_boats, selected_legs, show_stability)

        with tab3:
            render_wind_tab(race_data, race_data_full, selected_boats)

        with tab4:
            render_tactical_tab(race_data_full, selected_race_id, race_date)

        with tab5:
            render_pdf_tab(df_full, selected_race_id, selected_boats, selected_legs)
    else:
        st.info("Please select race data from the sidebar")


if __name__ == "__main__":
    main()
