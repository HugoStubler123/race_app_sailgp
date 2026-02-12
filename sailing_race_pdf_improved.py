"""
SailGP Race Report PDF Generator - IMPROVED VERSION
Professional race analysis PDFs with enhanced styling and better layouts
"""

import pandas as pd
import numpy as np
import math
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
from pathlib import Path
from geopy.distance import geodesic
from manoeuvres import boat_summary
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

def rotate_coordinates(x, y, angle_deg):
    """Rotate coordinates by angle in degrees"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    return x_rot, y_rot

def interpolate_color(value, min_val, max_val, reverse=False):
    """
    Generate color gradient from green (good) to red (bad)
    reverse=True inverts the gradient (red for high, green for low)
    """
    if max_val == min_val:
        return colors.HexColor('#ffffff')
    
    # Normalize value to 0-1
    normalized = (value - min_val) / (max_val - min_val)
    
    if reverse:
        normalized = 1 - normalized
    
    # Green to Yellow to Red gradient
    if normalized < 0.5:
        # Green to Yellow
        r = int(normalized * 2 * 255)
        g = 255
        b = 0
    else:
        # Yellow to Red
        r = 255
        g = int((1 - (normalized - 0.5) * 2) * 255)
        b = 0
    
    return colors.Color(r/255, g/255, b/255)

def interpolate_blue_gradient(value, min_val, max_val):
    """Generate blue gradient: darker blue for higher values, lighter for lower"""
    if max_val == min_val:
        return colors.HexColor('#BBDEFB')
    
    # Normalize value to 0-1
    normalized = (value - min_val) / (max_val - min_val)
    
    # Light blue to dark blue
    # Light: #BBDEFB, Dark: #0D47A1
    r = int(187 - normalized * (187 - 13))
    g = int(222 - normalized * (222 - 71))
    b = int(251 - normalized * (251 - 161))
    
    return colors.Color(r/255, g/255, b/255)

# ============================================================================
# DATA PROCESSING
# ============================================================================

def calculate_leg_summary(leg_data, boat):
    """Calculate summary statistics for a boat-leg"""
    boat_data = leg_data[leg_data['BOAT'] == boat].copy()
    man = boat_summary(boat_data, "MHU")
    boat_data['DATETIME'] = pd.to_datetime(boat_data['DATETIME'])
    if len(man) > 1:
        mdmg = man.loc[man["distance_vmg_cog"] > 0, "distance_vmg_cog"].mean()
        man_dist_made_good = np.round(mdmg, 0) if pd.notna(mdmg) else np.nan
        num_of_man = max(len(man) - 1, 0)

        # exclude manoeuvre windows
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
        'man_dist_made_good' : man_dist_made_good,
        'num_of_man': num_of_man,
        'heel_stability': heel_stability,
        'rh_stability': rh_stability
    }


def prepare_track_data(leg_data, boats):
    """Prepare rotated track data aligned to mean TWD"""
    # Calculate mean TWD across all boats
    mean_twd = leg_data['TWD_MHU_SGP_deg'].mean()
    
    tracks = {}
    for boat in boats:
        boat_data = leg_data[leg_data['BOAT'] == boat].copy()
        if len(boat_data) == 0:
            continue
        
        # Get GPS coordinates and convert to local cartesian
        lat = boat_data['LATITUDE_GPS_unk'].values
        lon = boat_data['LONGITUDE_GPS_unk'].values
        
        # Simple conversion to meters
        lat_mean = lat.mean()
        x = (lon - lon[0]) * 111320 * math.cos(math.radians(lat_mean))
        y = (lat - lat[0]) * 111320
        
        # Rotate to align with mean TWD (wind pointing up)
        x_rot, y_rot = rotate_coordinates(x, y, -mean_twd)
        
        tracks[boat] = {
            'x': x_rot,
            'y': y_rot,
            'bsp': boat_data['BOAT_SPEED_km_h_1'].values,
            'boat': boat
        }
    
    return tracks, mean_twd

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_track_plot(leg_data, boats, leg_num):
    """Create track plot and return as image bytes"""
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
            line=dict(color=boat_color, width=3)
        ))
    
    fig.update_layout(
        title=f"Leg {leg_num} - Rotated Tracks (Aligned to mean TWD = {mean_twd:.1f}°)",
        xaxis_title="X (m, rotated)",
        yaxis_title="Y (m, rotated - wind ↑)",
        template="plotly_white",
        showlegend=True,
        width=650,
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=10)
    )
    
    # Convert to image bytes
    img_bytes = fig.to_image(format="png", width=650, height=400)
    return BytesIO(img_bytes)



def create_track_plot(leg_data, boats, leg_num):
    """Create track plot (wind-from at top) and return as image bytes"""
    tracks, mean_twd = prepare_track_data(leg_data, boats)

    if not tracks:
        return None

    fig = go.Figure()

    # Rotate so that wind FROM mean_twd points to +Y (top of graph)
    rot_angle = np.radians(90 - mean_twd)
    cos_a = np.cos(rot_angle)
    sin_a = np.sin(rot_angle)

    for boat in boats:
        if boat not in tracks:
            continue

        track = tracks[boat]
        boat_color = COLOR_MAPPING.get(boat, '#888888')

        x = np.asarray(track['x'], dtype=float)
        y = np.asarray(track['y'], dtype=float)

        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        fig.add_trace(go.Scatter(
            x=x_rot,
            y=y_rot,
            mode='lines',
            name=boat,
            line=dict(color=boat_color, width=3)
        ))

    fig.update_layout(
        title=f"Leg {leg_num} - Tracks (Wind from {mean_twd:.1f}° ↑)",
        xaxis_title="Cross-wind (m)",
        yaxis_title="Upwind (m)",
        template="plotly_white",
        showlegend=True,
        width=650,
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=10)
    )

    img_bytes = fig.to_image(format="png", width=650, height=400)
    return BytesIO(img_bytes)


def create_wind_timeseries_plot(race_data, boat='AUS'):
    """Create TWS/TWD time series for first page"""
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
    
    for idx, leg in leg_series.items():
        if leg != prev_leg:
            pair = (int(prev_leg), int(leg))
            label = transition_label.get(pair, "")
            if label and label not in added_labels:
                fig.add_vline(
                    x=idx,
                    line=dict(color=label_color.get(label, "#888888"), width=1.5, dash="dash"),
                    annotation_text=label,
                    annotation_position="top",
                    annotation_font_size=8
                )
                added_labels.add(label)
            prev_leg = leg
    
    fig.update_layout(
        title=f"Wind Analysis - {boat}",
        xaxis_title="Time (index)",
        yaxis=dict(
            title="TWS (kt)",
            range=[max(0, tws_min - 2), tws_max + 2],
            side="left"
        ),
        yaxis2=dict(
            title="TWD (°)",
            overlaying="y",
            side="right",
            range=[max(0, twd_min - 10), min(360, twd_max + 10)]
        ),
        template="plotly_white",
        showlegend=True,
        width=700,
        height=250,
        margin=dict(l=50, r=50, t=40, b=40),
        font=dict(size=8)
    )
    
    img_bytes = fig.to_image(format="png", width=700, height=250)
    return BytesIO(img_bytes)

# ============================================================================
# PDF GENERATION
# ============================================================================

def create_summary_table(leg_data, boats, leg_num):
    """Create summary table data for PDF - TRANSPOSED with COLOR GRADIENTS"""
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary:
            summaries[boat] = summary
    
    if not summaries:
        return None, None
    
    metric_labels = [
        'Time (s)',
        'Distance (m)',
        'Avg BSP',
        'Avg VMG',
        'Avg TWD',
        'Avg TWS',
        '% In Phase'
    ]
    
    metric_keys = [
        'time_seconds',
        'total_distance_m',
        'avg_BSP',
        'avg_VMG',
        'avg_TWD',
        'avg_TWS',
        'pct_time_in_phase'
    ]
    
    # Metrics where higher is better (Green gradient: high=green, low=red)
    higher_is_better = {'avg_BSP', 'avg_VMG', 'pct_time_in_phase'}
    # Metrics where lower is better (Red gradient: low=green, high=red)
    lower_is_better = {'time_seconds'}
    
    headers = ['Metric'] + list(summaries.keys())
    rows = [headers]
    
    # Store color info for each row
    color_data = []
    
    for label, key in zip(metric_labels, metric_keys):
        row = [label]
        values = []
        for boat in summaries.keys():
            val = summaries[boat][key]
            row.append(str(val))
            values.append(val)
        rows.append(row)
        
        # Calculate gradient colors for this metric
        if len(values) > 0:
            min_val = min(values)
            max_val = max(values)
            
            if key in lower_is_better:
                # High values = green, low values = red
                colors_row = [interpolate_color(v, min_val, max_val, reverse=False) for v in values]
            elif key in higher_is_better:
                # Low values = green, high values = red
                colors_row = [interpolate_color(v, min_val, max_val, reverse=True) for v in values]
            else:
                # No color gradient for neutral metrics
                colors_row = [colors.white] * len(values)
            
            color_data.append(colors_row)
        else:
            color_data.append([colors.white] * len(summaries))
    
    return rows, color_data

def create_stability_table(leg_data, boats):
    """Create stability metrics table with COLOR GRADIENTS (lower is better = green)"""
    summaries = {}
    for boat in boats:
        summary = calculate_leg_summary(leg_data, boat)
        if summary and (summary['heel_stability'] is not None or summary['rh_stability'] is not None):
            summaries[boat] = summary
    
    if not summaries:
        return None, None
    
    metric_labels = ['Heel Stability', 'RH Stability']
    metric_keys = ['heel_stability', 'rh_stability']
    
    headers = ['Metric'] + list(summaries.keys())
    rows = [headers]
    
    color_data = []
    
    for label, key in zip(metric_labels, metric_keys):
        row = [label]
        values = []
        for boat in summaries.keys():
            val = summaries[boat].get(key)
            if val is not None:
                row.append(f"{val:.2f}")
                values.append(val)
            else:
                row.append("N/A")
                values.append(None)
        rows.append(row)
        
        # For stability: LOWER is BETTER (green for low, red for high)
        if len([v for v in values if v is not None]) > 0:
            valid_values = [v for v in values if v is not None]
            min_val = min(valid_values)
            max_val = max(valid_values)
            
            colors_row = []
            for v in values:
                if v is not None:
                    # Reverse=True: low values = green, high values = red
                    colors_row.append(interpolate_color(v, min_val, max_val, reverse=True))
                else:
                    colors_row.append(colors.white)
            color_data.append(colors_row)
        else:
            color_data.append([colors.white] * len(summaries))
    
    return rows, color_data

def create_start_table(df_start, selected_boats):
    """Create start metrics table with blue gradient"""
    pc_ttk_values = {}
    pc_tts_values = {}
    pc_ratio_values = {}
    pc_dtl_values = {}
    
    for boat in selected_boats:
        df_subset = df_start[df_start['BOAT'] == boat]
        df_subset_sorted = df_subset.sort_values(by="TTS_s")
        
        if 'LENGTH_DB_H_P_mm' in df_subset_sorted.columns:
            board_values = df_subset_sorted["LENGTH_DB_H_P_mm"]
            
            if len(board_values[board_values.diff().abs() > 1]) > 0:
                first_change_idx = board_values[board_values.diff().abs() > 1].idxmin()
                first_change_row = df_start.loc[first_change_idx]
                
                if 'PC_TTK_s' in first_change_row and 'TTS_s' in first_change_row:
                    pc_ttk_values[boat] = first_change_row["PC_TTK_s"]
                    pc_tts_values[boat] = first_change_row["TTS_s"]
                    pc_ratio_values[boat] = first_change_row["TTS_s"] / first_change_row["PC_TTK_s"]
                    pc_dtl_values[boat] = first_change_row.get('PC_DTL_m', None)
    
    # Prepare tables with blue gradients
    tables = {}
    color_data = {}
    
    for name, data_dict in [
        ('PC_TTK', pc_ttk_values),
        ('PC_TTS', pc_tts_values),
        ('PC_Ratio', pc_ratio_values),
        ('PC_DTL', pc_dtl_values)
    ]:
        if not data_dict:
            continue
        
        headers = ['Boat'] + [boat for boat in selected_boats if boat in data_dict]
        values_row = ['Value'] + [f"{data_dict[boat]:.2f}" if boat in data_dict else "N/A" 
                                   for boat in selected_boats if boat in data_dict]
        
        rows = [headers, values_row]
        
        # Blue gradient: higher values = darker blue
        values = [data_dict[boat] for boat in selected_boats if boat in data_dict]
        if values:
            min_val = min(values)
            max_val = max(values)
            colors_row = [interpolate_blue_gradient(v, min_val, max_val) for v in values]
        else:
            colors_row = [colors.white] * len(data_dict)
        
        tables[name] = rows
        color_data[name] = [colors_row]
    
    return tables, color_data

def generate_race_report_pdf(df, race_id, boats, legs, output_path=None):
    """Generate a professional race report PDF with IMPROVED FORMATTING"""
    
    # Setup document
    if output_path:
        doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    else:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    width, height = letter
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#003DA5'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1a2332'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    leg_title_style = ParagraphStyle(
        'LegTitle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#003DA5'),
        spaceAfter=6,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Title page
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Race Report", title_style))
    story.append(Paragraph(f"Race ID: {int(race_id)}", subtitle_style))
    story.append(Paragraph(f"Boats: {', '.join(boats)}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Filter data
    race_data = df[df['TRK_RACE_NUM_unk'] == race_id]
    race_data = race_data[race_data['BOAT'].isin(boats)]
    
    # Add wind time series plot to first page
    story.append(Paragraph("Wind Analysis", subtitle_style))
    wind_plot = create_wind_timeseries_plot(race_data, boats[0])
    if wind_plot:
        img = Image(wind_plot, width=6.5*inch, height=2.3*inch)
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
    
    # Process each leg - IMPROVED LAYOUT
    for leg_num in legs:
        story.append(PageBreak())
        
        story.append(Paragraph(f"LEG {int(leg_num)}", leg_title_style))
        story.append(Spacer(1, 0.05*inch))
        
        leg_data = race_data[race_data['TRK_LEG_NUM_unk'] == leg_num]
        
        # Keep plot and table together
        leg_elements = []
        
        # Create track plot
        plot_img = create_track_plot(leg_data, boats, int(leg_num))
        if plot_img:
            img = Image(plot_img, width=6*inch, height=3.7*inch)
            leg_elements.append(img)
            leg_elements.append(Spacer(1, 0.15*inch))
        
        # Summary table with color gradients
        table_data, color_data = create_summary_table(leg_data, boats, int(leg_num))
        if table_data:
            num_boats = len(boats)
            metric_col_width = 1.1*inch
            boat_col_width = (6*inch - metric_col_width) / num_boats
            col_widths = [metric_col_width] + [boat_col_width] * num_boats
            
            table = Table(table_data, colWidths=col_widths)
            
            table_style_list = [
                # Header
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2332')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                
                # Body
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                
                # Grid
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#1a2332')),
                
                # First column
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e8f0fe')),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]
            
            # Apply gradient colors
            if color_data:
                for row_idx, colors_row in enumerate(color_data):
                    for col_idx, color in enumerate(colors_row):
                        table_style_list.append(
                            ('BACKGROUND', (col_idx + 1, row_idx + 1), (col_idx + 1, row_idx + 1), color)
                        )
            
            table.setStyle(TableStyle(table_style_list))
            
            leg_elements.append(Paragraph(f"<b>Leg {int(leg_num)} Summary</b>", subtitle_style))
            leg_elements.append(table)
            leg_elements.append(Spacer(1, 0.1*inch))
        
        # Stability table
        stab_table_data, stab_color_data = create_stability_table(leg_data, boats)
        if stab_table_data:
            num_boats = len(boats)
            metric_col_width = 1.1*inch
            boat_col_width = (4*inch - metric_col_width) / num_boats
            col_widths = [metric_col_width] + [boat_col_width] * num_boats
            
            stab_table = Table(stab_table_data, colWidths=col_widths)
            
            stab_style_list = [
                # Header
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a2332')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ('TOPPADDING', (0, 0), (-1, 0), 5),
                
                # Body
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('TOPPADDING', (0, 1), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
                
                # Grid
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1a2332')),
                
                # First column
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e8f0fe')),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ]
            
            # Apply gradient colors
            if stab_color_data:
                for row_idx, colors_row in enumerate(stab_color_data):
                    for col_idx, color in enumerate(colors_row):
                        stab_style_list.append(
                            ('BACKGROUND', (col_idx + 1, row_idx + 1), (col_idx + 1, row_idx + 1), color)
                        )
            
            stab_table.setStyle(TableStyle(stab_style_list))
            
            leg_elements.append(Paragraph(f"<b>Stability Metrics</b>", subtitle_style))
            leg_elements.append(stab_table)
        
        # Use KeepTogether for all leg elements
        story.append(KeepTogether(leg_elements))
        
        # Fastest boat note
        fastest_boat = None
        fastest_time = float('inf')
        for boat in boats:
            summary = calculate_leg_summary(leg_data, boat)
            if summary and summary['time_seconds'] < fastest_time:
                fastest_time = summary['time_seconds']
                fastest_boat = boat
        
        if fastest_boat:
            story.append(Spacer(1, 0.1*inch))
            note_text = f"<b>Fastest:</b> {fastest_boat} ({fastest_time:.1f}s)"
            story.append(Paragraph(note_text, styles['Normal']))
    
    doc.build(story)
    
    if output_path:
        print(f"PDF saved to: {output_path}")
        return None
    else:
        buffer.seek(0)
        return buffer

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SailGP Race Report PDF')
    parser.add_argument('csv_path', help='Path to race data CSV file')
    parser.add_argument('race_id', type=float, help='Race ID to analyze')
    parser.add_argument('--boats', nargs='+', default=['AUS', 'FRA', 'GBR'], help='Boats to include')
    parser.add_argument('--legs', nargs='+', type=int, help='Legs to include (default: auto-detect)')
    parser.add_argument('--output', '-o', default='race_report.pdf', help='Output PDF path')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    if not args.legs:
        race_data = df[df['TRK_RACE_NUM_unk'] == args.race_id]
        race_data = race_data[race_data['BOAT'].isin(args.boats)]
        args.legs = sorted(race_data[race_data['TRK_LEG_NUM_unk'] > 1]['TRK_LEG_NUM_unk'].unique())
        print(f"Auto-detected legs: {args.legs}")
    
    print(f"Generating PDF for Race {int(args.race_id)}...")
    generate_race_report_pdf(df, args.race_id, args.boats, args.legs, args.output)
    print(f"✅ PDF generated successfully: {args.output}")

if __name__ == "__main__":
    main()
