import json
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET

import plotly.graph_objects as go

# ----------------------------
# Geometry helpers
# ----------------------------
EARTH_R = 6371000.0  # meters

def latlon_to_xy_m(lat, lon, lat0, lon0):
    """
    Local tangent plane XY (meters), equirectangular approximation.
    x: East (m), y: North (m)
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(lat0)
    lon0 = float(lon0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = EARTH_R * dlon * np.cos(np.radians(lat0))
    y = EARTH_R * dlat
    return x, y

def rotate_xy(x, y, angle_deg):
    """
    Rotate points by +angle_deg (CCW) in (x East, y North).
    With this definition, rotating by +avg_twd aligns the bearing avg_twd with +Y.
    """
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return xr, yr

def compute_plot_extent(*arrays, pad=1.15, min_span=200):
    """
    Choose a symmetric extent for x/y, padded.
    """
    vals = []
    for a in arrays:
        if a is None: 
            continue
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return (-min_span, min_span), (-min_span, min_span)

    allv = np.concatenate(vals)
    vmin, vmax = np.nanmin(allv), np.nanmax(allv)
    span = max(min_span, (vmax - vmin))
    half = 0.5 * span * pad
    mid = 0.5 * (vmin + vmax)
    return (mid - half, mid + half)

def compute_1d_extent(*arrays, pad=1.15, min_span=200):
    vals = []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)

    if not vals:
        return (-min_span, min_span)

    allv = np.concatenate(vals)
    vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
    span = max(min_span, (vmax - vmin))
    half = 0.5 * span * pad
    mid = 0.5 * (vmin + vmax)
    return (mid - half, mid + half)


def compute_xy_extent(x_arrays=(), y_arrays=(), pad=1.15, min_span=200):
    xlim = compute_1d_extent(*x_arrays, pad=pad, min_span=min_span)
    ylim = compute_1d_extent(*y_arrays, pad=pad, min_span=min_span)
    return xlim, ylim


def get_data_xml_list(course_id: str, timeout=15) -> pd.DataFrame:
    """
    Fetch from SailGP XML endpoint.
    Returns a DF with at least column 'xml'.
    """
    try:
        resp = requests.get(f"http://xml.sailgp.tech/xml/{course_id}", timeout=timeout)
        if resp.status_code != 200:
            return pd.DataFrame()

        txt = resp.content.decode("utf-8", errors="ignore")

        # sometimes endpoint returns JSON wrapper
        try:
            maybe = json.loads(txt)
            if isinstance(maybe, dict) and "xmls" in maybe:
                return pd.DataFrame(maybe["xmls"])
        except Exception:
            pass

        return pd.DataFrame({"xml": [txt]})
    except Exception:
        return pd.DataFrame()


def parse_race_xml(xml_text: str):
    """
    Returns:
      marks_df: columns [CompoundMarkName, MarkName, Lat, Lon]
      limits_df: columns [ZoneName, SeqID, Lat, Lon] sorted
    """
    root = ET.fromstring(xml_text)

    # ---- marks
    marks_rows = []
    course = root.find("Course")
    if course is not None:
        for cm in course.findall("CompoundMark"):
            cm_name = cm.attrib.get("Name")
            for m in cm.findall("Mark"):
                marks_rows.append({
                    "CompoundMarkName": cm_name,
                    "MarkName": m.attrib.get("Name"),
                    "Lat": float(m.attrib.get("TargetLat")) if m.attrib.get("TargetLat") else np.nan,
                    "Lon": float(m.attrib.get("TargetLng")) if m.attrib.get("TargetLng") else np.nan,
                })
    marks_df = pd.DataFrame(marks_rows)

    # ---- limits
    limit_rows = []
    for cl in root.findall("CourseLimit"):
        zone_name = cl.attrib.get("name")
        for lim in cl.findall("Limit"):
            limit_rows.append({
                "ZoneName": zone_name,
                "SeqID": int(lim.attrib.get("SeqID")) if lim.attrib.get("SeqID") else np.nan,
                "Lat": float(lim.attrib.get("Lat")) if lim.attrib.get("Lat") else np.nan,
                "Lon": float(lim.attrib.get("Lon")) if lim.attrib.get("Lon") else np.nan,
            })
    limits_df = (
        pd.DataFrame(limit_rows)
        .dropna(subset=["Lat", "Lon"], how="any")
        .sort_values(["ZoneName", "SeqID"], ignore_index=True)
    )

    return marks_df, limits_df


def get_mark_latlon(marks_df: pd.DataFrame, name: str):
    g = marks_df[marks_df["MarkName"] == name]
    if g.empty:
        return None
    r = g.iloc[0]
    if pd.isna(r["Lat"]) or pd.isna(r["Lon"]):
        return None
    return float(r["Lat"]), float(r["Lon"])


DEFAULT_COLOR_MAPPING = {
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
    "ITA": "mediumseagreen",
}

def plot_course_and_tracks(
    course_id: str,
    boats: pd.DataFrame,               # your boats dataframe (for avg TWD + tracks)         # same as boats, or another DF containing tracks
    *,
    twd_col="TWD_MHU_SGP_deg",
    tts_col="TTS_s",
    boat_col="Boat",
    lat_col="Lat",
    lon_col="Lon",
    board_col="BOARD",                 # <-- set to your actual board column name
    color_mapping=None,
):
    """
    Notebook plot:
    - load XML by course_id
    - compute avg TWD from 'boats'
    - rotate to align upwind(avg TWD) with +Y
    - plot boundary (thin red), SL1/SL2, dashed SL line, +135° rays, tracks in TTS window
    - yellow marker per boat at board_values[board_values.diff().abs()>1].idxmin()
    """
    color_mapping = color_mapping or DEFAULT_COLOR_MAPPING

    # ---- load & parse xml
    data_df = get_data_xml_list(course_id)
    if data_df.empty or "xml" not in data_df.columns:
        raise ValueError(f"Could not load XML for course_id={course_id}")

    xml_str = data_df["xml"].dropna().astype(str).iloc[0]
    marks_df, limits_df = parse_race_xml(xml_str)

    # ---- extract key marks
    SL1 = get_mark_latlon(marks_df, "SL1")
    SL2 = get_mark_latlon(marks_df, "SL2")
    if SL1 is None or SL2 is None:
        raise ValueError("SL1 or SL2 not found in XML marks.")

    # ---- center = avg(SL1, SL2)
    clat = 0.5 * (SL1[0] + SL2[0])
    clon = 0.5 * (SL1[1] + SL2[1])

    # ---- avg TWD from boats
    avg_twd = np.nan
    if isinstance(boats, pd.DataFrame) and (twd_col in boats.columns):
        avg_twd = pd.to_numeric(boats[twd_col], errors="coerce").dropna().mean()
    if not np.isfinite(avg_twd):
        raise ValueError(f"avg_twd is NaN. Check twd_col='{twd_col}' in boats DF.")

    # ---- boundary (ZoneName == Boundary)
    boundary = None
    if not limits_df.empty:
        boundary = limits_df[limits_df["ZoneName"] == "Boundary"].copy()
        if not boundary.empty:
            boundary = boundary.sort_values("SeqID")
            # close polygon
            if not (np.isclose(boundary.iloc[0]["Lat"], boundary.iloc[-1]["Lat"]) and
                    np.isclose(boundary.iloc[0]["Lon"], boundary.iloc[-1]["Lon"])):
                boundary = pd.concat([boundary, boundary.iloc[[0]]], ignore_index=True)

    # ---- convert boundary to rotated XY
    bx = by = None
    if boundary is not None and not boundary.empty:
        bx0, by0 = latlon_to_xy_m(boundary["Lat"].values, boundary["Lon"].values, clat, clon)
        bx, by = rotate_xy(bx0, by0, angle_deg=avg_twd)

    # ---- marks to rotated XY
    marks_use = marks_df.dropna(subset=["Lat", "Lon"], how="any").copy()
    mx0, my0 = latlon_to_xy_m(marks_use["Lat"].values, marks_use["Lon"].values, clat, clon)
    mx, my = rotate_xy(mx0, my0, angle_deg=avg_twd)
    marks_use["x"] = mx
    marks_use["y"] = my

    # ---- SL1/SL2 rotated XY
    sl1x0, sl1y0 = latlon_to_xy_m([SL1[0]], [SL1[1]], clat, clon)
    sl2x0, sl2y0 = latlon_to_xy_m([SL2[0]], [SL2[1]], clat, clon)
    sl1x, sl1y = rotate_xy(sl1x0, sl1y0, angle_deg=avg_twd)
    sl2x, sl2y = rotate_xy(sl2x0, sl2y0, angle_deg=avg_twd)
    sl1x, sl1y = float(sl1x[0]), float(sl1y[0])
    sl2x, sl2y = float(sl2x[0]), float(sl2y[0])

    # ---- track filter
    dfw = boats.copy()
    if tts_col not in dfw.columns:
        raise ValueError(f"Tracks DF missing '{tts_col}'")
    dfw = dfw[(dfw[tts_col] < 60) & (dfw[tts_col] > -10)].copy()

    for c in [lat_col, lon_col]:
        if c not in dfw.columns:
            raise ValueError(f"Tracks DF missing '{c}'")
    if boat_col not in dfw.columns:
        raise ValueError(f"Tracks DF missing '{boat_col}'")

    tx0, ty0 = latlon_to_xy_m(dfw[lat_col].values, dfw[lon_col].values, clat, clon)
    tx, ty = rotate_xy(tx0, ty0, angle_deg=avg_twd)
    dfw["x"] = tx
    dfw["y"] = ty

    # ---- plot
    fig = go.Figure()

    # boundary (thin red line, filled optional)
    if bx is not None and by is not None:
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="lines",
            name="Boundary",
            line=dict(width=1, color="red"),
            hoverinfo="skip",
        ))

    # marks
    fig.add_trace(go.Scatter(
        x=marks_use["x"], y=marks_use["y"],
        mode="markers+text",
        text=marks_use["MarkName"],
        textposition="top center",
        marker=dict(size=10),
        name="Marks",
        hovertemplate="<b>%{text}</b><br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
    ))

    # dashed start line between SL1 & SL2
    fig.add_trace(go.Scatter(
        x=[sl1x, sl2x],
        y=[sl1y, sl2y],
        mode="lines",
        name="Start line",
        line=dict(width=1, dash="dash", color="black"),
        hoverinfo="skip",
    ))

    # two dashed rays from SL1 & SL2 at +135° relative to wind axis (+Y) in rotated frame
    # Wind axis is +Y. Clockwise +135° from +Y => direction toward bottom-right.
    # In XY, angle from +X is -45°.
    # length based on extent
    xlim, ylim = compute_xy_extent(
    x_arrays=(bx, dfw["x"].values),
    y_arrays=(by, dfw["y"].values),
    pad=1.10,
    min_span=400
    )
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    L = max(500.0, 0.6 * span)


    ang_from_x_deg = -45.0
    dx = L * np.cos(np.radians(ang_from_x_deg))
    dy = L * np.sin(np.radians(ang_from_x_deg))

    for (x0, y0, nm) in [(sl1x, sl1y, "SL1 +135°"), (sl2x, sl2y, "SL2 +135°")]:
        fig.add_trace(go.Scatter(
            x=[x0, x0 + dx],
            y=[y0, y0 + dy],
            mode="lines",
            name=nm,
            line=dict(width=1, dash="dash", color="black"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # tracks
    for boat, g in dfw.groupby(boat_col):
        g = g.sort_values(tts_col)
        col = color_mapping.get(str(boat), None)

        fig.add_trace(go.Scatter(
            x=g["x"], y=g["y"],
            mode="lines",
            name=str(boat),
            line=dict(width=3, color=col) if col else dict(width=3),
            hovertemplate=f"<b>{boat}</b><br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<br>{tts_col}=%{{customdata:.1f}} s<extra></extra>",
            customdata=g[tts_col].values,
        ))

        # yellow marker: board_values[board_values.diff().abs()>1].idxmin()
        if board_col in g.columns:
            board_values = pd.to_numeric(g[board_col], errors="coerce")
            s = board_values[board_values.diff().abs() > 1].dropna()
            if not s.empty:
                idx = s.idxmin()  # exactly as requested
                r = g.loc[idx]
                fig.add_trace(go.Scatter(
                    x=[r["x"]], y=[r["y"]],
                    mode="markers",
                    name=f"{boat} event",
                    marker=dict(size=12, color="yellow", line=dict(width=1, color="black")),
                    hovertemplate=f"<b>{boat} marker</b><br>{board_col}=%{{customdata:.2f}}<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<extra></extra>",
                    customdata=[float(r[board_col]) if pd.notna(r[board_col]) else np.nan],
                    showlegend=False,
                ))

    # axes / layout — optimised for interactive use in Streamlit
    fig.update_layout(
        title=dict(
            text=f"Start — rotated by avg TWD = {avg_twd:.1f}° (wind ↑)",
            font=dict(size=14, color="#004D40", family="Inter, sans-serif"),
        ),
        template="plotly_white",
        xaxis_title="Cross-wind (m)",
        yaxis_title="Upwind (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        dragmode="pan",       # default to pan (easier than zoom box)
        height=550,
    )

    # Auto-fit to the data extent instead of hardcoded center
    xlim, ylim = compute_xy_extent(
        x_arrays=(np.array([sl1x, sl2x]), bx, dfw["x"].values),
        y_arrays=(np.array([sl1y, sl2y]), by, dfw["y"].values),
        pad=1.10,
        min_span=600
    )
    fig.update_xaxes(range=list(xlim))
    fig.update_yaxes(range=list(ylim))

    # Interactive config hint (double-click to reset zoom)
    fig.update_layout(
        modebar=dict(orientation="v"),
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
    )


    return fig, {"avg_twd": avg_twd, "center_lat": clat, "center_lon": clon}


