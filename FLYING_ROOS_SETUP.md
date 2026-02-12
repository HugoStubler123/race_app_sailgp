# BONDS FLYING ROOS - Race Analysis App Setup Guide

## 🎯 What's New

### ✅ Implemented (Priority 1 & 2):

1. **Flying Roos Corporate Branding**
   - Logo in header (icon size, clickable to Google Drive)
   - Teal (#006B5E) primary color scheme
   - Gold (#FFD700) accents
   - Clean, corporate look throughout
   - Professional Roboto font

2. **Tab 4: Tactical Analysis - ALL BOATS**
   - Uses your exact notebook code
   - Shows ALL boats' tracks overlaid
   - TWD deviation coloring (red/yellow/green)
   - Line width = speed (thicker = faster)
   - Separate tabs for Upwind and Downwind
   - Gate performance tables with color coding

3. **Table Formatting**
   - All numbers formatted to max 1 decimal place
   - No more .0000 - clean and readable

4. **Preserved Features**
   - COLOR_MAPPING for boats untouched (as requested)
   - All original functionality maintained

## 📦 Files Delivered

1. **sailing_race_app_FLYING_ROOS.py** - Main app with Flying Roos branding
2. **gate_crossing_analyzer.py** - Gate crossing analysis class
3. **sailing_race_pdf_improved.py** - PDF generator (from earlier)

## 🚀 Quick Start

### Setup

1. **Place files in your project:**
   ```bash
   /your_project/
   ├── sailing_race_app_FLYING_ROOS.py
   ├── gate_crossing_analyzer.py
   └── sailing_race_pdf_improved.py
   ```

2. **Ensure data files exist:**
   ```
   /Users/hugostubler/Documents/SailGP /Report_Pipeline/data/logs/
   ├── log_20260118.csv
   └── marks.csv
   ```

3. **Run the app:**
   ```bash
   streamlit run sailing_race_app_FLYING_ROOS.py
   ```

## 📊 Tab 4: Tactical Analysis Details

### How It Works

**Data Flow:**
1. Loads race data (boats) + marks.csv (gate positions)
2. Runs `ImprovedGateCrossingAnalyzer` to detect gate crossings
3. Builds upwind/downwind legs for ALL boats
4. Creates combined visualization using your exact code

**What You See:**
- **ALL boats' tracks** overlaid on one plot
- **Color coding**: TWD deviation from mean (green=headed, red=lifted)
- **Line thickness**: Speed (faster legs = thicker lines)
- **Rotation**: All legs aligned to mean wind direction
- **Tables**: Gate performance with color gradient (green=fast, red=slow)

### Your Exact Code Implementation

The plotting code in `create_multi_boat_tactical_plot()` is a direct copy of your notebook:

```python
# rotation angle (deg -> rad)
rotation_angle = np.deg2rad(summary["avg_TWD"].mean())
cos_angle = np.cos(rotation_angle)
sin_angle = np.sin(rotation_angle)

# line width scaling (inverse of time)
time_values = summary.set_index("leg_id")["time_seconds"].astype(float)
q75 = time_values.quantile(0.75)
q25 = time_values.quantile(0.25)
den = (q75 - q25) if (q75 - q25) != 0 else 1.0

# global color normalization across ALL points
twd_mean = summary["avg_TWD"].mean()
twd_dev_all = (legs["TWD_MHU_SGP_deg"].astype(float) - twd_mean).to_numpy()

# ... exact same logic for plotting all boats
```

## 🎨 Flying Roos Color Scheme

### Colors Used

```python
Primary Teal: #006B5E    # Headers, buttons, metrics
Deep Green:   #004D43    # Button hover, secondary
Gold:         #FFD700    # Accents, AUS highlights
Light Teal:   #00A896    # Backgrounds
Light Gray:   #F5F5F5    # Page background
```

### Where Applied

- **Headers**: Teal (#006B5E)
- **Buttons**: Teal with hover to Deep Green
- **Active Tab**: Teal background
- **Metrics**: Teal colored values
- **Wind Chart**: Teal for TWS, Gold for TWD
- **Logo**: Top left, clickable

### Boat Colors (PRESERVED)

```python
COLOR_MAPPING = {
    "AUS": "#009A00",    # Green - DO NOT TOUCH
    "FRA": "#0E8DFB",    # Blue
    "GBR": "#bd9c67",    # Tan
    # ... etc
}
```

## 📋 Features by Tab

### Tab 1: Start Analysis
- Race overview metrics
- Start metrics with **subtle blue gradient** (readable text)
- PC TTK, TTS values formatted to 1 decimal

### Tab 2: Legs
- Leg-by-leg track visualization
- **Uses COLOR_MAPPING** for boats (preserved)
- Summary tables with **color gradients**:
  - Performance: High=Green, Low=Red
  - Stability: Low=Green, High=Red
- All values **formatted to 1 decimal**

### Tab 3: Wind
- TWS/TWD time series (Teal & Gold colors)
- Wind statistics
- Flying Roos branded charts

### Tab 4: Tactical - ALL BOATS ⭐
- **Upwind tab**: All boats' upwind legs overlaid
- **Downwind tab**: All boats' downwind legs overlaid
- Uses your exact notebook code
- TWD deviation coloring (RdYlGn colormap)
- Line width by speed
- Gate performance tables (green=fast, red=slow)
- All numbers **formatted to 1 decimal**

### Tab 5: PDF
- PDF report generation
- Can integrate sailing_race_pdf_improved.py

## 🔧 Technical Details

### Data Requirements

**For Tactical Analysis (Tab 4):**

Boat data needs:
- boat
- _time (or DATETIME)
- LATITUDE_GPS_unk
- LONGITUDE_GPS_unk
- TWD_MHU_SGP_deg
- BOAT_SPEED_km_h_1
- TWA_BOW_SGP_deg

Marks data (marks.csv) needs:
- DATETIME
- mark (e.g., 'WG1', 'WG2', 'LG1', 'LG2')
- LATITUDE_MDSS_deg
- LONGITUDE_MDSS_deg

### Gate Crossing Logic

The `ImprovedGateCrossingAnalyzer`:
1. Merges time-varying gate positions with boat tracks
2. Detects gate crossings (tolerant detection)
3. Debounces repeated crossings
4. Builds legs from consecutive opposite-gate crossings
5. Labels gates based on TWA sign
6. Calculates leg statistics

## 🎯 Key Differences from Previous Version

### What Changed:

1. **Tab 4 now shows ALL boats** (not just one boat selector)
2. **Flying Roos branding** throughout
3. **Corporate colors** (teal/gold instead of blue/purple)
4. **All numbers formatted** to max 1 decimal
5. **Logo integrated** with clickable link
6. **Cleaner, professional design**

### What's Preserved:

1. **COLOR_MAPPING** for boats in Legs tab (untouched)
2. **All original functionality**
3. **Data processing logic**
4. **Track visualization**

## 🐛 Troubleshooting

### "Marks file not found"
- Ensure marks.csv exists in: `/Users/hugostubler/Documents/SailGP /Report_Pipeline/data/logs/marks.csv`
- Check file name is exactly `marks.csv` (case-sensitive)

### "No legs detected"
- Check data quality (need GPS + TWD columns)
- Verify gate positions in marks.csv
- Try increasing `max_leg_time_minutes` parameter

### "Module not found: gate_crossing_analyzer"
- Ensure gate_crossing_analyzer.py is in same directory as main app
- Check Python path

### Logo not showing
- Logo path: `/mnt/user-data/uploads/1770794708759_FLYING_ROOS.png`
- Copy logo to accessible location or update path in code

## 📈 Usage Tips

1. **Start with Tab 4** to see the tactical overview of all boats
2. **Use Legs tab** to dive into specific leg performance
3. **Color gradients** help spot patterns quickly:
   - Green = good performance
   - Red = poor performance
4. **All numbers clean** - no more decimal overflow

## 🎉 You're Ready!

Your Flying Roos app is fully branded and functional with multi-boat tactical analysis!

**Priority 1 ✅**: Tab 4 with ALL boats using your exact code
**Priority 2 ✅**: Flying Roos colors and logo

Run it and enjoy! 🦘
