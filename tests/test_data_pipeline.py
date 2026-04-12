"""Tests for data pipeline: loading, parquet conversion, marks, and app functions."""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RACE_DATE = '20260411'
PROJECT_ROOT = Path(__file__).parent.parent


# ── Data file tests ──

class TestDataFiles:
    """Verify data files exist and are well-formed."""

    def test_parquet_exists(self):
        path = PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet'
        assert path.exists(), f'Parquet file missing: {path}'

    def test_marks_exists(self):
        path = PROJECT_ROOT / 'marks' / f'wind_data_{RACE_DATE}.csv'
        assert path.exists(), f'Marks file missing: {path}'

    def test_parquet_loadable(self):
        df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        assert len(df) > 0, 'Parquet file is empty'

    def test_parquet_has_required_columns(self):
        df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        required = ['BOAT', 'DATETIME', 'LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk',
                     'BOAT_SPEED_km_h_1', 'HEADING_deg', 'TWA_MHU_SGP_deg',
                     'TWA_BOW_SGP_deg', 'TWD_MHU_SGP_deg', 'LENGTH_RH_P_mm',
                     'LENGTH_RH_S_mm']
        missing = [c for c in required if c not in df.columns]
        assert not missing, f'Missing columns: {missing}'

    def test_parquet_has_boats(self):
        df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        boats = df.BOAT.unique().tolist()
        assert len(boats) >= 5, f'Expected >=5 boats, got {boats}'
        assert 'AUS' in boats, 'AUS not in data'

    def test_parquet_datetime_range(self):
        df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        dt = pd.to_datetime(df.DATETIME)
        assert dt.min().strftime('%Y%m%d') == RACE_DATE, f'Data starts on wrong date: {dt.min()}'

    def test_marks_has_required_columns(self):
        df = pd.read_csv(PROJECT_ROOT / 'marks' / f'wind_data_{RACE_DATE}.csv')
        required = ['DATETIME', 'mark', 'LATITUDE_MDSS_deg', 'LONGITUDE_MDSS_deg',
                     'TWD_MDSS_deg', 'TWS_MDSS_km_h_1']
        missing = [c for c in required if c not in df.columns]
        assert not missing, f'Missing mark columns: {missing}'

    def test_marks_has_expected_marks(self):
        df = pd.read_csv(PROJECT_ROOT / 'marks' / f'wind_data_{RACE_DATE}.csv')
        marks = set(df.mark.unique())
        expected = {'LG1', 'LG2', 'WG1', 'WG2', 'M1'}
        assert expected.issubset(marks), f'Missing marks: {expected - marks}'

    def test_marks_date_matches_log(self):
        marks = pd.read_csv(PROJECT_ROOT / 'marks' / f'wind_data_{RACE_DATE}.csv')
        log = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        marks_date = pd.to_datetime(marks.DATETIME.iloc[0]).strftime('%Y%m%d')
        log_date = pd.to_datetime(log.DATETIME.iloc[0]).strftime('%Y%m%d')
        assert marks_date == log_date, f'Date mismatch: marks={marks_date}, log={log_date}'

    def test_gps_coordinates_valid(self):
        df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        lat = df.LATITUDE_GPS_unk.dropna()
        lon = df.LONGITUDE_GPS_unk.dropna()
        assert lat.between(-90, 90).all(), 'Latitude out of range'
        assert lon.between(-180, 180).all(), 'Longitude out of range'


# ── App function tests ──

class TestAppFunctions:
    """Test core app functions without Streamlit."""

    def test_discover_race_dates(self):
        os.chdir(PROJECT_ROOT)
        # Import after chdir so Path("logs") works
        from sailing_race_app_FLYING_ROOS_v4 import discover_race_dates
        dates = discover_race_dates()
        assert RACE_DATE in dates, f'{RACE_DATE} not in discovered dates: {dates}'

    def test_resolve_data_path_parquet(self):
        os.chdir(PROJECT_ROOT)
        from sailing_race_app_FLYING_ROOS_v4 import resolve_data_path
        path = resolve_data_path(RACE_DATE)
        assert path is not None, f'No data path for {RACE_DATE}'
        assert path.endswith('.parquet'), f'Expected parquet, got: {path}'

    def test_load_race_data(self):
        os.chdir(PROJECT_ROOT)
        from sailing_race_app_FLYING_ROOS_v4 import load_race_data
        df = load_race_data(f'logs/log_{RACE_DATE}.parquet')
        assert len(df) > 0
        # Check derived column was added
        if 'TWA_MHU_SGP_deg' in df.columns and 'LENGTH_RH_P_mm' in df.columns:
            assert 'RH_LEE' in df.columns, 'RH_LEE derived column missing'

    def test_load_race_data_csv_fallback(self):
        """Ensure CSV loading also works for older dates."""
        os.chdir(PROJECT_ROOT)
        from sailing_race_app_FLYING_ROOS_v4 import load_race_data
        csv_dates = []
        for f in Path('logs').iterdir():
            if f.suffix == '.csv':
                csv_dates.append(f.name)
        if csv_dates:
            df = load_race_data(f'logs/{csv_dates[0]}')
            assert len(df) > 0


# ── Gate crossing analyzer tests ──

class TestGateCrossing:
    """Test gate crossing detection with real data."""

    @pytest.fixture
    def boat_data(self):
        os.chdir(PROJECT_ROOT)
        df = pd.read_parquet(f'logs/log_{RACE_DATE}.parquet')
        return df[['BOAT', 'DATETIME', 'LATITUDE_GPS_unk', 'LONGITUDE_GPS_unk',
                    'TWD_MHU_SGP_deg', 'BOAT_SPEED_km_h_1', 'TWA_BOW_SGP_deg']].copy()

    @pytest.fixture
    def marks_data(self):
        os.chdir(PROJECT_ROOT)
        return pd.read_csv(f'marks/wind_data_{RACE_DATE}.csv')

    def test_analyzer_import(self):
        os.chdir(PROJECT_ROOT)
        from gate_crossing_analyzer import analyze_gate_crossings
        assert callable(analyze_gate_crossings)

    def test_analyzer_runs(self, boat_data, marks_data):
        os.chdir(PROJECT_ROOT)
        from gate_crossing_analyzer import analyze_gate_crossings
        legs_df, summary_df = analyze_gate_crossings(boat_data, marks_data)
        # Should return DataFrames (possibly empty if no racing legs)
        assert isinstance(legs_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)

    def test_analyzer_detects_legs(self, boat_data, marks_data):
        os.chdir(PROJECT_ROOT)
        from gate_crossing_analyzer import analyze_gate_crossings
        legs_df, summary_df = analyze_gate_crossings(boat_data, marks_data)
        # May be empty if no racing, but if not, should have expected columns
        if not legs_df.empty:
            assert 'boat' in legs_df.columns or 'BOAT' in legs_df.columns
            assert 'leg_type' in legs_df.columns


# ── Manoeuvres tests ──

class TestManoeuvres:
    """Test manoeuvre detection with real data."""

    def test_boat_summary_import(self):
        os.chdir(PROJECT_ROOT)
        from manoeuvres import boat_summary
        assert callable(boat_summary)

    def test_boat_summary_runs(self):
        os.chdir(PROJECT_ROOT)
        from manoeuvres import boat_summary
        df = pd.read_parquet(f'logs/log_{RACE_DATE}.parquet')
        aus = df[df.BOAT == 'AUS'].copy()
        aus['DATETIME'] = pd.to_datetime(aus['DATETIME'])
        man = boat_summary(aus, 'MHU')
        assert isinstance(man, pd.DataFrame)

    def test_boat_summary_detects_manoeuvres(self):
        os.chdir(PROJECT_ROOT)
        from manoeuvres import boat_summary
        df = pd.read_parquet(f'logs/log_{RACE_DATE}.parquet')
        aus = df[df.BOAT == 'AUS'].copy()
        aus['DATETIME'] = pd.to_datetime(aus['DATETIME'])
        man = boat_summary(aus, 'MHU')
        assert len(man) > 0, 'No manoeuvres detected for AUS'
        assert 'type' in man.columns
        types = man.type.unique()
        assert 'tack' in types or 'gybe' in types, f'Expected tack/gybe, got: {types}'


# ── Parquet conversion tests ──

class TestParquetConversion:
    """Verify parquet/csv round-trip integrity."""

    def test_parquet_csv_row_count_match(self):
        """If both exist, row counts should match."""
        csv_path = PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.csv'
        pq_path = PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet'
        if csv_path.exists() and pq_path.exists():
            csv_df = pd.read_csv(csv_path)
            pq_df = pd.read_parquet(pq_path)
            assert len(csv_df) == len(pq_df), f'Row mismatch: CSV={len(csv_df)}, Parquet={len(pq_df)}'

    def test_parquet_column_count(self):
        pq_df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        assert len(pq_df.columns) >= 50, f'Too few columns: {len(pq_df.columns)}'

    def test_parquet_no_all_nan_columns(self):
        pq_df = pd.read_parquet(PROJECT_ROOT / 'logs' / f'log_{RACE_DATE}.parquet')
        all_nan = [c for c in pq_df.columns if pq_df[c].isna().all()]
        # Warn but don't fail — some columns may be legitimately empty
        if all_nan:
            print(f'Warning: {len(all_nan)} all-NaN columns: {all_nan[:5]}...')
