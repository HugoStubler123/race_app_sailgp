"""
Gate Crossing Analyzer for Sailing Race Analysis
Detects gate crossings and builds upwind/downwind legs
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings("ignore")


class ImprovedGateCrossingAnalyzer:
    """
    Gate crossing analyzer that:
    - Uses time-varying gate positions (merge_asof against marks timeline)
    - Uses tolerant line-crossing detection (sign change OR near-line)
    - Debounces repeated same-gate crossings
    - Builds legs from consecutive opposite-gate crossings per boat
    - Uses existing boat column: TWA_BOW_SGP_deg (NO heading/TWA recompute)
    """

    def __init__(
        self,
        max_leg_time_minutes: int = 15,
        debounce_seconds: int = 12,
        near_gate_threshold_m: float = 60.0,
        near_line_threshold_m: float = 6.0
    ):
        self.max_leg_time = timedelta(minutes=max_leg_time_minutes)
        self.debounce_seconds = debounce_seconds
        self.near_gate_threshold_deg = near_gate_threshold_m / 111_000.0
        self.near_line_threshold_deg = near_line_threshold_m / 111_000.0
        self.all_crossings: List[dict] = []
        self.legs_data: List[dict] = []
        self.leg_counter = 0

    @staticmethod
    def signed_side(p, a, b) -> float:
        """Signed cross product wrt directed line a->b (all in lon/lat degrees)."""
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    @staticmethod
    def dist_point_to_segment(p, a, b) -> float:
        """Euclidean distance in lon/lat degrees (OK at ~km scales)."""
        p = np.asarray(p, float)
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        ab = b - a
        if np.allclose(ab, 0):
            return float(np.linalg.norm(p - a))
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
        t = float(np.clip(t, 0, 1))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def crosses_gate_line(self, p1, p2, g1, g2) -> bool:
        """Tolerant crossing detection"""
        s1 = self.signed_side(p1, g1, g2)
        s2 = self.signed_side(p2, g1, g2)
        sign_cross = (s1 * s2) < 0
        gate_len = np.linalg.norm(np.asarray(g2) - np.asarray(g1)) + 1e-12
        d1 = abs(s1) / gate_len
        d2 = abs(s2) / gate_len
        touch_cross = (d1 < self.near_line_threshold_deg) or (d2 < self.near_line_threshold_deg)
        if not (sign_cross or touch_cross):
            return False
        near_gate = (
            self.dist_point_to_segment(p1, g1, g2) < self.near_gate_threshold_deg
            or self.dist_point_to_segment(p2, g1, g2) < self.near_gate_threshold_deg
        )
        return near_gate

    @staticmethod
    def count_twa_sign_changes(twa_series: pd.Series) -> int:
        if len(twa_series) < 2:
            return 0
        s = np.sign(twa_series.astype(float).to_numpy())
        s = s[s != 0]
        if len(s) < 2:
            return 0
        return int(np.sum(np.diff(s) != 0))

    @staticmethod
    def label_gate_from_twa(leg_type: str, twa0: float) -> str:
        """Gate label from TWA sign"""
        if np.isnan(twa0):
            return "Unknown"
        if leg_type == "uw":
            return "Right turn" if np.sign(twa0) > 0 else "Left turn"
        else:
            return "Left turn" if np.sign(twa0) > 0 else "Right turn"

    def analyze_leg(self, leg_data: pd.DataFrame) -> Dict:
        """Analyze completed leg"""
        if leg_data.empty or len(leg_data) < 2:
            return {
                "avg_TWD": np.nan,
                "time_seconds": 0.0,
                "twa_sign_changes": 0,
                "avg_BSP": np.nan,
                "avg_distance_meters": 0.0,
                "twa0": np.nan,
                "twa0_sign": np.nan,
            }
        leg_data = leg_data.sort_values("_time").copy()
        time_diff = (leg_data["_time"].iloc[-1] - leg_data["_time"].iloc[0]).total_seconds()
        twa_series = leg_data["TWA_BOW_SGP_deg"].astype(float)
        twa0 = float(twa_series.iloc[0])
        avg_bsp = float(leg_data["BOAT_SPEED_km_h_1"].astype(float).mean())
        return {
            "avg_TWD": float(leg_data["TWD_MHU_SGP_deg"].astype(float).mean()),
            "time_seconds": float(time_diff),
            "twa_sign_changes": self.count_twa_sign_changes(twa_series),
            "avg_BSP": avg_bsp,
            "avg_distance_meters": (avg_bsp / 3.6) * float(time_diff),
            "twa0": twa0,
            "twa0_sign": float(np.sign(twa0)),
        }

    @staticmethod
    def build_marks_wide(marks_df: pd.DataFrame) -> pd.DataFrame:
        """Convert marks timeline to wide format"""
        marks_df = marks_df.copy()
        marks_df["DATETIME"] = pd.to_datetime(marks_df["DATETIME"])
        wide = marks_df.pivot_table(
            index="DATETIME",
            columns="mark",
            values=["LATITUDE_MDSS_deg", "LONGITUDE_MDSS_deg"],
            aggfunc="first",
        ).sort_index()
        wide.columns = [f"{a}_{b}" for (a, b) in wide.columns]
        wide = wide.reset_index().sort_values("DATETIME")
        return wide

    def process_data(self, boat_df: pd.DataFrame, marks_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        boat_df = boat_df.copy()
        marks_df = marks_df.copy()
        
        # Required columns check
        required_boat = ["boat", "_time", "LATITUDE_GPS_unk", "LONGITUDE_GPS_unk", 
                        "TWD_MHU_SGP_deg", "BOAT_SPEED_km_h_1", "TWA_BOW_SGP_deg"]
        missing_boat = [c for c in required_boat if c not in boat_df.columns]
        if missing_boat:
            raise ValueError(f"Boat dataframe missing columns: {missing_boat}")

        required_marks = ["DATETIME", "mark", "LATITUDE_MDSS_deg", "LONGITUDE_MDSS_deg"]
        missing_marks = [c for c in required_marks if c not in marks_df.columns]
        if missing_marks:
            raise ValueError(f"Marks dataframe missing columns: {missing_marks}")

        boat_df["_time"] = pd.to_datetime(boat_df["_time"])
        marks_wide = self.build_marks_wide(marks_df)
        
        boat_df = boat_df.sort_values(["boat", "_time"])
        marks_wide = marks_wide.sort_values("DATETIME")
        
        merged_parts = []
        for boat_name, dfb in boat_df.groupby("boat", sort=False):
            dfb = dfb.sort_values("_time")
            m = pd.merge_asof(
                dfb, marks_wide,
                left_on="_time", right_on="DATETIME",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=2),
            )
            merged_parts.append(m)
        boat_df = pd.concat(merged_parts, ignore_index=True)

        required_gate_cols = [
            "LATITUDE_MDSS_deg_WG1", "LONGITUDE_MDSS_deg_WG1",
            "LATITUDE_MDSS_deg_WG2", "LONGITUDE_MDSS_deg_WG2",
            "LATITUDE_MDSS_deg_LG1", "LONGITUDE_MDSS_deg_LG1",
            "LATITUDE_MDSS_deg_LG2", "LONGITUDE_MDSS_deg_LG2",
        ]
        missing = [c for c in required_gate_cols if c not in boat_df.columns]
        if missing:
            raise ValueError(f"Missing required gate columns: {missing}")

        boats = sorted(boat_df["boat"].unique())
        crossings = []

        for boat_name in boats:
            bt = boat_df[boat_df["boat"] == boat_name].sort_values("_time").reset_index(drop=True)
            for i in range(1, len(bt)):
                prev = bt.iloc[i - 1]
                curr = bt.iloc[i]
                p1 = (float(prev["LONGITUDE_GPS_unk"]), float(prev["LATITUDE_GPS_unk"]))
                p2 = (float(curr["LONGITUDE_GPS_unk"]), float(curr["LATITUDE_GPS_unk"]))
                wg1 = (float(curr["LONGITUDE_MDSS_deg_WG1"]), float(curr["LATITUDE_MDSS_deg_WG1"]))
                wg2 = (float(curr["LONGITUDE_MDSS_deg_WG2"]), float(curr["LATITUDE_MDSS_deg_WG2"]))
                lg1 = (float(curr["LONGITUDE_MDSS_deg_LG1"]), float(curr["LATITUDE_MDSS_deg_LG1"]))
                lg2 = (float(curr["LONGITUDE_MDSS_deg_LG2"]), float(curr["LATITUDE_MDSS_deg_LG2"]))
                t = curr["_time"]
                if self.crosses_gate_line(p1, p2, wg1, wg2):
                    crossings.append({"boat": boat_name, "gate": "WG", "timestamp": t, "index": i})
                if self.crosses_gate_line(p1, p2, lg1, lg2):
                    crossings.append({"boat": boat_name, "gate": "LG", "timestamp": t, "index": i})

        crossings = sorted(crossings, key=lambda x: x["timestamp"])
        
        # Debounce
        per_boat: Dict[str, List[dict]] = {}
        for c in crossings:
            per_boat.setdefault(c["boat"], []).append(c)

        cleaned_crossings = []
        for boat_name, clist in per_boat.items():
            clist = sorted(clist, key=lambda x: x["timestamp"])
            filt = []
            for c in clist:
                if (filt and c["gate"] == filt[-1]["gate"] and 
                    (c["timestamp"] - filt[-1]["timestamp"]).total_seconds() < self.debounce_seconds):
                    filt[-1] = c
                else:
                    filt.append(c)
            cleaned_crossings.extend(filt)

        cleaned_crossings = sorted(cleaned_crossings, key=lambda x: x["timestamp"])
        self.all_crossings = cleaned_crossings

        # Build legs
        self.legs_data = []
        self.leg_counter = 0

        for boat_name in boats:
            clist = [c for c in cleaned_crossings if c["boat"] == boat_name]
            clist = sorted(clist, key=lambda x: x["timestamp"])

            for k in range(len(clist) - 1):
                a = clist[k]
                b = clist[k + 1]
                if a["gate"] == b["gate"]:
                    continue

                start_time = a["timestamp"]
                end_time = b["timestamp"]
                duration = end_time - start_time
                if duration > self.max_leg_time:
                    continue

                leg_boat_data = boat_df[
                    (boat_df["boat"] == boat_name) &
                    (boat_df["_time"] >= start_time) &
                    (boat_df["_time"] <= end_time)
                ].copy()

                leg_type = "uw" if a["gate"] == "LG" else "dw"
                leg_summary = self.analyze_leg(leg_boat_data)
                gate_label = self.label_gate_from_twa(leg_type, leg_summary["twa0"])

                self.leg_counter += 1
                self.legs_data.append({
                    "leg_id": self.leg_counter,
                    "boat": boat_name,
                    "leg_type": leg_type,
                    "start_gate": a["gate"],
                    "end_gate": b["gate"],
                    "start_time": start_time,
                    "end_time": end_time,
                    "leg_data": leg_boat_data,
                    "gate": gate_label,
                    "twa0": leg_summary["twa0"],
                    "twa0_sign": leg_summary["twa0_sign"],
                    **{k: v for k, v in leg_summary.items() if k not in ["twa0", "twa0_sign"]},
                })

        legs_df = self.get_legs_dataframe()
        leg_summary_df = self.get_leg_summary_dataframe()
        return legs_df, leg_summary_df

    def get_legs_dataframe(self) -> pd.DataFrame:
        if not self.legs_data:
            return pd.DataFrame()
        out = []
        for leg in self.legs_data:
            df = leg["leg_data"].copy()
            df["leg_id"] = leg["leg_id"]
            df["boat_name"] = leg["boat"]
            df["leg_type"] = leg["leg_type"]
            df["start_gate"] = leg["start_gate"]
            df["end_gate"] = leg["end_gate"]
            df["leg_start_time"] = leg["start_time"]
            df["leg_end_time"] = leg["end_time"]
            df["gate_label"] = leg.get("gate", None)
            df["twa0"] = leg.get("twa0", np.nan)
            df["twa0_sign"] = leg.get("twa0_sign", np.nan)
            out.append(df)
        return pd.concat(out, ignore_index=True)

    def get_leg_summary_dataframe(self) -> pd.DataFrame:
        if not self.legs_data:
            return pd.DataFrame()
        rows = []
        for leg in self.legs_data:
            rows.append({
                "leg_id": leg["leg_id"],
                "boat": leg["boat"],
                "leg_type": leg["leg_type"],
                "start_gate": leg["start_gate"],
                "end_gate": leg["end_gate"],
                "start_time": leg["start_time"],
                "end_time": leg["end_time"],
                "gate": leg.get("gate", None),
                "twa0": leg.get("twa0", np.nan),
                "twa0_sign": leg.get("twa0_sign", np.nan),
                "avg_TWD": leg.get("avg_TWD", np.nan),
                "time_seconds": leg.get("time_seconds", np.nan),
                "twa_sign_changes": leg.get("twa_sign_changes", np.nan),
                "avg_BSP": leg.get("avg_BSP", np.nan),
                "avg_distance_meters": leg.get("avg_distance_meters", np.nan),
            })
        return pd.DataFrame(rows)


def analyze_gate_crossings(
    boat_df: pd.DataFrame,
    marks_df: pd.DataFrame,
    max_leg_time_minutes: int = 15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point for gate crossing analysis"""
    analyzer = ImprovedGateCrossingAnalyzer(max_leg_time_minutes=max_leg_time_minutes)
    return analyzer.process_data(boat_df, marks_df)
