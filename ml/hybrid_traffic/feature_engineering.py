"""
Feature Engineering Engine for Hybrid Traffic Intelligence
===========================================================
Calculates mathematically sound kinematic, spatial, geometric, and temporal features
from multi-object vehicle tracking logs (traffic_tracks.csv).

Explicit Constraints:
- Image-space velocities are expressed strictly as pixels per second (px/s).
- Ground-plane velocities are expressed as normalized projection units per second.
- NO metric km/h conversion is assumed without physical ground-truth surveyed distances.
- Zone definitions are loaded dynamically from .npy polygon files without hard-coding.
"""

import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from .config import (
    HybridTrafficConfig,
    CLASS_NAMES,
    DEFAULT_ZONE_FILES,
    ZONE_TO_LANE_MAP,
)


class ZoneAssigner:
    """
    Loads spatial zone polygons dynamically from .npy files and determines
    point containment for vehicle tire-road contact patches.
    """

    def __init__(self, zone_files: Optional[Dict[str, str]] = None, zone_to_lane_map: Optional[Dict[str, str]] = None):
        self.zone_files = zone_files or DEFAULT_ZONE_FILES
        self.zone_to_lane = zone_to_lane_map or ZONE_TO_LANE_MAP
        self.polygons: Dict[str, np.ndarray] = {}
        self._load_zones()

    def _load_zones(self) -> None:
        """Dynamically loads and validates zone polygon coordinates from disk."""
        for zone_id, file_path in self.zone_files.items():
            if os.path.exists(file_path):
                try:
                    pts = np.load(file_path).astype(np.int32)
                    if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3:
                        self.polygons[zone_id] = pts
                except Exception as e:
                    print(f"[ZoneAssigner] Warning: Failed to load {file_path}: {e}")
            else:
                print(f"[ZoneAssigner] Warning: Zone file not found: {file_path}")

    def assign(self, x: float, y: float) -> Tuple[str, str]:
        """
        Assigns zone_id and lane_id given contact point coordinates (x, y).
        Returns ('UNKNOWN', 'UNKNOWN') if not contained in any defined zone.
        """
        point = (float(x), float(y))
        for zone_id, polygon in self.polygons.items():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                lane_id = self.zone_to_lane.get(zone_id, "UNKNOWN")
                return zone_id, lane_id
        return "UNKNOWN", "UNKNOWN"

    def batch_assign(self, x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[List[str], List[str]]:
        """Vectorized/batched zone assignment for array of points."""
        zone_ids = []
        lane_ids = []
        for x, y in zip(x_coords, y_coords):
            zid, lid = self.assign(x, y)
            zone_ids.append(zid)
            lane_ids.append(lid)
        return zone_ids, lane_ids


class PerspectiveProjector:
    """
    Applies 2D projective homography transformation to convert image coordinates
    into normalized bird's-eye ground-plane coordinates.
    """

    def __init__(self, matrix_file: Optional[str] = None):
        self.matrix: Optional[np.ndarray] = None
        if matrix_file and os.path.exists(matrix_file):
            try:
                self.matrix = np.load(matrix_file).astype(np.float64)
            except Exception as e:
                print(f"[PerspectiveProjector] Warning: Error loading homography: {e}")

    def transform_points(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects (x, y) coordinates using the loaded homography matrix.
        Returns original coordinates if transformation matrix is unavailable.
        """
        if self.matrix is None:
            return x.copy(), y.copy()

        points = np.column_stack([x, y]).astype(np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(points, self.matrix).reshape(-1, 2)
        return warped[:, 0], warped[:, 1]


class FeatureEngineer:
    """
    Main analytical engine that derives comprehensive spatial, kinematic,
    temporal, and zone features for vehicle tracking observations.
    """

    def __init__(self, config: Optional[HybridTrafficConfig] = None):
        self.config = config or HybridTrafficConfig()
        self.zone_assigner = ZoneAssigner(
            zone_files=self.config.zone_files,
            zone_to_lane_map=self.config.zone_to_lane_map,
        )
        self.projector = PerspectiveProjector(self.config.perspective_matrix_file)

    @staticmethod
    def _compute_heading_direction(delta_x: np.ndarray, delta_y: np.ndarray, movement_threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Calculates heading angle in radians, degrees, and 8-way compass category.
        Note: In image coordinates, +Y is DOWNWARD and +X is RIGHTWARD.
        """
        displacements = np.sqrt(delta_x ** 2 + delta_y ** 2)
        heading_rad = np.arctan2(delta_y, delta_x)
        heading_deg = (np.degrees(heading_rad) + 360.0) % 360.0

        compass_sectors = [
            ("EAST", 337.5, 360.0),
            ("EAST", 0.0, 22.5),
            ("SOUTHEAST", 22.5, 67.5),
            ("SOUTH", 67.5, 112.5),
            ("SOUTHWEST", 112.5, 157.5),
            ("WEST", 157.5, 202.5),
            ("NORTHWEST", 202.5, 247.5),
            ("NORTH", 247.5, 292.5),
            ("NORTHEAST", 292.5, 337.5),
        ]

        directions = []
        for d, deg in zip(displacements, heading_deg):
            if d < movement_threshold:
                directions.append("STATIONARY")
            else:
                matched = "UNKNOWN"
                for name, low, high in compass_sectors:
                    if low <= deg < high:
                        matched = name
                        break
                directions.append(matched)

        return heading_rad, heading_deg, directions

    @staticmethod
    def _compute_heading_change(heading_rad: np.ndarray, is_first: np.ndarray, is_stat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates difference between consecutive headings, wrapped to [-pi, +pi].
        """
        raw_diff = np.diff(heading_rad, prepend=heading_rad[0]).copy()
        raw_diff[is_first] = 0.0
        wrapped_rad = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))
        wrapped_rad = wrapped_rad.copy()
        wrapped_rad[is_stat] = 0.0
        wrapped_deg = np.degrees(wrapped_rad)
        return wrapped_rad, wrapped_deg

    @staticmethod
    def _calculate_consecutive_stopped_duration(is_stopped: np.ndarray, time_deltas: np.ndarray, is_first: np.ndarray) -> np.ndarray:
        """
        Calculates cumulative consecutive seconds a vehicle has remained stopped.
        Resets to 0.0 as soon as the vehicle resumes motion or a new track begins.
        """
        durations = np.zeros(len(is_stopped), dtype=np.float64)
        current = 0.0
        for i in range(len(is_stopped)):
            if is_first[i]:
                current = 0.0
                durations[i] = 0.0
            elif is_stopped[i]:
                current += time_deltas[i]
                durations[i] = current
            else:
                current = 0.0
                durations[i] = 0.0
        return durations

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ingests raw tracking DataFrame and produces the enriched feature set.
        Guarantees deterministic sorting and observation integrity using fast vectorized calculations.
        """
        required = ["frame", "time", "track_id", "class", "x1", "y1", "x2", "y2"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        data = df.copy()

        # Ensure correct data types
        data["frame"] = data["frame"].astype(int)
        data["time"] = data["time"].astype(float)
        data["track_id"] = data["track_id"].astype(int)
        data["class"] = data["class"].astype(int)
        for col in ["x1", "y1", "x2", "y2"]:
            data[col] = data[col].astype(float)

        # 1. Geometric bounding box features
        data["center_x"] = ((data["x1"] + data["x2"]) / 2.0).round(2)
        data["center_y"] = ((data["y1"] + data["y2"]) / 2.0).round(2)
        data["box_width"] = (data["x2"] - data["x1"]).round(2)
        data["box_height"] = (data["y2"] - data["y1"]).round(2)
        data["box_area"] = (data["box_width"] * data["box_height"]).round(2)
        data["aspect_ratio"] = np.where(data["box_height"] > 0, (data["box_width"] / data["box_height"]).round(3), 0.0)

        # Vehicle ground contact point (bottom-center)
        data["bottom_center_x"] = data["center_x"]
        data["bottom_center_y"] = data["y2"]

        # Class naming mapping
        data["vehicle_type"] = data["class"].map(self.config.class_names).fillna("vehicle")

        # 2. Perspective & Ground-Plane Projection
        gx, gy = self.projector.transform_points(
            data["bottom_center_x"].values,
            data["bottom_center_y"].values
        )
        data["ground_x"] = np.round(gx, 2)
        data["ground_y"] = np.round(gy, 2)

        # 3. Zone and Lane Assignment
        zones, lanes = self.zone_assigner.batch_assign(
            data["bottom_center_x"].values,
            data["bottom_center_y"].values
        )
        data["zone_id"] = zones
        data["lane_id"] = lanes

        # Sort strictly by track_id and frame for sequential calculations
        data = data.sort_values(["track_id", "frame"]).reset_index(drop=True)

        # Vectorized track boundary mask (True where a new track begins)
        first_mask = (data["track_id"] != data["track_id"].shift(1)).values

        # 4. Temporal deltas
        data["frame_delta"] = data["frame"].diff().fillna(0).astype(int)
        data.loc[first_mask, "frame_delta"] = 0
        data["time_delta"] = data["time"].diff().fillna(0.0).round(4)
        data.loc[first_mask, "time_delta"] = 0.0
        data.loc[data["time_delta"] < 0, "time_delta"] = 0.0

        # 5. Spatial image-plane displacements
        data["delta_x"] = data["center_x"].diff().fillna(0.0).round(2)
        data.loc[first_mask, "delta_x"] = 0.0
        data["delta_y"] = data["center_y"].diff().fillna(0.0).round(2)
        data.loc[first_mask, "delta_y"] = 0.0
        data["displacement_image"] = np.sqrt(data["delta_x"] ** 2 + data["delta_y"] ** 2).round(2)

        # 6. Ground-plane displacements
        data["ground_delta_x"] = data["ground_x"].diff().fillna(0.0).round(2)
        data.loc[first_mask, "ground_delta_x"] = 0.0
        data["ground_delta_y"] = data["ground_y"].diff().fillna(0.0).round(2)
        data.loc[first_mask, "ground_delta_y"] = 0.0
        data["displacement_ground"] = np.sqrt(data["ground_delta_x"] ** 2 + data["ground_delta_y"] ** 2).round(2)

        # 7. Speed & Acceleration (Safe vectorized division without warnings)
        valid_time = (data["time_delta"] > 0).values

        speed_img = np.zeros(len(data), dtype=np.float64)
        np.divide(
            data["displacement_image"].values,
            data["time_delta"].values,
            out=speed_img,
            where=valid_time,
        )
        data["speed_image_px_per_sec"] = np.round(speed_img, 2)

        speed_gnd = np.zeros(len(data), dtype=np.float64)
        np.divide(
            data["displacement_ground"].values,
            data["time_delta"].values,
            out=speed_gnd,
            where=valid_time,
        )
        data["ground_speed_norm_per_sec"] = np.round(speed_gnd, 2)

        delta_speed = data["speed_image_px_per_sec"].diff().fillna(0.0).to_numpy(copy=True)
        delta_speed[first_mask] = 0.0
        accel_img = np.zeros(len(data), dtype=np.float64)
        np.divide(
            delta_speed,
            data["time_delta"].values,
            out=accel_img,
            where=valid_time,
        )
        data["accel_image_px_per_sec2"] = np.round(accel_img, 2)

        # 8. Direction & Heading
        h_rad, h_deg, directions = self._compute_heading_direction(
            data["delta_x"].values,
            data["delta_y"].values,
            movement_threshold=self.config.movement_threshold_px,
        )
        data["heading_rad"] = np.round(h_rad, 4)
        data["heading_deg"] = np.round(h_deg, 2)
        data["direction"] = directions

        is_stat = (data["direction"] == "STATIONARY").values
        d_rad, d_deg = self._compute_heading_change(h_rad, first_mask, is_stat)
        data["heading_change_rad"] = np.round(d_rad, 4)
        data["heading_change_deg"] = np.round(d_deg, 2)

        # 9. Track-level & Cumulative Metrics
        data["distance_travelled"] = data.groupby("track_id")["displacement_image"].cumsum().round(2)
        data["trajectory_length"] = data.groupby("track_id")["displacement_image"].transform("sum").round(2)

        track_first_time = data.groupby("track_id")["time"].transform("first")
        track_last_time = data.groupby("track_id")["time"].transform("last")
        data["track_duration"] = (track_last_time - track_first_time).round(3)
        data["elapsed_track_time"] = (data["time"] - track_first_time).round(3)
        data["track_total_observations"] = data.groupby("track_id")["frame"].transform("count").astype(int)

        # 10. Stopped Duration
        is_stopped = (data["speed_image_px_per_sec"] < self.config.stopped_speed_threshold_px).values
        data["is_stopped"] = is_stopped
        data["stopped_duration"] = self._calculate_consecutive_stopped_duration(
            is_stopped, data["time_delta"].values, first_mask
        ).round(3)

        # 11. Movement Variance (window = 5)
        w = self.config.variance_window
        data["speed_variance_rolling5"] = (
            data.groupby("track_id")["speed_image_px_per_sec"]
            .transform(lambda s: s.rolling(w, min_periods=1).var().fillna(0.0))
            .round(2)
        )
        data["displacement_variance_rolling5"] = (
            data.groupby("track_id")["displacement_image"]
            .transform(lambda d: d.rolling(w, min_periods=1).var().fillna(0.0))
            .round(2)
        )

        return data
