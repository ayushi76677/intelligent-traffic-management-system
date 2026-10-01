"""
Live Feature Adapter for Real-Time Traffic Intelligence
=======================================================

Extracts and adapts live tracking observations into the exact 20 numerical features
required by the trained TCN-Transformer Gated Hybrid model.

Feature Ordering (Strictly 20 features, identical to training and sequence_metadata.json):
 1. center_x                    - Bounding box center X (px)
 2. center_y                    - Bounding box center Y (px)
 3. delta_x                     - Image displacement dX (px)
 4. delta_y                     - Image displacement dY (px)
 5. displacement_image          - Euclidean pixel displacement (px)
 6. speed_image_px_per_sec      - Speed in image plane (px/s)
 7. accel_image_px_per_sec2     - Acceleration in image plane (px/s^2)
 8. heading_rad                 - Direction angle [-pi, +pi] (rad)
 9. heading_change_rad          - Wrapped angular velocity [-pi, +pi] (rad)
10. ground_x                    - Homography projected ground X
11. ground_y                    - Homography projected ground Y
12. ground_delta_x              - Ground displacement dGX
13. ground_delta_y              - Ground displacement dGY
14. displacement_ground         - Euclidean ground displacement
15. ground_speed_norm_per_sec   - Ground speed (norm units/s)
16. box_width                   - Bounding box width (px)
17. box_height                  - Bounding box height (px)
18. box_area                    - Bounding box area (px^2)
19. stopped_duration            - Cumulative consecutive seconds stopped (s)
20. speed_variance_rolling5     - Rolling 5-step speed variance ((px/s)^2)
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.hybrid_traffic.config import (
    HybridTrafficConfig,
    DEFAULT_ZONE_FILES,
    ZONE_TO_LANE_MAP,
    CLASS_NAMES,
    DEFAULT_PERSPECTIVE_MATRIX,
)

logger = logging.getLogger("TrafficIntelligence.FeatureAdapter")

# Exact feature ordering contract
EXACT_FEATURE_NAMES = [
    "center_x",
    "center_y",
    "delta_x",
    "delta_y",
    "displacement_image",
    "speed_image_px_per_sec",
    "accel_image_px_per_sec2",
    "heading_rad",
    "heading_change_rad",
    "ground_x",
    "ground_y",
    "ground_delta_x",
    "ground_delta_y",
    "displacement_ground",
    "ground_speed_norm_per_sec",
    "box_width",
    "box_height",
    "box_area",
    "stopped_duration",
    "speed_variance_rolling5",
]

FEATURE_COUNT = len(EXACT_FEATURE_NAMES)  # 20


class LiveFeatureAdapter:
    """
    Computes mathematically sound, robust, and leakage-free features from live track observations.
    Integrates perspective homography and zone polygons dynamically.
    """

    def __init__(
        self,
        config: Optional[HybridTrafficConfig] = None,
        perspective_matrix_path: Optional[str] = None,
        zone_files: Optional[Dict[str, str]] = None,
        zone_to_lane_map: Optional[Dict[str, str]] = None,
    ):
        self.config = config or HybridTrafficConfig()
        self.perspective_matrix_path = perspective_matrix_path or self.config.perspective_matrix_file
        self.zone_files = zone_files or self.config.zone_files
        self.zone_to_lane = zone_to_lane_map or self.config.zone_to_lane_map

        # Kinematic thresholds
        self.movement_threshold_px = self.config.movement_threshold_px  # 1.0 px
        self.stopped_speed_threshold_px = self.config.stopped_speed_threshold_px  # 5.0 px/s
        self.fps = self.config.fps  # 30.0

        # Load homography matrix
        self.perspective_matrix: Optional[np.ndarray] = None
        self._load_perspective_matrix()

        # Load zone polygons
        self.zone_polygons: Dict[str, np.ndarray] = {}
        self._load_zone_polygons()

    def _load_perspective_matrix(self) -> None:
        """Loads homography matrix from disk if present."""
        if os.path.exists(self.perspective_matrix_path):
            try:
                mat = np.load(self.perspective_matrix_path).astype(np.float64)
                if mat.shape == (3, 3):
                    self.perspective_matrix = mat
                    logger.debug("[FeatureAdapter] Loaded perspective matrix (3x3).")
                else:
                    logger.warning(f"[FeatureAdapter] Invalid matrix shape: {mat.shape}")
            except Exception as e:
                logger.warning(f"[FeatureAdapter] Failed loading perspective matrix: {e}")
        else:
            logger.warning(f"[FeatureAdapter] Perspective matrix not found at: {self.perspective_matrix_path}")

    def _load_zone_polygons(self) -> None:
        """Loads and validates zone polygon coordinates from .npy files."""
        for zone_id, path in self.zone_files.items():
            if os.path.exists(path):
                try:
                    pts = np.load(path).astype(np.int32)
                    if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 3:
                        self.zone_polygons[zone_id] = pts
                except Exception as e:
                    logger.warning(f"[FeatureAdapter] Failed to load zone {zone_id}: {e}")

    def project_ground(self, x: float, y: float) -> Tuple[float, float]:
        """Projects image bottom-center point to ground coordinates."""
        if self.perspective_matrix is None:
            return round(float(x), 2), round(float(y), 2)

        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        try:
            warped = cv2.perspectiveTransform(pt, self.perspective_matrix)
            gx, gy = warped[0, 0, 0], warped[0, 0, 1]
            return round(float(gx), 2), round(float(gy), 2)
        except Exception:
            return round(float(x), 2), round(float(y), 2)

    def assign_zone_and_lane(self, x: float, y: float) -> Tuple[str, str]:
        """Determines spatial containment in defined zones and resolves functional lane."""
        pt = (float(x), float(y))
        for zone_id, poly in self.zone_polygons.items():
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                lane_id = self.zone_to_lane.get(zone_id, "UNKNOWN")
                return zone_id, lane_id
        return "UNKNOWN", "UNKNOWN"

    def compute_observation_features(
        self,
        current_obs: Dict[str, Any],
        prev_obs: Optional[Dict[str, Any]] = None,
        prev_features: Optional[Dict[str, float]] = None,
        recent_speeds: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes 20 features for a single track observation given its previous state.
        
        Args:
            current_obs: Dict with 'x1', 'y1', 'x2', 'y2', and optional 'time'/'frame'.
            prev_obs: Previous observation dict (or None if first observation).
            prev_features: Dict of previously computed features (for heading, stopped duration).
            recent_speeds: List of recent speed values up to window size 5.

        Returns:
            feature_vector: np.ndarray of shape (20,) strictly ordered.
            metadata: Dict with derived geometric and zone attributes.
        """
        x1 = float(current_obs["x1"])
        y1 = float(current_obs["y1"])
        x2 = float(current_obs["x2"])
        y2 = float(current_obs["y2"])

        # 1. Geometric bounding box features
        center_x = round((x1 + x2) / 2.0, 2)
        center_y = round((y1 + y2) / 2.0, 2)
        box_width = round(max(0.0, x2 - x1), 2)
        box_height = round(max(0.0, y2 - y1), 2)
        box_area = round(box_width * box_height, 2)

        # Bottom-center ground contact point
        bottom_x = center_x
        bottom_y = y2

        # Ground projection
        ground_x, ground_y = self.project_ground(bottom_x, bottom_y)

        # Zone and lane
        zone_id, lane_id = self.assign_zone_and_lane(bottom_x, bottom_y)

        # Time delta
        curr_time = float(current_obs.get("time", 0.0))
        if prev_obs is not None:
            prev_time = float(prev_obs.get("time", 0.0))
            time_delta = max(0.0, curr_time - prev_time)
            # Fallback to frame rate if time delta is zero/missing
            if time_delta <= 0.0:
                curr_frame = int(current_obs.get("frame", 0))
                prev_frame = int(prev_obs.get("frame", 0))
                frame_diff = max(1, curr_frame - prev_frame)
                time_delta = frame_diff / self.fps
        else:
            time_delta = 0.0

        # Kinematics relative to previous observation
        if prev_obs is not None and time_delta > 0.0:
            prev_cx = round((float(prev_obs["x1"]) + float(prev_obs["x2"])) / 2.0, 2)
            prev_cy = round((float(prev_obs["y1"]) + float(prev_obs["y2"])) / 2.0, 2)
            delta_x = round(center_x - prev_cx, 2)
            delta_y = round(center_y - prev_cy, 2)
            disp_img = round(float(np.sqrt(delta_x ** 2 + delta_y ** 2)), 2)
            speed_img = round(disp_img / time_delta, 2)

            # Acceleration
            prev_speed = prev_features.get("speed_image_px_per_sec", 0.0) if prev_features else 0.0
            accel_img = round((speed_img - prev_speed) / time_delta, 2)

            # Ground deltas
            prev_gx = prev_features.get("ground_x", ground_x) if prev_features else ground_x
            prev_gy = prev_features.get("ground_y", ground_y) if prev_features else ground_y
            ground_dx = round(ground_x - prev_gx, 2)
            ground_dy = round(ground_y - prev_gy, 2)
            disp_gnd = round(float(np.sqrt(ground_dx ** 2 + ground_dy ** 2)), 2)
            speed_gnd = round(disp_gnd / time_delta, 2)

            # Heading and Heading Change
            heading_rad = round(float(np.arctan2(delta_y, delta_x)), 4)
            if disp_img < self.movement_threshold_px:
                # Stationary
                heading_change_rad = 0.0
            else:
                prev_heading = prev_features.get("heading_rad", heading_rad) if prev_features else heading_rad
                raw_diff = heading_rad - prev_heading
                # Wrap to [-pi, +pi]
                heading_change_rad = round(float(np.arctan2(np.sin(raw_diff), np.cos(raw_diff))), 4)

            # Stopped duration
            prev_stopped = prev_features.get("stopped_duration", 0.0) if prev_features else 0.0
            if speed_img < self.stopped_speed_threshold_px:
                stopped_duration = round(prev_stopped + time_delta, 3)
            else:
                stopped_duration = 0.0

        else:
            # First observation in track
            delta_x = 0.0
            delta_y = 0.0
            disp_img = 0.0
            speed_img = 0.0
            accel_img = 0.0
            ground_dx = 0.0
            ground_dy = 0.0
            disp_gnd = 0.0
            speed_gnd = 0.0
            heading_rad = 0.0
            heading_change_rad = 0.0
            stopped_duration = 0.0

        # Rolling 5-step speed variance
        speeds = list(recent_speeds) if recent_speeds else []
        speeds.append(speed_img)
        # Keep window 5
        speeds = speeds[-5:]
        if len(speeds) >= 2:
            speed_variance = round(float(np.var(speeds, ddof=1)), 2)
        else:
            speed_variance = 0.0

        # Build feature dictionary
        feat_dict = {
            "center_x": center_x,
            "center_y": center_y,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "displacement_image": disp_img,
            "speed_image_px_per_sec": speed_img,
            "accel_image_px_per_sec2": accel_img,
            "heading_rad": heading_rad,
            "heading_change_rad": heading_change_rad,
            "ground_x": ground_x,
            "ground_y": ground_y,
            "ground_delta_x": ground_dx,
            "ground_delta_y": ground_dy,
            "displacement_ground": disp_gnd,
            "ground_speed_norm_per_sec": speed_gnd,
            "box_width": box_width,
            "box_height": box_height,
            "box_area": box_area,
            "stopped_duration": stopped_duration,
            "speed_variance_rolling5": speed_variance,
        }

        # Construct strict array in EXACT_FEATURE_NAMES order
        feature_vector = np.zeros(FEATURE_COUNT, dtype=np.float32)
        for idx, feat_name in enumerate(EXACT_FEATURE_NAMES):
            val = float(feat_dict.get(feat_name, 0.0))
            if np.isnan(val) or np.isinf(val):
                logger.warning(f"[FeatureAdapter] Non-finite value in feature '{feat_name}': {val}. Clamped to 0.0.")
                val = 0.0
            feature_vector[idx] = val

        metadata = {
            "center_x": center_x,
            "center_y": center_y,
            "bottom_x": bottom_x,
            "bottom_y": bottom_y,
            "ground_x": ground_x,
            "ground_y": ground_y,
            "zone_id": zone_id,
            "lane_id": lane_id,
            "speed_px_per_sec": speed_img,
            "stopped_duration": stopped_duration,
            "feature_dict": feat_dict,
            "recent_speeds": speeds,
        }

        return feature_vector, metadata
