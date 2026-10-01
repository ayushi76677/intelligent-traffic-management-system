"""
Traffic State Adapter for SUMO Telemetry
=========================================
Maps live SUMO / TraCI vehicle observations to the exact 20-feature input
representation required by the frozen TCN-Transformer Gated Hybrid model,
and maintains temporal FIFO sequence buffers with warm-up tracking.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np

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
DEFAULT_SEQUENCE_LENGTH = 20


@dataclass
class VehicleSequenceState:
    """Represents temporal status and sequence tensor for one vehicle."""
    vehicle_id: str
    observation_count: int
    is_ready: bool
    status: str                         # "WARMING_UP" or "ACTIVE"
    warmup_progress: float              # 0.05 to 1.0
    sequence: Optional[np.ndarray]      # Shape (20, 20) float32 when is_ready is True
    latest_feature_vector: np.ndarray   # Shape (20,) float32
    latest_feature_dict: Dict[str, float]
    edge_id: str
    lane_id: str
    raw_state: Dict[str, Any]


class VehicleTrackMemory:
    """Internal temporal memory for one simulated vehicle track."""

    def __init__(self, vehicle_id: str, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        self.vehicle_id = vehicle_id
        self.sequence_length = sequence_length
        self.observations: deque = deque(maxlen=sequence_length)
        self.feature_vectors: deque = deque(maxlen=sequence_length)
        self.recent_speeds: deque = deque(maxlen=5)
        self.prev_obs: Optional[Dict[str, Any]] = None
        self.prev_features: Optional[Dict[str, float]] = None
        self.total_observations: int = 0
        self.last_step: int = -1
        self.last_time: float = 0.0


class TrafficStateAdapter:
    """
    Transforms raw TraCI vehicle observations into normalized 20-feature vectors
    and manages multi-track rolling sequence buffers.
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        network_width: float = 200.0,
        network_height: float = 200.0,
        canvas_width: float = 1920.0,
        canvas_height: float = 1080.0
    ):
        self.sequence_length = sequence_length
        self.network_width = network_width
        self.network_height = network_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.tracks: Dict[str, VehicleTrackMemory] = {}

    def _project_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Projects SUMO metric coordinates (0..200m) to standard video canvas (1920x1080)."""
        u = (x / max(1.0, self.network_width)) * self.canvas_width
        # SUMO Y is bottom-to-top, image canvas Y is top-to-bottom
        v = (1.0 - (y / max(1.0, self.network_height))) * self.canvas_height
        return float(u), float(v)

    def extract_features(
        self,
        obs: Dict[str, Any],
        prev_obs: Optional[Dict[str, Any]],
        prev_feats: Optional[Dict[str, float]],
        recent_speeds: deque,
        dt: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Computes the exact 20-feature representation from TraCI observation.
        Follows PHASE9_FEATURE_MAPPING.md specifications.
        """
        # 1 & 2: Projected image plane center
        gx = float(obs.get("x", 0.0))
        gy = float(obs.get("y", 0.0))
        u, v = self._project_to_canvas(gx, gy)

        # Displacements on image plane
        if prev_obs is not None:
            prev_u, prev_v = self._project_to_canvas(prev_obs["x"], prev_obs["y"])
            delta_x = u - prev_u
            delta_y = v - prev_v
        else:
            delta_x = 0.0
            delta_y = 0.0

        disp_img = math.sqrt(delta_x**2 + delta_y**2)
        speed_img = disp_img / max(1e-4, dt)

        # Image acceleration
        if prev_feats is not None:
            prev_spd = prev_feats.get("speed_image_px_per_sec", speed_img)
            accel_img = (speed_img - prev_spd) / max(1e-4, dt)
        else:
            accel_img = 0.0

        # Heading in radians
        deg = float(obs.get("angle", 0.0))
        heading_rad = (deg * math.pi / 180.0) - math.pi

        if prev_feats is not None:
            prev_heading = prev_feats.get("heading_rad", heading_rad)
            h_diff = heading_rad - prev_heading
            # Wrap to [-pi, pi]
            heading_change = math.atan2(math.sin(h_diff), math.cos(h_diff))
        else:
            heading_change = 0.0

        # Ground coordinates & displacements
        if prev_obs is not None:
            g_dx = gx - float(prev_obs["x"])
            g_dy = gy - float(prev_obs["y"])
        else:
            g_dx = 0.0
            g_dy = 0.0

        disp_ground = math.sqrt(g_dx**2 + g_dy**2)
        ground_speed = float(obs.get("speed", disp_ground / max(1e-4, dt)))

        # Bounding box projection
        veh_w = float(obs.get("width", 2.0))
        veh_l = float(obs.get("length", 5.0))
        box_w = (veh_w / max(1.0, self.network_width)) * self.canvas_width
        box_h = (veh_l / max(1.0, self.network_height)) * self.canvas_height
        box_area = box_w * box_h

        # Stopped duration
        wait_time = float(obs.get("waiting_time", 0.0))
        # Normalization factor matching Phase 5/6 scale
        stopped_dur = min(1.0, wait_time / 60.0)

        # Rolling speed variance
        recent_speeds.append(speed_img)
        if len(recent_speeds) >= 2:
            spd_var = float(np.var(recent_speeds))
        else:
            spd_var = 0.0

        feat_dict = {
            "center_x": u,
            "center_y": v,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "displacement_image": disp_img,
            "speed_image_px_per_sec": speed_img,
            "accel_image_px_per_sec2": accel_img,
            "heading_rad": heading_rad,
            "heading_change_rad": heading_change,
            "ground_x": gx,
            "ground_y": gy,
            "ground_delta_x": g_dx,
            "ground_delta_y": g_dy,
            "displacement_ground": disp_ground,
            "ground_speed_norm_per_sec": ground_speed,
            "box_width": box_w,
            "box_height": box_h,
            "box_area": box_area,
            "stopped_duration": stopped_dur,
            "speed_variance_rolling5": spd_var,
        }

        feat_vec = np.zeros(FEATURE_COUNT, dtype=np.float32)
        for idx, fname in enumerate(EXACT_FEATURE_NAMES):
            feat_vec[idx] = float(feat_dict[fname])

        return feat_vec, feat_dict

    def update_vehicle(
        self,
        obs: Dict[str, Any],
        step_num: int,
        sim_time: float,
        dt: float = 1.0
    ) -> VehicleSequenceState:
        """Ingests one vehicle observation and returns sequence buffer state."""
        vid = obs["vehicle_id"]
        if vid not in self.tracks:
            self.tracks[vid] = VehicleTrackMemory(vid, self.sequence_length)

        track = self.tracks[vid]
        feat_vec, feat_dict = self.extract_features(
            obs=obs,
            prev_obs=track.prev_obs,
            prev_feats=track.prev_features,
            recent_speeds=track.recent_speeds,
            dt=dt
        )

        track.observations.append(obs)
        track.feature_vectors.append(feat_vec)
        track.prev_obs = obs
        track.prev_features = feat_dict
        track.total_observations += 1
        track.last_step = step_num
        track.last_time = sim_time

        obs_count = len(track.feature_vectors)
        is_ready = bool(obs_count >= self.sequence_length)
        status = "ACTIVE" if is_ready else "WARMING_UP"
        warmup_progress = min(1.0, obs_count / float(self.sequence_length))

        seq_arr = None
        if is_ready:
            seq_arr = np.array(track.feature_vectors, dtype=np.float32)

        return VehicleSequenceState(
            vehicle_id=vid,
            observation_count=obs_count,
            is_ready=is_ready,
            status=status,
            warmup_progress=warmup_progress,
            sequence=seq_arr,
            latest_feature_vector=feat_vec,
            latest_feature_dict=feat_dict,
            edge_id=obs.get("edge_id", ""),
            lane_id=obs.get("lane_id", ""),
            raw_state=obs
        )

    def prune_departed(self, active_vehicle_ids: Set[str]):
        """Evicts vehicles that have exited the simulation."""
        to_del = [vid for vid in self.tracks if vid not in active_vehicle_ids]
        for vid in to_del:
            del self.tracks[vid]

    def reset(self):
        """Clears all track memory buffers."""
        self.tracks.clear()
