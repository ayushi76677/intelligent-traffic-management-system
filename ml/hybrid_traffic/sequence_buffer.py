"""
Temporal Sequence Buffer for Multi-Track Vehicle Trajectory Analytics
======================================================================

Maintains sliding-window temporal feature buffers for active vehicle tracks.
Guarantees strict track-level isolation (no cross-track bleeding), graceful warm-up
accounting for tracks with insufficient history, and automatic pruning of stale tracks.

Contract:
- Sequence length: Exactly 20 timesteps (matches trained TCN-Transformer Hybrid).
- Feature count: Exactly 20 features per timestep.
- Status: 'WARMING_UP' (1..19 observations) or 'READY' (>=20 observations).
- Zero data leakage between distinct track IDs.
"""

import os
import sys
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure workspace root in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.hybrid_traffic.feature_adapter import LiveFeatureAdapter, EXACT_FEATURE_NAMES, FEATURE_COUNT

logger = logging.getLogger("TrafficIntelligence.SequenceBuffer")

DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_MAX_IDLE_FRAMES = 30  # ~1.0 second at 30 fps


@dataclass
class TrackBufferState:
    """Represents instantaneous buffer status for a single vehicle track."""
    track_id: int
    observation_count: int
    sequence_length: int
    is_ready: bool
    status: str                         # "WARMING_UP" or "READY"
    warmup_progress: float              # 0.05 to 1.0
    sequence: Optional[np.ndarray]      # Shape (20, 20) float32 when is_ready is True
    latest_feature_vector: np.ndarray   # Shape (20,) float32
    latest_feature_dict: Dict[str, float]
    zone_id: str
    lane_id: str
    last_frame: int
    last_time: float
    raw_observation: Dict[str, Any]


class TrackState:
    """Internal memory container tracking temporal observations and features for one track."""

    def __init__(self, track_id: int, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        self.track_id = track_id
        self.sequence_length = sequence_length
        self.raw_observations: deque = deque(maxlen=sequence_length)
        self.feature_vectors: deque = deque(maxlen=sequence_length)
        self.prev_obs: Optional[Dict[str, Any]] = None
        self.prev_features: Optional[Dict[str, float]] = None
        self.recent_speeds: List[float] = []
        self.first_frame: int = -1
        self.last_frame: int = -1
        self.last_time: float = 0.0
        self.total_observations: int = 0
        self.zone_id: str = "UNKNOWN"
        self.lane_id: str = "UNKNOWN"


class TrackSequenceBuffer:
    """
    Per-track temporal buffer manager for real-time video intelligence.
    Handles track lifecycle: birth (warmup), continuous updates (ready),
    reappearance, and stale track eviction.
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        max_idle_frames: int = DEFAULT_MAX_IDLE_FRAMES,
        feature_adapter: Optional[LiveFeatureAdapter] = None,
    ):
        self.sequence_length = sequence_length
        self.max_idle_frames = max_idle_frames
        self.feature_adapter = feature_adapter or LiveFeatureAdapter()
        self.tracks: Dict[int, TrackState] = {}
        self.active_tracks_cache: set = set()

    def update(self, observation: Dict[str, Any]) -> TrackBufferState:
        """
        Ingests a live vehicle observation, updates temporal history,
        and returns buffer status and sequence if ready.
        
        Args:
            observation: Dict with 'track_id', 'x1', 'y1', 'x2', 'y2', and optional 'frame', 'time'.
            
        Returns:
            TrackBufferState containing warm-up or ready prediction payload.
        """
        raw_tid = observation.get("track_id")
        if raw_tid is None:
            raise ValueError("Observation is missing required field 'track_id'.")
        track_id = int(raw_tid)

        frame = int(observation.get("frame", 0))
        time_sec = float(observation.get("time", frame / 30.0))

        # Check track existence or stale gap reset
        track = self.tracks.get(track_id)
        if track is None:
            track = TrackState(track_id=track_id, sequence_length=self.sequence_length)
            track.first_frame = frame
            self.tracks[track_id] = track
            logger.debug(f"[SequenceBuffer] Registered new track {track_id} at frame {frame}.")
        else:
            # Handle large temporal gap / ID reuse
            if frame - track.last_frame > self.max_idle_frames:
                logger.info(
                    f"[SequenceBuffer] Track {track_id} frame gap ({frame} - {track.last_frame} > {self.max_idle_frames}). "
                    f"Resetting history to prevent cross-event contamination."
                )
                track = TrackState(track_id=track_id, sequence_length=self.sequence_length)
                track.first_frame = frame
                self.tracks[track_id] = track

        # Compute instantaneous 20 features
        feat_vec, meta = self.feature_adapter.compute_observation_features(
            current_obs=observation,
            prev_obs=track.prev_obs,
            prev_features=track.prev_features,
            recent_speeds=track.recent_speeds,
        )

        # Update track state
        track.raw_observations.append(observation)
        track.feature_vectors.append(feat_vec)
        track.prev_obs = dict(observation)
        track.prev_features = dict(meta["feature_dict"])
        track.recent_speeds = list(meta["recent_speeds"])
        track.last_frame = frame
        track.last_time = time_sec
        track.total_observations += 1
        track.zone_id = meta["zone_id"]
        track.lane_id = meta["lane_id"]

        obs_count = len(track.feature_vectors)
        is_ready = bool(obs_count >= self.sequence_length)
        warmup_progress = min(1.0, float(obs_count) / float(self.sequence_length))
        status = "READY" if is_ready else "WARMING_UP"

        if is_ready:
            sequence_arr = np.stack(list(track.feature_vectors), axis=0).astype(np.float32)
        else:
            sequence_arr = None

        return TrackBufferState(
            track_id=track_id,
            observation_count=obs_count,
            sequence_length=self.sequence_length,
            is_ready=is_ready,
            status=status,
            warmup_progress=round(warmup_progress, 3),
            sequence=sequence_arr,
            latest_feature_vector=feat_vec,
            latest_feature_dict=meta["feature_dict"],
            zone_id=track.zone_id,
            lane_id=track.lane_id,
            last_frame=frame,
            last_time=time_sec,
            raw_observation=dict(observation),
        )

    def cleanup_stale_tracks(self, current_frame: int, max_idle: Optional[int] = None) -> List[int]:
        """
        Removes tracks that have not been updated for max_idle frames.
        Prevents unbounded memory accumulation during long-running streams.
        
        Returns:
            List of evicted track IDs.
        """
        threshold = max_idle if max_idle is not None else self.max_idle_frames
        stale_ids = [
            tid for tid, state in self.tracks.items()
            if (current_frame - state.last_frame) > threshold
        ]

        for tid in stale_ids:
            del self.tracks[tid]
            logger.debug(f"[SequenceBuffer] Pruned stale track {tid} (idle > {threshold} frames).")

        return stale_ids

    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        """Returns internal state for an active track."""
        return self.tracks.get(track_id)

    def get_active_track_ids(self) -> List[int]:
        """Returns list of currently buffered track IDs."""
        return sorted(list(self.tracks.keys()))

    def __len__(self) -> int:
        return len(self.tracks)

    def reset(self) -> None:
        """Flushes all buffered tracks."""
        self.tracks.clear()
        logger.info("[SequenceBuffer] All tracks cleared.")
