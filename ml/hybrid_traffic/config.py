"""
Hybrid Traffic Intelligence - Configuration & Hyperparameters
============================================================
Defines standard paths, zone coordinates mapping, class labels,
and feature engineering / sequence builder settings.

Ground-Truth Principle:
- Pixel coordinates and speeds are strictly kept in image/normalized space.
- No metric km/h conversion is assumed without physical road survey calibration.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Workspace root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Canonical Data Paths
DEFAULT_TRACKS_FILE = os.path.join(BASE_DIR, "traffic_tracks.csv")
DEFAULT_CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration_points.npy")
DEFAULT_PERSPECTIVE_MATRIX = os.path.join(BASE_DIR, "perspective_matrix.npy")

# Output Data Directory
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, "data", "hybrid_traffic")
PROCESSED_FEATURES_FILE = os.path.join(OUTPUT_DATA_DIR, "processed_features.csv")
TRAIN_DATA_FILE = os.path.join(OUTPUT_DATA_DIR, "train_data.csv")
VAL_DATA_FILE = os.path.join(OUTPUT_DATA_DIR, "val_data.csv")
TEST_DATA_FILE = os.path.join(OUTPUT_DATA_DIR, "test_data.csv")
METADATA_FILE = os.path.join(OUTPUT_DATA_DIR, "dataset_metadata.json")
REPORT_PATH = os.path.join(BASE_DIR, "HYBRID_DATASET_REPORT.md")

# Sequence Dataset Directory
SEQUENCES_DIR = os.path.join(OUTPUT_DATA_DIR, "sequences")
SEQUENCE_METADATA_FILE = os.path.join(SEQUENCES_DIR, "sequence_metadata.json")
SEQUENCE_REPORT_PATH = os.path.join(BASE_DIR, "HYBRID_SEQUENCE_REPORT.md")

# Label Dataset Directory
LABELS_DIR = os.path.join(OUTPUT_DATA_DIR, "labels")
LABEL_METADATA_FILE = os.path.join(LABELS_DIR, "label_metadata.json")
LABEL_REPORT_PATH = os.path.join(BASE_DIR, "HYBRID_LABEL_REPORT.md")

# Input files for label generation
APPROACHING_ANALYSIS_FILE = os.path.join(BASE_DIR, "approaching_vehicle_analysis.csv")
VIOLATIONS_JSON_FILE = os.path.join(BASE_DIR, "traffic", "violations.json")

# Zone definition files (resolved dynamically, never hard-coded)
DEFAULT_ZONE_FILES = {
    "ZONE_1": os.path.join(BASE_DIR, "lane_zone_points.npy"),
    "ZONE_2": os.path.join(BASE_DIR, "zone_2_points.npy"),
    "ZONE_3": os.path.join(BASE_DIR, "zone_3_points.npy"),
    "ZONE_4": os.path.join(BASE_DIR, "zone_4_points.npy"),
    "ZONE_5": os.path.join(BASE_DIR, "zone_5_points.npy"),
    "ZONE_6": os.path.join(BASE_DIR, "zone_6_points.npy"),
}

# Functional lane mapping based on corridor roles
# Zone 3 is the central intersection junction box (crossway), not an individual lane.
ZONE_TO_LANE_MAP = {
    "ZONE_1": "LANE_NORTH_INFLOW",
    "ZONE_2": "LANE_EAST_TURNING",
    "ZONE_3": "UNKNOWN",  # Core crossing junction box
    "ZONE_4": "LANE_SOUTH_QUEUE",
    "ZONE_5": "LANE_WEST_INFLOW",
    "ZONE_6": "LANE_SE_OUTFLOW",
}

# Vehicle class mapping (YOLOv8 / COCO class IDs)
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Video and Sensor Parameters
VIDEO_FPS = 30.0
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Kinematics and Filter Thresholds
MIN_TRACK_LENGTH = 10              # Tracks with <10 observations are discarded from usable feature set
MOVEMENT_THRESHOLD_PX = 1.0        # Below this pixel displacement, vehicle is stationary
STOPPED_SPEED_THRESHOLD_PX = 5.0   # Speed (px/s) below which vehicle is considered stopped
ROLLING_VARIANCE_WINDOW = 5        # Observations window for local movement variance calculation

# Splitting Ratios (Track-level stratified split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Sequence Builder Defaults (for temporal models)
SEQUENCE_LENGTH = 20               # 20 timesteps (~0.67s at 30fps)
FORECAST_HORIZON = 5               # 5 timesteps ahead (~0.16s)
SEQUENCE_STRIDE = 1                # Step size between consecutive sequences


@dataclass
class HybridTrafficConfig:
    """Config container allowing override of parameters."""
    tracks_file: str = DEFAULT_TRACKS_FILE
    calibration_file: str = DEFAULT_CALIBRATION_FILE
    perspective_matrix_file: str = DEFAULT_PERSPECTIVE_MATRIX
    output_dir: str = OUTPUT_DATA_DIR
    sequences_dir: str = SEQUENCES_DIR
    zone_files: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ZONE_FILES))
    zone_to_lane_map: Dict[str, str] = field(default_factory=lambda: dict(ZONE_TO_LANE_MAP))
    class_names: Dict[int, str] = field(default_factory=lambda: dict(CLASS_NAMES))
    
    fps: float = VIDEO_FPS
    min_track_length: int = MIN_TRACK_LENGTH
    movement_threshold_px: float = MOVEMENT_THRESHOLD_PX
    stopped_speed_threshold_px: float = STOPPED_SPEED_THRESHOLD_PX
    variance_window: int = ROLLING_VARIANCE_WINDOW
    
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    random_seed: int = RANDOM_SEED
    
    sequence_length: int = SEQUENCE_LENGTH
    forecast_horizon: int = FORECAST_HORIZON
    sequence_stride: int = SEQUENCE_STRIDE
