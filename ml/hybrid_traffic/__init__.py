"""
Hybrid Traffic Intelligence Model - Package Initialization
===========================================================
Modular, non-disruptive machine learning training-data pipeline
built on top of YOLOv8 + ByteTrack tracking observations.
"""

from .config import HybridTrafficConfig
from .feature_engineering import FeatureEngineer, ZoneAssigner, PerspectiveProjector
from .sequence_builder import SequenceBuilder
from .dataset import HybridTrafficDataset, run_pipeline
from .split_dataset import TrackStratifiedSplitter
from .label_schema import (
    LabelSource,
    TaskCategory,
    CongestionLevel,
    RiskLevel,
    MotionState,
    ManeuverType,
    InfractionType,
    TargetDefinition,
    TARGET_REGISTRY,
)
from .label_generation import LabelGenerator
from .feature_adapter import LiveFeatureAdapter, EXACT_FEATURE_NAMES
from .sequence_buffer import TrackSequenceBuffer, TrackBufferState
from .realtime_inference import RealTimeTrafficPredictor
from .traffic_aggregator import TrafficIntelligenceAggregator

__all__ = [
    "HybridTrafficConfig",
    "FeatureEngineer",
    "ZoneAssigner",
    "PerspectiveProjector",
    "SequenceBuilder",
    "HybridTrafficDataset",
    "TrackStratifiedSplitter",
    "LabelSource",
    "TaskCategory",
    "CongestionLevel",
    "RiskLevel",
    "MotionState",
    "ManeuverType",
    "InfractionType",
    "TargetDefinition",
    "TARGET_REGISTRY",
    "LabelGenerator",
    "run_pipeline",
    "LiveFeatureAdapter",
    "EXACT_FEATURE_NAMES",
    "TrackSequenceBuffer",
    "TrackBufferState",
    "RealTimeTrafficPredictor",
    "TrafficIntelligenceAggregator",
]

