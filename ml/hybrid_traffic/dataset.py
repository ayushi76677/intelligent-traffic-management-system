"""
Dataset Pipeline Orchestrator for Hybrid Traffic Intelligence
==============================================================
Loads tracking logs, applies feature engineering, filters usable vehicle tracks,
performs track-aware stratified partitioning, and persists data artifacts.
"""

import json
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    HybridTrafficConfig,
    DEFAULT_TRACKS_FILE,
    OUTPUT_DATA_DIR,
    PROCESSED_FEATURES_FILE,
    TRAIN_DATA_FILE,
    VAL_DATA_FILE,
    TEST_DATA_FILE,
    METADATA_FILE,
)
from .feature_engineering import FeatureEngineer
from .split_dataset import TrackStratifiedSplitter


class HybridTrafficDataset:
    """
    End-to-end dataset manager for processing, filtering, and splitting
    traffic tracking data for machine learning model development.
    """

    def __init__(self, config: Optional[HybridTrafficConfig] = None):
        self.config = config or HybridTrafficConfig()
        self.feature_engineer = FeatureEngineer(self.config)
        self.splitter = TrackStratifiedSplitter(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            random_seed=self.config.random_seed,
        )

        self.raw_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.usable_df: Optional[pd.DataFrame] = None
        self.discarded_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, any] = {}

    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Loads and verifies traffic_tracks.csv."""
        path = file_path or self.config.tracks_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tracking file not found: {path}")

        df = pd.read_csv(path)
        required = ["frame", "time", "track_id", "class", "x1", "y1", "x2", "y2"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Tracking file is missing required columns: {missing}")

        self.raw_df = df
        return df

    def run_pipeline(self) -> Dict[str, any]:
        """
        Executes complete feature engineering and dataset generation pipeline.
        Returns execution telemetry and validation summary.
        """
        if self.raw_df is None:
            self.load_raw_data()

        print(f"[Dataset] Ingested raw observations: {len(self.raw_df):,}")
        print(f"[Dataset] Raw unique tracks: {self.raw_df['track_id'].nunique():,}")

        # 1. Feature Engineering
        print("[Dataset] Computing spatial, kinematic, perspective, and zone features...")
        self.features_df = self.feature_engineer.process_dataframe(self.raw_df)
        print(f"[Dataset] Engineered {len(self.features_df.columns)} feature columns.")

        # 2. Track length analysis & Usability Filtering
        track_lengths = self.features_df.groupby("track_id").size()
        min_len = self.config.min_track_length
        usable_track_ids = track_lengths[track_lengths >= min_len].index
        discarded_track_ids = track_lengths[track_lengths < min_len].index

        self.usable_df = self.features_df[self.features_df["track_id"].isin(usable_track_ids)].copy().reset_index(drop=True)
        self.discarded_df = self.features_df[self.features_df["track_id"].isin(discarded_track_ids)].copy().reset_index(drop=True)

        print(f"[Dataset] Usable tracks (>={min_len} obs): {len(usable_track_ids):,} ({len(self.usable_df):,} observations)")
        print(f"[Dataset] Discarded tracks (<{min_len} obs): {len(discarded_track_ids):,} ({len(self.discarded_df):,} observations)")

        # 3. Track-Aware Stratified Split
        print("[Dataset] Partitioning usable tracks into Train/Val/Test sets...")
        self.train_df, self.val_df, self.test_df, split_summary = self.splitter.split(self.usable_df)
        print(f"[Dataset] Train set: {len(self.train_df):,} obs ({split_summary['train_tracks']} tracks)")
        print(f"[Dataset] Val set:   {len(self.val_df):,} obs ({split_summary['val_tracks']} tracks)")
        print(f"[Dataset] Test set:  {len(self.test_df):,} obs ({split_summary['test_tracks']} tracks)")

        # 4. Assemble Metadata
        quantiles = track_lengths.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
        quantiles_clean = {k: round(float(v), 2) for k, v in quantiles.items()}

        usable_quantiles = (
            self.usable_df.groupby("track_id").size()
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
        )
        usable_quantiles_clean = {k: round(float(v), 2) for k, v in usable_quantiles.items()}

        zone_stats = self.usable_df["zone_id"].value_counts().to_dict()
        lane_stats = self.usable_df["lane_id"].value_counts().to_dict()
        class_stats_raw = self.raw_df["class"].value_counts().to_dict()
        class_stats_usable = self.usable_df["class"].value_counts().to_dict()

        self.metadata = {
            "dataset_name": "Hybrid Traffic Intelligence Dataset",
            "source_file": os.path.basename(self.config.tracks_file),
            "total_raw_observations": int(len(self.raw_df)),
            "total_raw_tracks": int(self.raw_df["track_id"].nunique()),
            "min_track_length_threshold": min_len,
            "usable_tracks_count": int(len(usable_track_ids)),
            "discarded_tracks_count": int(len(discarded_track_ids)),
            "usable_observations_count": int(len(self.usable_df)),
            "discarded_observations_count": int(len(self.discarded_df)),
            "track_length_quantiles_all": quantiles_clean,
            "track_length_quantiles_usable": usable_quantiles_clean,
            "classes_raw": {str(k): int(v) for k, v in class_stats_raw.items()},
            "classes_usable": {str(k): int(v) for k, v in class_stats_usable.items()},
            "zone_distribution_usable": {str(k): int(v) for k, v in zone_stats.items()},
            "lane_distribution_usable": {str(k): int(v) for k, v in lane_stats.items()},
            "split_summary": split_summary,
            "feature_columns": list(self.features_df.columns),
            "coordinate_systems": {
                "image_space": "1920x1080 (Top-left origin, +X right, +Y down)",
                "ground_plane": "1200x700 bird's-eye projection via perspective_matrix.npy",
                "real_world_calibration_used": False,
                "speed_units": "pixels/second (image) & normalized units/second (ground)",
            },
        }

        # 5. Persist Datasets
        self.save_artifacts()
        return self.metadata

    def save_artifacts(self) -> None:
        """Persists processed features, splits, and metadata to disk."""
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        print(f"[Dataset] Saving artifacts to: {out_dir}")

        # Save processed features (usable dataset)
        self.usable_df.to_csv(PROCESSED_FEATURES_FILE, index=False)
        print(f"  -> Saved: {PROCESSED_FEATURES_FILE}")

        # Save train/val/test splits
        self.train_df.to_csv(TRAIN_DATA_FILE, index=False)
        self.val_df.to_csv(VAL_DATA_FILE, index=False)
        self.test_df.to_csv(TEST_DATA_FILE, index=False)
        print(f"  -> Saved: {TRAIN_DATA_FILE}")
        print(f"  -> Saved: {VAL_DATA_FILE}")
        print(f"  -> Saved: {TEST_DATA_FILE}")

        # Save metadata JSON
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  -> Saved: {METADATA_FILE}")


def run_pipeline() -> Dict[str, any]:
    """Helper entry point."""
    pipeline = HybridTrafficDataset()
    return pipeline.run_pipeline()


if __name__ == "__main__":
    run_pipeline()
