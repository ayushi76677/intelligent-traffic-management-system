"""
Track-Aware Stratified Dataset Splitter
=======================================
Guarantees zero data leakage by partitioning observations strictly by unique vehicle track_id.
Stratifies across vehicle classes to preserve balanced fleet representation in all splits.

Partitions:
- Train: ~70% of tracks
- Validation: ~15% of tracks
- Test: ~15% of tracks
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import HybridTrafficConfig, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


class TrackStratifiedSplitter:
    """
    Partitions tracking data at the track_id level with class stratification.
    """

    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_seed: int = RANDOM_SEED,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, any]]:
        """
        Splits DataFrame into train, val, and test partitions based on track_id.

        Returns:
            train_df, val_df, test_df, split_summary_dict
        """
        if "track_id" not in df.columns:
            raise ValueError("Input DataFrame must contain 'track_id'")

        # 1. Determine dominant class for each unique track
        track_meta = (
            df.groupby("track_id")
            .agg(
                dominant_class=("class", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                obs_count=("frame", "count"),
            )
            .reset_index()
        )

        unique_tracks = track_meta["track_id"].values
        track_classes = track_meta["dominant_class"].values

        # Compute relative sizes: test_size = test_ratio, val relative to remaining
        test_size = self.test_ratio
        val_relative = self.val_ratio / (self.train_ratio + self.val_ratio)

        # First split: (Train + Val) vs Test
        train_val_tracks, test_tracks = train_test_split(
            unique_tracks,
            test_size=test_size,
            stratify=track_classes,
            random_state=self.random_seed,
        )

        # Get classes for train_val_tracks
        tv_meta = track_meta[track_meta["track_id"].isin(train_val_tracks)]
        tv_tracks = tv_meta["track_id"].values
        tv_classes = tv_meta["dominant_class"].values

        # Second split: Train vs Val
        train_tracks, val_tracks = train_test_split(
            tv_tracks,
            test_size=val_relative,
            stratify=tv_classes,
            random_state=self.random_seed,
        )

        train_set = set(train_tracks)
        val_set = set(val_tracks)
        test_set = set(test_tracks)

        train_df = df[df["track_id"].isin(train_set)].copy().reset_index(drop=True)
        val_df = df[df["track_id"].isin(val_set)].copy().reset_index(drop=True)
        test_df = df[df["track_id"].isin(test_set)].copy().reset_index(drop=True)

        summary = {
            "total_tracks": len(unique_tracks),
            "train_tracks": len(train_tracks),
            "val_tracks": len(val_tracks),
            "test_tracks": len(test_tracks),
            "total_observations": len(df),
            "train_observations": len(train_df),
            "val_observations": len(val_df),
            "test_observations": len(test_df),
            "train_obs_pct": round(len(train_df) / len(df) * 100, 2),
            "val_obs_pct": round(len(val_df) / len(df) * 100, 2),
            "test_obs_pct": round(len(test_df) / len(df) * 100, 2),
            "train_classes": train_df["class"].value_counts().to_dict(),
            "val_classes": val_df["class"].value_counts().to_dict(),
            "test_classes": test_df["class"].value_counts().to_dict(),
        }

        return train_df, val_df, test_df, summary
