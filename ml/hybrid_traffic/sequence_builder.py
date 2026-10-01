"""
Sequence Builder for Temporal & Trajectory Hybrid Traffic Models
================================================================
Constructs sliding-window observation sequences from vehicle tracks
for time-series forecasting, trajectory prediction, and sequential ML models.

Key Capabilities:
1. Strict Track-Level Isolation: Sequences NEVER combine observations across tracks.
2. Leakage-Free Splitting: Partitions tracks into Train (70%), Val (15%), Test (15%)
   BEFORE sequence generation. Zero overlap across splits.
3. Explicit Short-Track Accounting: Tracks shorter than sequence_length are categorized,
   counted, and reported (never silently dropped).
4. Compressed NPZ Persistence: Saves 3D numpy arrays (N, sequence_length, features)
   with associated metadata, frame ranges, class labels, and future step targets.
5. Structured Telemetry: Exports sequence_metadata.json and prints verification checks.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

# Ensure workspace root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import (
        HybridTrafficConfig,
        PROCESSED_FEATURES_FILE,
        SEQUENCES_DIR,
        SEQUENCE_METADATA_FILE,
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED,
    )
except ImportError:
    from ml.hybrid_traffic.config import (
        HybridTrafficConfig,
        PROCESSED_FEATURES_FILE,
        SEQUENCES_DIR,
        SEQUENCE_METADATA_FILE,
        SEQUENCE_LENGTH,
        SEQUENCE_STRIDE,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED,
    )

# Standard 20 numerical features for spatial, kinematic, perspective, and variance modeling
DEFAULT_SEQUENCE_FEATURES = [
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


class SequenceBuilder:
    """
    Constructs temporal sliding-window sequences from multi-object vehicle tracks.
    Enforces strict track-level data isolation to prevent frame-adjacent data leakage.
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        stride: int = SEQUENCE_STRIDE,
        feature_columns: Optional[List[str]] = None,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_seed: int = RANDOM_SEED,
    ):
        if sequence_length < 2:
            raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

        self.seq_len = sequence_length
        self.stride = stride
        self.feature_cols = feature_columns or list(DEFAULT_SEQUENCE_FEATURES)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = random_seed

    def filter_and_audit_tracks(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Audits track lengths against the sequence_length threshold.
        Separates usable tracks from short (discarded) tracks.

        Returns:
            usable_df: DataFrame with tracks meeting sequence_length threshold.
            audit_meta: Dict containing total, usable, and discarded track counts & details.
        """
        if "track_id" not in df.columns:
            raise ValueError("Input DataFrame must contain 'track_id'")

        track_lengths = df.groupby("track_id").size()
        total_tracks = int(len(track_lengths))

        usable_mask = track_lengths >= self.seq_len
        usable_ids = set(track_lengths[usable_mask].index)
        discarded_ids = set(track_lengths[~usable_mask].index)

        usable_df = df[df["track_id"].isin(usable_ids)].copy().reset_index(drop=True)
        discarded_df = df[df["track_id"].isin(discarded_ids)].copy().reset_index(drop=True)

        # Audit discarded short tracks by length
        discarded_lens = track_lengths.loc[list(discarded_ids)]
        discarded_len_dist = discarded_lens.value_counts().sort_index().to_dict()
        discarded_len_dist_clean = {int(k): int(v) for k, v in discarded_len_dist.items()}

        # Class breakdown among discarded tracks
        if not discarded_df.empty:
            discarded_classes = (
                discarded_df.groupby("track_id")["class"]
                .first()
                .value_counts()
                .to_dict()
            )
            discarded_classes_clean = {str(k): int(v) for k, v in discarded_classes.items()}
        else:
            discarded_classes_clean = {}

        audit_meta = {
            "total_tracks": total_tracks,
            "usable_tracks": len(usable_ids),
            "discarded_tracks": len(discarded_ids),
            "usable_observations": len(usable_df),
            "discarded_observations": len(discarded_df),
            "sequence_length_threshold": self.seq_len,
            "discarded_track_length_distribution": discarded_len_dist_clean,
            "discarded_track_class_distribution": discarded_classes_clean,
            "discarded_track_ids": sorted([int(tid) for tid in discarded_ids]),
        }

        return usable_df, audit_meta

    def split_usable_tracks(
        self, usable_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Partitions usable tracks into Train (70%), Validation (15%), and Test (15%)
        using class-stratified track-level partitioning.

        Returns:
            train_track_ids, val_track_ids, test_track_ids, verification_dict
        """
        if usable_df.empty:
            raise ValueError("Cannot split an empty usable DataFrame.")

        # Determine dominant vehicle class for each unique track
        track_meta = (
            usable_df.groupby("track_id")
            .agg(
                dom_class=("class", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                obs_count=("frame", "count"),
            )
            .reset_index()
        )

        unique_tracks = track_meta["track_id"].values
        track_classes = track_meta["dom_class"].values

        # Split: (Train + Val) vs Test
        test_size = self.test_ratio
        val_relative = self.val_ratio / (self.train_ratio + self.val_ratio)

        train_val_tracks, test_tracks = train_test_split(
            unique_tracks,
            test_size=test_size,
            stratify=track_classes,
            random_state=self.seed,
        )

        tv_meta = track_meta[track_meta["track_id"].isin(train_val_tracks)]
        tv_tracks = tv_meta["track_id"].values
        tv_classes = tv_meta["dom_class"].values

        # Split: Train vs Val
        train_tracks, val_tracks = train_test_split(
            tv_tracks,
            test_size=val_relative,
            stratify=tv_classes,
            random_state=self.seed,
        )

        # Leakage verification checks
        train_set = set(train_tracks)
        val_set = set(val_tracks)
        test_set = set(test_tracks)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        leakage_detected = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        verification = {
            "train_track_count": len(train_tracks),
            "val_track_count": len(val_tracks),
            "test_track_count": len(test_tracks),
            "total_usable_tracks": len(unique_tracks),
            "train_track_pct": round(len(train_tracks) / len(unique_tracks) * 100, 2),
            "val_track_pct": round(len(val_tracks) / len(unique_tracks) * 100, 2),
            "test_track_pct": round(len(test_tracks) / len(unique_tracks) * 100, 2),
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_overlap_ids": sorted(list(train_val_overlap)),
            "train_test_overlap_ids": sorted(list(train_test_overlap)),
            "val_test_overlap_ids": sorted(list(val_test_overlap)),
            "leakage_detected": leakage_detected,
            "status": "PASS" if not leakage_detected else "FAIL",
        }

        return train_tracks, val_tracks, test_tracks, verification

    def build_sequences_for_tracks(
        self, df: pd.DataFrame, target_track_ids: Optional[Union[List[int], set, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Builds temporal rolling-window sequences for specified vehicle tracks.
        A sequence NEVER combines observations from different track IDs.

        Returns:
            X: np.ndarray of shape (N, seq_len, num_features)
            y_next_disp: np.ndarray of shape (N, 2) -> (delta_x, delta_y) of next frame (or NaN if terminal)
            has_next: np.ndarray of shape (N,) -> bool flag indicating if next observation exists
            metadata: Dict with track_ids, classes, start_frames, end_frames
        """
        valid_cols = [c for c in self.feature_cols if c in df.columns]
        if not valid_cols:
            raise ValueError(f"None of the requested features {self.feature_cols} exist in the DataFrame.")

        if target_track_ids is not None:
            target_set = set(target_track_ids)
            sub_df = df[df["track_id"].isin(target_set)].copy()
        else:
            sub_df = df.copy()

        X_list: List[np.ndarray] = []
        y_disp_list: List[np.ndarray] = []
        has_next_list: List[bool] = []
        meta_track_ids: List[int] = []
        meta_classes: List[int] = []
        meta_start_frames: List[int] = []
        meta_end_frames: List[int] = []

        # Iterate strictly track by track to guarantee zero cross-track contamination
        for track_id, grp in sub_df.groupby("track_id", sort=False):
            # Guarantee strict chronological ordering by frame
            grp_sorted = grp.sort_values("frame").reset_index(drop=True)
            grp_len = len(grp_sorted)

            if grp_len < self.seq_len:
                # Short track, unable to form a sequence of length seq_len
                continue

            grp_features = grp_sorted[valid_cols].values.astype(np.float32)
            centers = grp_sorted[["center_x", "center_y"]].values.astype(np.float32)
            frames = grp_sorted["frame"].values.astype(np.int64)
            classes = grp_sorted["class"].values.astype(np.int64)

            # Rolling window: start from 0 up to grp_len - seq_len with step stride
            for start in range(0, grp_len - self.seq_len + 1, self.stride):
                end = start + self.seq_len  # Exclusive boundary for length seq_len

                # Input sequence window: exactly seq_len timesteps from this single track
                x_seq = grp_features[start:end]

                # Target displacement: immediately next observation if available
                if end < grp_len:
                    disp = centers[end] - centers[end - 1]
                    has_next = True
                else:
                    disp = np.array([np.nan, np.nan], dtype=np.float32)
                    has_next = False

                X_list.append(x_seq)
                y_disp_list.append(disp)
                has_next_list.append(has_next)
                meta_track_ids.append(int(track_id))
                meta_classes.append(int(classes[0]))
                meta_start_frames.append(int(frames[start]))
                meta_end_frames.append(int(frames[end - 1]))

        if not X_list:
            X = np.empty((0, self.seq_len, len(valid_cols)), dtype=np.float32)
            y_disp = np.empty((0, 2), dtype=np.float32)
            has_next_arr = np.empty((0,), dtype=bool)
            meta = {
                "track_ids": np.empty(0, dtype=np.int64),
                "classes": np.empty(0, dtype=np.int64),
                "start_frames": np.empty(0, dtype=np.int64),
                "end_frames": np.empty(0, dtype=np.int64),
            }
            return X, y_disp, has_next_arr, meta

        X = np.stack(X_list, axis=0)
        y_disp = np.stack(y_disp_list, axis=0)
        has_next_arr = np.array(has_next_list, dtype=bool)
        meta = {
            "track_ids": np.array(meta_track_ids, dtype=np.int64),
            "classes": np.array(meta_classes, dtype=np.int64),
            "start_frames": np.array(meta_start_frames, dtype=np.int64),
            "end_frames": np.array(meta_end_frames, dtype=np.int64),
        }
        return X, y_disp, has_next_arr, meta

    def build_3d_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Backward-compatible method returning (X, y_disp, metadata).
        """
        X, y_disp, _, meta = self.build_sequences_for_tracks(df)
        return X, y_disp, meta

    def build_tabular_lagged_dataset(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Constructs flattened 2D tabular dataset with lagged features for scikit-learn models.
        """
        valid_cols = [c for c in self.feature_cols if c in df.columns]
        rows: List[Dict[str, Union[int, float]]] = []

        for track_id, grp in df.groupby("track_id", sort=False):
            grp_sorted = grp.sort_values("frame").reset_index(drop=True)
            grp_len = len(grp_sorted)
            if grp_len < self.seq_len:
                continue

            grp_arr = grp_sorted[valid_cols].values
            centers = grp_sorted[["center_x", "center_y"]].values
            frames = grp_sorted["frame"].values

            for start in range(0, grp_len - self.seq_len + 1, self.stride):
                end = start + self.seq_len
                row_dict: Dict[str, Union[int, float]] = {
                    "track_id": int(track_id),
                    "current_frame": int(frames[end - 1]),
                }

                # Add lagged feature values: lag0 is current timestep, lag(seq_len-1) is oldest
                for lag_idx in range(self.seq_len):
                    time_offset = self.seq_len - 1 - lag_idx
                    for col_idx, col_name in enumerate(valid_cols):
                        key = f"{col_name}_lag{time_offset}"
                        row_dict[key] = float(grp_arr[start + lag_idx, col_idx])

                # Target displacement
                if end < grp_len:
                    fut_dx = float(centers[end, 0] - centers[end - 1, 0])
                    fut_dy = float(centers[end, 1] - centers[end - 1, 1])
                    row_dict["target_next_dx"] = round(fut_dx, 2)
                    row_dict["target_next_dy"] = round(fut_dy, 2)
                    row_dict["target_next_dist"] = round(float(np.sqrt(fut_dx**2 + fut_dy**2)), 2)
                else:
                    row_dict["target_next_dx"] = np.nan
                    row_dict["target_next_dy"] = np.nan
                    row_dict["target_next_dist"] = np.nan

                rows.append(row_dict)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def print_verification_checks(self, verification: Dict[str, any]) -> None:
        """
        Prints rigorous track-level leakage verification checks to stdout.
        """
        print("\n" + "=" * 72)
        print("DATA LEAKAGE PREVENTION & PARTITION VERIFICATION")
        print("=" * 72)
        print(f"Total Usable Tracks: {verification['total_usable_tracks']}")
        print(f"  - Train Partition: {verification['train_track_count']} tracks ({verification['train_track_pct']}%)")
        print(f"  - Val Partition:   {verification['val_track_count']} tracks ({verification['val_track_pct']}%)")
        print(f"  - Test Partition:  {verification['test_track_count']} tracks ({verification['test_track_pct']}%)")
        print("-" * 72)

        # Check 1
        c1_overlap = verification["train_val_overlap_count"]
        c1_status = "PASS (Empty)" if c1_overlap == 0 else f"FAIL ({c1_overlap} overlapping)"
        print(f"Check 1: train_track_ids ∩ validation_track_ids = empty")
        print(f"         -> Result: {c1_status} [count: {c1_overlap}]")

        # Check 2
        c2_overlap = verification["train_test_overlap_count"]
        c2_status = "PASS (Empty)" if c2_overlap == 0 else f"FAIL ({c2_overlap} overlapping)"
        print(f"Check 2: train_track_ids ∩ test_track_ids = empty")
        print(f"         -> Result: {c2_status} [count: {c2_overlap}]")

        # Check 3
        c3_overlap = verification["val_test_overlap_count"]
        c3_status = "PASS (Empty)" if c3_overlap == 0 else f"FAIL ({c3_overlap} overlapping)"
        print(f"Check 3: validation_track_ids ∩ test_track_ids = empty")
        print(f"         -> Result: {c3_status} [count: {c3_overlap}]")

        print("-" * 72)
        overall_status = verification["status"]
        if overall_status == "PASS":
            print("OVERALL VERIFICATION: PASS — Zero data leakage across partitions.")
        else:
            print("OVERALL VERIFICATION: FAIL — Data leakage detected! Aborting.")
        print("=" * 72 + "\n")

    def run_pipeline(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Executes end-to-end Phase 3 temporal sequence generation:
        1. Ingests processed feature dataset from Phase 2.
        2. Filters & audits short tracks against sequence_length threshold.
        3. Partitions usable tracks into Train/Val/Test (70/15/15) with zero leakage.
        4. Verifies and prints partition independence checks.
        5. Generates 3D temporal sequence arrays for each split and complete set.
        6. Persists compressed NPZ files and sequence_metadata.json.
        7. Returns comprehensive metadata telemetry.
        """
        input_path = data_path or PROCESSED_FEATURES_FILE
        out_dir = output_dir or SEQUENCES_DIR
        os.makedirs(out_dir, exist_ok=True)

        print(f"[SequenceBuilder] Loading processed feature dataset: {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Feature dataset not found: {input_path}")

        df = pd.read_csv(input_path)
        print(f"[SequenceBuilder] Ingested {len(df):,} observations across {df['track_id'].nunique():,} tracks.")

        # 1. Audit tracks against sequence_length
        usable_df, audit_meta = self.filter_and_audit_tracks(df)
        print(f"[SequenceBuilder] Track Auditing (seq_len={self.seq_len}):")
        print(f"  -> Total tracks:     {audit_meta['total_tracks']:,}")
        print(f"  -> Usable tracks:    {audit_meta['usable_tracks']:,} ({audit_meta['usable_observations']:,} obs)")
        print(f"  -> Discarded tracks: {audit_meta['discarded_tracks']:,} ({audit_meta['discarded_observations']:,} obs)")

        # 2. Partition usable tracks
        train_tracks, val_tracks, test_tracks, verification = self.split_usable_tracks(usable_df)

        # 3. Print leakage checks
        self.print_verification_checks(verification)
        if verification["status"] != "PASS":
            raise RuntimeError("Data leakage verification failed!")

        # 4. Generate sequences for each partition
        valid_cols = [c for c in self.feature_cols if c in df.columns]
        feature_names_arr = np.array(valid_cols, dtype=str)

        print(f"[SequenceBuilder] Generating sequence arrays (window={self.seq_len}, stride={self.stride}, features={len(valid_cols)})...")

        # Train Sequences
        X_train, y_train, has_next_train, meta_train = self.build_sequences_for_tracks(usable_df, train_tracks)
        print(f"  -> Train sequences: {len(X_train):,} (shape: {X_train.shape}) across {len(train_tracks)} tracks")

        # Val Sequences
        X_val, y_val, has_next_val, meta_val = self.build_sequences_for_tracks(usable_df, val_tracks)
        print(f"  -> Val sequences:   {len(X_val):,} (shape: {X_val.shape}) across {len(val_tracks)} tracks")

        # Test Sequences
        X_test, y_test, has_next_test, meta_test = self.build_sequences_for_tracks(usable_df, test_tracks)
        print(f"  -> Test sequences:  {len(X_test):,} (shape: {X_test.shape}) across {len(test_tracks)} tracks")

        # All Usable Sequences
        X_all, y_all, has_next_all, meta_all = self.build_sequences_for_tracks(usable_df)
        total_sequences = len(X_all)
        print(f"  -> Total sequences: {total_sequences:,} (shape: {X_all.shape}) across {audit_meta['usable_tracks']} tracks")

        # 5. Sequences-per-track statistics
        seqs_per_track = pd.Series(meta_all["track_ids"]).value_counts()
        seq_stats = seqs_per_track.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
        seq_stats_clean = {k: round(float(v), 2) for k, v in seq_stats.items()}

        # 6. Save compressed NPZ files
        train_file = os.path.join(out_dir, "train_sequences.npz")
        val_file = os.path.join(out_dir, "val_sequences.npz")
        test_file = os.path.join(out_dir, "test_sequences.npz")
        all_file = os.path.join(out_dir, "all_sequences.npz")

        print(f"[SequenceBuilder] Saving compressed NPZ archives to: {out_dir}")

        np.savez_compressed(
            train_file,
            X=X_train,
            y_next_disp=y_train,
            has_next=has_next_train,
            track_ids=meta_train["track_ids"],
            classes=meta_train["classes"],
            start_frames=meta_train["start_frames"],
            end_frames=meta_train["end_frames"],
            feature_names=feature_names_arr,
        )
        print(f"  -> Saved: {train_file} ({os.path.getsize(train_file) / (1024*1024):.2f} MB)")

        np.savez_compressed(
            val_file,
            X=X_val,
            y_next_disp=y_val,
            has_next=has_next_val,
            track_ids=meta_val["track_ids"],
            classes=meta_val["classes"],
            start_frames=meta_val["start_frames"],
            end_frames=meta_val["end_frames"],
            feature_names=feature_names_arr,
        )
        print(f"  -> Saved: {val_file} ({os.path.getsize(val_file) / (1024*1024):.2f} MB)")

        np.savez_compressed(
            test_file,
            X=X_test,
            y_next_disp=y_test,
            has_next=has_next_test,
            track_ids=meta_test["track_ids"],
            classes=meta_test["classes"],
            start_frames=meta_test["start_frames"],
            end_frames=meta_test["end_frames"],
            feature_names=feature_names_arr,
        )
        print(f"  -> Saved: {test_file} ({os.path.getsize(test_file) / (1024*1024):.2f} MB)")

        np.savez_compressed(
            all_file,
            X=X_all,
            y_next_disp=y_all,
            has_next=has_next_all,
            track_ids=meta_all["track_ids"],
            classes=meta_all["classes"],
            start_frames=meta_all["start_frames"],
            end_frames=meta_all["end_frames"],
            feature_names=feature_names_arr,
        )
        print(f"  -> Saved: {all_file} ({os.path.getsize(all_file) / (1024*1024):.2f} MB)")

        # Class distribution of sequences
        train_class_dist = pd.Series(meta_train["classes"]).value_counts().to_dict()
        val_class_dist = pd.Series(meta_val["classes"]).value_counts().to_dict()
        test_class_dist = pd.Series(meta_test["classes"]).value_counts().to_dict()

        # 7. Assemble structured metadata JSON
        metadata = {
            "dataset_name": "Hybrid Traffic Temporal Sequences",
            "phase": "Phase 3: Temporal Sequence Generation",
            "source_file": os.path.basename(input_path),
            "sequence_length": self.seq_len,
            "sequence_stride": self.stride,
            "feature_count": len(valid_cols),
            "feature_names": valid_cols,
            "total_tracks": audit_meta["total_tracks"],
            "usable_tracks": audit_meta["usable_tracks"],
            "discarded_tracks": audit_meta["discarded_tracks"],
            "usable_observations": audit_meta["usable_observations"],
            "discarded_observations": audit_meta["discarded_observations"],
            "total_sequences": total_sequences,
            "sequences_per_track_statistics": seq_stats_clean,
            "discarded_tracks_audit": {
                "reason": f"track_length < sequence_length ({self.seq_len})",
                "length_distribution": audit_meta["discarded_track_length_distribution"],
                "class_distribution": audit_meta["discarded_track_class_distribution"],
                "discarded_track_ids_sample": audit_meta["discarded_track_ids"][:20],
            },
            "split_methodology": "Track-Level Stratified by Dominant Vehicle Class (Zero Data Leakage)",
            "split_ratios_target": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "splits": {
                "train": {
                    "tracks": int(len(train_tracks)),
                    "track_pct": verification["train_track_pct"],
                    "sequences": int(len(X_train)),
                    "sequence_pct": round(len(X_train) / total_sequences * 100, 2) if total_sequences else 0,
                    "array_shape": list(X_train.shape),
                    "file": os.path.basename(train_file),
                    "classes": {str(k): int(v) for k, v in train_class_dist.items()},
                },
                "val": {
                    "tracks": int(len(val_tracks)),
                    "track_pct": verification["val_track_pct"],
                    "sequences": int(len(X_val)),
                    "sequence_pct": round(len(X_val) / total_sequences * 100, 2) if total_sequences else 0,
                    "array_shape": list(X_val.shape),
                    "file": os.path.basename(val_file),
                    "classes": {str(k): int(v) for k, v in val_class_dist.items()},
                },
                "test": {
                    "tracks": int(len(test_tracks)),
                    "track_pct": verification["test_track_pct"],
                    "sequences": int(len(X_test)),
                    "sequence_pct": round(len(X_test) / total_sequences * 100, 2) if total_sequences else 0,
                    "array_shape": list(X_test.shape),
                    "file": os.path.basename(test_file),
                    "classes": {str(k): int(v) for k, v in test_class_dist.items()},
                },
            },
            "data_leakage_verification": {
                "train_val_overlap_count": verification["train_val_overlap_count"],
                "train_test_overlap_count": verification["train_test_overlap_count"],
                "val_test_overlap_count": verification["val_test_overlap_count"],
                "leakage_detected": verification["leakage_detected"],
                "status": verification["status"],
            },
            "storage_format": "Compressed NumPy Archives (.npz)",
            "ready_for_phase_4_model_training": True,
        }

        # Save metadata JSON
        meta_path = os.path.join(out_dir, "sequence_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"  -> Saved: {meta_path}")

        # Summary printout
        print("\n" + "=" * 72)
        print("PHASE 3 TEMPORAL SEQUENCE GENERATION COMPLETED")
        print("=" * 72)
        print(f"  Total Tracks:               {audit_meta['total_tracks']}")
        print(f"  Usable Tracks (>= {self.seq_len} obs):  {audit_meta['usable_tracks']}")
        print(f"  Discarded Tracks (< {self.seq_len} obs):{audit_meta['discarded_tracks']}")
        print(f"  Sequence Length:            {self.seq_len} timesteps")
        print(f"  Sequence Stride:            {self.stride}")
        print(f"  Feature Count:              {len(valid_cols)}")
        print(f"  Total Sequences:            {total_sequences:,}")
        print(f"  Sequences/Track Stats:      min={seq_stats_clean['min']}, median={seq_stats_clean['50%']}, mean={seq_stats_clean['mean']}, max={seq_stats_clean['max']}")
        print(f"  Train Sequences:            {len(X_train):,} ({metadata['splits']['train']['sequence_pct']}%)")
        print(f"  Val Sequences:              {len(X_val):,} ({metadata['splits']['val']['sequence_pct']}%)")
        print(f"  Test Sequences:             {len(X_test):,} ({metadata['splits']['test']['sequence_pct']}%)")
        print(f"  Data Leakage Status:        {verification['status']} (Zero Overlap)")
        print("=" * 72 + "\n")

        return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate temporal sequences for hybrid traffic models.")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH, help="Length of temporal sequence window (default: 20)")
    parser.add_argument("--stride", type=int, default=SEQUENCE_STRIDE, help="Stride step size for rolling window (default: 1)")
    parser.add_argument("--data-file", type=str, default=PROCESSED_FEATURES_FILE, help="Path to processed features CSV")
    parser.add_argument("--output-dir", type=str, default=SEQUENCES_DIR, help="Output directory for sequence NPZ files")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO, help="Train track split ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation track split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="Test track split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for track splitting (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    builder = SequenceBuilder(
        sequence_length=args.sequence_length,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    builder.run_pipeline(data_path=args.data_file, output_dir=args.output_dir)
