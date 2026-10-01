# HYBRID TRAFFIC INTELLIGENCE TEMPORAL SEQUENCE REPORT
**Project:** Intelligent Traffic Management System (Emerge)  
**Document:** `HYBRID_SEQUENCE_REPORT.md`  
**Pipeline Phase:** Phase 3 — Temporal Sequence Generation & Leakage-Free Partitioning  
**Execution Date:** September 2026  
**Status:** Validated & Completed (Zero-Model Training Phase)  

---

## 1. Executive Summary & Governance Compliance

This report documents the implementation, empirical execution, and rigorous validation of **Phase 3: Temporal Sequence Generation** within the Hybrid Traffic Intelligence framework. Operating on top of the Phase 2 feature dataset (`data/hybrid_traffic/processed_features.csv`), the sequence builder module transforms frame-by-frame vehicle tracking observations into structured 3D temporal tensors for downstream deep learning and sequential machine learning models.

### Strict Governance Compliance:
* **Zero Disruption Rule:** The existing computer vision perception layer (YOLOv8 + ByteTrack), Flask REST/streaming server (`app.py`), analytical engines, SUMO simulation files, and frontend command center remain **100% untouched and unmodified**.
* **Zero Model Training Rule:** Consistent with instructions, **no machine learning models were trained**. All outputs are confined to preprocessed temporal sequence datasets, structured metadata, and audit artifacts.
* **Strict Track-Level Isolation:** Observations from different vehicle tracks are **never combined**. Every sequence represents a continuous temporal segment from one and only one vehicle.
* **Zero Data Leakage:** Partitioning is performed strictly at the **`track_id` level** prior to sequence extraction. No vehicle track appears in more than one partition.
* **No Silent Data Loss:** Short tracks unable to satisfy the window requirement are cataloged, classified, and fully accounted for in metadata and reports.

---

## 2. Temporal Sequence Generation Parameters

The sequence builder module ([`sequence_builder.py`](file:///c:/Users/ayush/Emerge%20Root00/ml/hybrid_traffic/sequence_builder.py)) implements a configurable sliding-window generator:

| Parameter | Default Value | Configurable | Operational Description |
| :--- | :---: | :---: | :--- |
| **`sequence_length` ($L$)** | **20 timesteps** | Yes (`--sequence-length`) | Length of temporal observation history (~0.667 seconds at 30 FPS). |
| **`stride` ($S$)** | **1 timestep** | Yes (`--stride`) | Step size between consecutive rolling windows. Maximum temporal sample density. |
| **Input Source** | `processed_features.csv` | Yes (`--data-file`) | 29,139 engineered observations across 697 usable tracks from Phase 2. |
| **Output Directory** | `data/hybrid_traffic/sequences/` | Yes (`--output-dir`) | Compressed NumPy archives (`.npz`) and structured telemetry JSON. |
| **Feature Dimension ($D$)** | **20 features** | Yes (Custom list / Config) | Curated spatial, kinematic, heading, and variance metrics. |
| **Splitting Strategy** | **Track-Level Stratified** | Fixed | Dominant vehicle class stratification; 70% Train, 15% Val, 15% Test. |
| **Random Seed** | **42** | Yes (`--seed`) | Deterministic reproducibility. |

### Rolling Window Mechanics (Single Track ID Example):
For a vehicle with ID `101` having $M$ sequential observations:
```
Track 101 Observation 1  (t = 0)
Track 101 Observation 2  (t = 1)
...
Track 101 Observation 20 (t = 19)
───────────────────────────────────► Sequence 0: Shape (20, 20)

Track 101 Observation 2  (t = 1)
Track 101 Observation 3  (t = 2)
...
Track 101 Observation 21 (t = 20)
───────────────────────────────────► Sequence 1: Shape (20, 20)
```
* If $M < 20$: The track cannot form a full unpadded sequence; it is explicitly audited as a **discarded short track**.
* If $M = 20$: Exactly **1 sequence** is generated.
* If $M > 20$: Exactly $M - 20 + 1$ rolling sequences are generated (with stride $S=1$).

---

## 3. High-Level Telemetry & Dataset Summary

| Metric | Count / Value | Proportion (%) | Reference Context |
| :--- | :---: | :---: | :--- |
| **Total Raw Input Tracks (`traffic_tracks.csv`)** | **1,754** | — | Raw ByteTrack tracking output |
| **Total Tracks in Feature Dataset (`processed_features.csv`)** | **697** | 100.0% | Tracks with $\ge 10$ observations from Phase 2 |
| **Usable Tracks ($\text{Length} \ge 20$)** | **320** | **45.91%** | Tracks meeting the 20-frame sequence requirement |
| **Discarded Tracks ($\text{Length} < 20$)** | **377** | **54.09%** | Short tracks ($10 \le \text{length} \le 19$) |
| **Usable Observations in Sequences** | **23,881** | **81.95%** | Retained from 29,139 usable Phase 2 observations |
| **Discarded Observations** | **5,258** | **18.05%** | Belonging to the 377 discarded tracks |
| **Total Temporal Sequences Generated ($N$)** | **17,801** | **100.0%** | Full dataset sequence count |
| **Sequence Tensor Dimensions ($X$)** | **`(17801, 20, 20)`** | — | `(N_samples, sequence_length, feature_count)` |
| **Feature Count ($D$)** | **20** | — | Continuous kinematic & spatial variables |
| **Sequence Length ($L$)** | **20** | — | ~0.67s temporal depth at 30 FPS |
| **Storage Footprint** | **3.22 MB** | — | Total compressed size for all 4 `.npz` files |

---

## 4. Short-Track Accounting & Discard Audit

To ensure **no data is silently discarded**, the sequence pipeline performs an exhaustive accounting of all 377 tracks shorter than 20 observations:

### Why Short Tracks Are Discarded:
A vehicle track with fewer than 20 observations ($< 0.667\text{s}$) cannot form a complete sliding window without either:
1. **Artificial zero-padding or edge-replication**, which introduces false stationary biases, velocity discontinuities, or synthetic acceleration spikes; or
2. **Concatenating observations across different vehicles**, which strictly violates the fundamental physics constraint: *a sequence must never combine observations from different track IDs*.

Consequently, unpadded rolling windows require $\text{length} \ge 20$. However, rather than silently dropping these tracks, their lengths, classes, and IDs are persisted in [`sequence_metadata.json`](file:///c:/Users/ayush/Emerge%20Root00/data/hybrid_traffic/sequences/sequence_metadata.json).

### Discarded Track Length Distribution:
| Track Length (frames) | Duration (seconds) | Discarded Track Count | Total Observations |
| :---: | :---: | :---: | :---: |
| **10** | 0.333s | 24 | 240 |
| **11** | 0.367s | 33 | 363 |
| **12** | 0.400s | 117 | 1,404 |
| **13** | 0.433s | 83 | 1,079 |
| **14** | 0.467s | 5 | 70 |
| **15** | 0.500s | 5 | 75 |
| **16** | 0.533s | 6 | 96 |
| **17** | 0.567s | 6 | 102 |
| **18** | 0.600s | 33 | 594 |
| **19** | 0.633s | 65 | 1,235 |
| **Total** | — | **377** | **5,258** |

### Vehicle Class Distribution of Discarded Tracks:
* **Motorcycles (Class 3):** 216 tracks (57.29%) — Fast-moving two-wheelers traversing camera boundaries rapidly or briefly occluded by larger vehicles.
* **Cars (Class 2):** 95 tracks (25.20%) — Fast boundary exits or short corner cut-throughs.
* **Trucks (Class 7):** 44 tracks (11.67%) — Boundary detections near camera edge.
* **Buses (Class 5):** 22 tracks (5.84%) — Distant inflow entries.

---

## 5. Sequences-per-Track Distribution Statistics

Because vehicle tracks vary in duration depending on vehicle speed, congestion state, and turning maneuvers, the number of sequences generated per track exhibits a characteristic long-tailed urban intersection distribution:

| Statistic | Value (Sequences / Track) | Physical Interpretation |
| :--- | :---: | :--- |
| **Minimum** | **1.0** | Vehicle track of exactly 20 frames (observed for 0.667s). |
| **10th Percentile** | **5.0** | 24 frames of tracking history. |
| **25th Percentile ($Q_1$)** | **7.0** | 26 frames of tracking history. |
| **Median ($50\%$, $Q_2$)** | **19.0** | 38 frames of tracking history (~1.27s). |
| **Mean $\pm$ Std** | **$55.63 \pm 114.87$** | Mean track yields ~56 sliding windows. |
| **75th Percentile ($Q_3$)** | **55.25** | 74 frames of tracking history (~2.47s). |
| **90th Percentile** | **136.0** | 155 frames of tracking history (~5.17s). |
| **95th Percentile** | **206.05** | 225 frames of tracking history (~7.50s). |
| **Maximum** | **1,082.0** | Durable queueing vehicle observed for 1,101 frames (~36.7s). |

---

## 6. Data Leakage Prevention & Partition Verification

### The Data Leakage Threat:
If temporal sequences are randomly split into training and testing sets, consecutive windows from the same vehicle (e.g. frames $1..20$ and frames $2..21$) would be partitioned across train and test sets. Because frame $t$ and frame $t+1$ share 95% identical observations, a model would simply memorize vehicle identity and interpolate adjacent frames, inflating evaluation metrics to near 100% while failing completely on unseen real-world vehicles.

### The Enforcement Protocol:
1. All 320 usable tracks were evaluated for their **dominant vehicle class**.
2. Partitioning was performed strictly at the **`track_id` level** using class-stratified sampling:
   * **Train Split:** 70.0% of tracks ($N=224$)
   * **Validation Split:** 15.0% of tracks ($N=48$)
   * **Test Split:** 15.0% of tracks ($N=48$)
3. Sequences were subsequently generated **independently within each partition**.
4. Every sequence from a given vehicle belongs exclusively to one partition.

### Mathematical Verification:
Programmatic set-intersection verification was executed on the generated partitions:

```
========================================================================
DATA LEAKAGE PREVENTION & PARTITION VERIFICATION
========================================================================
Total Usable Tracks: 320
  - Train Partition: 224 tracks (70.0%)
  - Val Partition:   48 tracks (15.0%)
  - Test Partition:  48 tracks (15.0%)
------------------------------------------------------------------------
Check 1: train_track_ids ∩ validation_track_ids = empty
         -> Result: PASS (Empty) [count: 0]
Check 2: train_track_ids ∩ test_track_ids = empty
         -> Result: PASS (Empty) [count: 0]
Check 3: validation_track_ids ∩ test_track_ids = empty
         -> Result: PASS (Empty) [count: 0]
------------------------------------------------------------------------
OVERALL VERIFICATION: PASS — Zero data leakage across partitions.
========================================================================
```

**Verification Status:** **PASS** — Exactly zero overlapping track IDs exist between Train, Validation, and Test sets.

---

## 7. Dataset Partitions Breakdown

| Partition | Unique Tracks | Track Share | Sequences Generated | Sequence Share | Tensor Shape ($X$) | Target Shape ($y$) | Compressed File | Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Train Set** | **224** | **70.0%** | **12,414** | **69.74%** | `(12414, 20, 20)` | `(12414, 2)` | `train_sequences.npz` | 1.13 MB |
| **Validation Set** | **48** | **15.0%** | **2,790** | **15.67%** | `(2790, 20, 20)` | `(2790, 2)` | `val_sequences.npz` | 0.25 MB |
| **Test Set** | **48** | **15.0%** | **2,597** | **14.59%** | `(2597, 20, 20)` | `(2597, 2)` | `test_sequences.npz` | 0.24 MB |
| **Full Usable Set** | **320** | **100.0%** | **17,801** | **100.0%** | `(17801, 20, 20)` | `(17801, 2)` | `all_sequences.npz` | 1.60 MB |

### Fleet Class Representation in Sequence Partitions:
| Vehicle Class | YOLO COCO ID | Train Sequences | Train Share | Val Sequences | Val Share | Test Sequences | Test Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Car** | `2` | 5,423 | 43.68% | 866 | 31.04% | 1,283 | 49.40% |
| **Motorcycle** | `3` | 3,442 | 27.73% | 438 | 15.70% | 858 | 33.04% |
| **Truck** | `7` | 2,465 | 19.86% | 1,181 | 42.33% | 105 | 4.04% |
| **Bus** | `5` | 1,084 | 8.73% | 305 | 10.93% | 351 | 13.52% |
| **Total** | — | **12,414** | **100.0%** | **2,790** | **100.0%** | **2,597** | **100.0%** |

---

## 8. Engineered Sequence Feature Inventory (20 Dimensions)

Each timestep in the sequence contains 20 continuous numerical variables:

| Index | Feature Column Name | Data Type | Units / Range | Functional Description |
| :---: | :--- | :---: | :---: | :--- |
| **`0`** | `center_x` | `float32` | $[0, 1920]\text{ px}$ | Image-space horizontal centroid coordinate. |
| **`1`** | `center_y` | `float32` | $[0, 1080]\text{ px}$ | Image-space vertical centroid coordinate. |
| **`2`** | `delta_x` | `float32` | Pixels | Centroid horizontal displacement from preceding frame: $x_t - x_{t-1}$. |
| **`3`** | `delta_y` | `float32` | Pixels | Centroid vertical displacement from preceding frame: $y_t - y_{t-1}$. |
| **`4`** | `displacement_image` | `float32` | Pixels | Euclidean step distance on image canvas: $\sqrt{\Delta x^2 + \Delta y^2}$. |
| **`5`** | `speed_image_px_per_sec`| `float32` | $\text{px}/\text{s}$ | Instantaneous image-plane velocity: $\text{displacement} / \Delta t$. |
| **`6`** | `accel_image_px_per_sec2`| `float32` | $\text{px}/\text{s}^2$ | Instantaneous acceleration: $\Delta \text{speed} / \Delta t$. |
| **`7`** | `heading_rad` | `float32` | $[-\pi, \pi]\text{ rad}$ | Direction angle of movement vector: $\text{atan2}(\Delta y, \Delta x)$. |
| **`8`** | `heading_change_rad` | `float32` | $[-\pi, \pi]\text{ rad}$ | Angular deviation between consecutive steps (turning rate). |
| **`9`** | `ground_x` | `float32` | $[0, 1200]$ | Bird's-eye horizontal coordinate projected via homography $H$. |
| **`10`**| `ground_y` | `float32` | $[0, 700]$ | Bird's-eye vertical coordinate projected via homography $H$. |
| **`11`**| `ground_delta_x` | `float32` | Ground units | Projected horizontal displacement on bird's-eye canvas. |
| **`12`**| `ground_delta_y` | `float32` | Ground units | Projected vertical displacement on bird's-eye canvas. |
| **`13`**| `displacement_ground` | `float32` | Ground units | Projected Euclidean step distance on ground plane. |
| **`14`**| `ground_speed_norm_per_sec`| `float32` | $\text{units}/\text{s}$ | Normalized velocity on bird's-eye canvas. |
| **`15`**| `box_width` | `float32` | Pixels | Bounding box pixel width ($x_2 - x_1$). |
| **`16`**| `box_height` | `float32` | Pixels | Bounding box pixel height ($y_2 - y_1$). |
| **`17`**| `box_area` | `float32` | $\text{px}^2$ | Bounding box area ($\text{width} \times \text{height}$); proxy for vehicle scale. |
| **`18`**| `stopped_duration` | `float32` | Seconds | Continuous duration vehicle has remained stationary ($< 5\text{ px/s}$). |
| **`19`**| `speed_variance_rolling5`| `float32` | $(\text{px}/\text{s})^2$ | 5-step rolling sample variance of image-space velocity. |

---

## 9. Storage Layout & Array Specifications

All generated sequence datasets are persisted under `data/hybrid_traffic/sequences/`:

```
data/hybrid_traffic/sequences/
├── all_sequences.npz          # 17,801 sequences (1.60 MB)
├── train_sequences.npz        # 12,414 sequences (1.13 MB)
├── val_sequences.npz          #  2,790 sequences (0.25 MB)
├── test_sequences.npz         #  2,597 sequences (0.24 MB)
└── sequence_metadata.json     # Complete structured telemetry & audit (3.3 KB)
```

### Internal NPZ Archive Array Schema:
Each `.npz` archive contains 8 synchronized arrays:

1. **`X` (`float32`, shape `[N, 20, 20]`):** The temporal sequence tensor containing 20 timesteps of 20 kinematic features.
2. **`y_next_disp` (`float32`, shape `[N, 2]`):** The target horizontal and vertical displacement $(\Delta x, \Delta y)$ from the last frame of the window ($t=19$) to the next frame ($t=20$). Stored as `[np.nan, np.nan]` if the window terminates at the end of the vehicle's lifespan.
3. **`has_next` (`bool`, shape `[N]`):** Boolean flag indicating whether the vehicle was observed in the subsequent frame (`True` for 98.2% of sequences; `False` only for the terminal sequence of each track).
4. **`track_ids` (`int64`, shape `[N]`):** Persistent ByteTrack vehicle identifier for each sequence.
5. **`classes` (`int64`, shape `[N]`):** Vehicle class integer (`2`: car, `3`: motorcycle, `5`: bus, `7`: truck).
6. **`start_frames` (`int64`, shape `[N]`):** Video frame index corresponding to timestep $t=0$ of the window.
7. **`end_frames` (`int64`, shape `[N]`):** Video frame index corresponding to timestep $t=19$ of the window.
8. **`feature_names` (`<U30`, shape `[20]`):** Array of string feature names in column order.

### Universal Compatibility:
Because `.npz` is a zero-pickle, standardized format:
* **PyTorch:** Readily loaded via `torch.from_numpy(np.load('train_sequences.npz')['X'])`.
* **TensorFlow / Keras:** Loaded directly via `tf.convert_to_tensor(np.load(...))`.
* **Scikit-Learn:** Usable via `X.reshape(N, -1)` or the companion lagged matrix generator [`build_tabular_lagged_dataset()`](file:///c:/Users/ayush/Emerge%20Root00/ml/hybrid_traffic/sequence_builder.py).

---

## 10. CLI Usage Reference

The sequence builder module can be executed directly from the terminal or invoked programmatically:

```bash
# Default execution (sequence_length=20, stride=1, seed=42)
python ml/hybrid_traffic/sequence_builder.py

# Custom configuration (e.g. sequence_length=30, stride=2)
python ml/hybrid_traffic/sequence_builder.py --sequence-length 30 --stride 2

# Custom data paths and split ratios
python ml/hybrid_traffic/sequence_builder.py \
    --data-file data/hybrid_traffic/processed_features.csv \
    --output-dir data/hybrid_traffic/sequences \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
```

---

## 11. Conclusion & Phase 4 Readiness

Phase 3 is complete and verified:
1. **17,801 temporal sequences** across 320 usable tracks were generated with strict single-track containment.
2. **Zero data leakage** was achieved and mathematically confirmed ($\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$).
3. **Short tracks (377 tracks, 5,258 observations)** were thoroughly audited and cataloged with zero silent data loss.
4. **All production code and applications remain 100% operational and undisturbed.**
5. **No machine learning models were trained.**

**The temporal sequence dataset is fully persisted and ready for Phase 4 model architecture design and training.**
