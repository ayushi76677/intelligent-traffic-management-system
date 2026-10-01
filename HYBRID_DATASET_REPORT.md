# HYBRID TRAFFIC INTELLIGENCE DATASET PREPROCESSING REPORT
**Project:** Intelligent Traffic Management System (Emerge)  
**Document:** `HYBRID_DATASET_REPORT.md`  
**Pipeline Phase:** Phase 2 — Machine Learning Feature Pipeline & Dataset Preparation  
**Execution Date:** September 2026  
**Status:** Validated & Completed (Zero-Model Training Phase)  

---

## 1. Executive Overview

This report details the execution and empirical validation of the **Phase 2 Hybrid Traffic Intelligence Dataset Pipeline**. The pipeline operates on top of pre-existing ByteTrack multi-object tracking data (`traffic_tracks.csv`) and produces an enriched, clean, leak-free feature dataset for subsequent machine learning model development.

### Strict Governance Compliance:
* **Zero Disruption:** The production computer vision models, video feeds, calibration arrays, SUMO simulation files, Flask backend, and frontend dashboard remain **100% untouched**.
* **Zero In-Place Overwrites:** `traffic_tracks.csv` was verified as completely unmodified.
* **No Premature Model Training:** No machine learning models were trained.
* **Metric Realism:** Velocities are rigorously labeled as image-space pixels per second (`px/s`) and normalized ground-plane units per second (`norm/s`). **No ungrounded metric km/h claims were fabricated.**

---

## 2. Dataset Ingestion & Observation Counts

| Dataset Stage | Observations | Unique Tracks | Storage Location |
| :--- | :---: | :---: | :--- |
| **Raw Tracking Input (`traffic_tracks.csv`)** | **34,679** | **1,754** | `c:/Users/ayush/Emerge Root00/traffic_tracks.csv` |
| **Filtered Usable Dataset (`processed_features.csv`)** | **29,139** | **697** | `data/hybrid_traffic/processed_features.csv` |
| **Discarded Short Tracks ($< 10$ frames)** | **5,540** | **1,057** | Filtered during ingestion |
| **Training Partition (`train_data.csv`)** | **19,825** | **487** | `data/hybrid_traffic/train_data.csv` |
| **Validation Partition (`val_data.csv`)** | **4,484** | **105** | `data/hybrid_traffic/val_data.csv` |
| **Test Partition (`test_data.csv`)** | **4,830** | **105** | `data/hybrid_traffic/test_data.csv` |

* **Usable Observation Retention:** 84.02% of all raw observations belong to tracks with $\ge 10$ observations.
* **Discarded Tracks Justification:** Tracks shorter than 10 frames ($< 0.33$ seconds) predominantly represent boundary detection jitter, transient occlusions, or brief tracker re-identifications. They lack sufficient historical depth to calculate moving variances, curvature, or multi-step temporal sequences.

---

## 3. Track-Length Distribution

The distribution of track lengths demonstrates a long-tailed urban intersection profile, with high-frequency short-lived tracking segments and durable trajectories for turning and queueing vehicles:

### A. Raw Tracking Set ($N = 1,754$ tracks)
* **Minimum:** 1 frame (0.033s)
* **5th Percentile:** 3.0 frames
* **10th Percentile:** 4.3 frames
* **25th Percentile ($Q_1$):** 5.0 frames
* **Median ($50\%$, $Q_2$):** 6.0 frames
* **Mean $\pm$ Std:** $19.77 \pm 55.57$ frames
* **75th Percentile ($Q_3$):** 14.0 frames
* **90th Percentile:** 33.0 frames
* **95th Percentile:** 68.4 frames
* **99th Percentile:** 222.41 frames
* **Maximum:** 1,101 frames (36.70s)

### B. Usable Filtered Set ($\text{Track Length} \ge 10$, $N = 697$ tracks)
* **Minimum:** 10 frames (0.333s)
* **5th Percentile:** 11.0 frames
* **10th Percentile:** 12.0 frames
* **25th Percentile ($Q_1$):** 13.0 frames
* **Median ($50\%$, $Q_2$):** 19.0 frames
* **Mean $\pm$ Std:** $41.81 \pm 83.48$ frames
* **75th Percentile ($Q_3$):** 34.0 frames
* **90th Percentile:** 80.0 frames
* **95th Percentile:** 143.4 frames
* **99th Percentile:** 384.52 frames
* **Maximum:** 1,101 frames (36.70s)

---

## 4. Vehicle Class Distribution

Class IDs originate strictly from the YOLOv8 COCO pre-trained model mapping `{2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}`:

| Class ID | Vehicle Category | Raw Observations | Raw Share (%) | Usable Observations | Usable Share (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **`2`** | **Car** | 13,596 | 39.21% | 12,505 | **42.91%** |
| **`3`** | **Motorcycle** | 12,609 | 36.36% | 9,014 | **30.93%** |
| **`7`** | **Truck** | 5,274 | 15.21% | 4,756 | **16.32%** |
| **`5`** | **Bus** | 3,200 | 9.23% | 2,864 | **9.83%** |
| **Total** | | **34,679** | **100.0%** | **29,139** | **100.0%** |

* **Fleet Observation:** Cars and motorcycles constitute $\approx 73.8\%$ of usable intersection traffic, reflecting typical mixed Indian urban traffic conditions.

---

## 5. Spatial Zone & Lane Assignment Statistics

Spatial zones were evaluated by testing vehicle tire-road contact points $( (x_1+x_2)/2, y_2 )$ against the 6 polygon definitions loaded dynamically from `lane_zone_points.npy` and `zone_2_points.npy` through `zone_6_points.npy`:

| Zone ID | Corridor / Zone Role | Lane Designation (`lane_id`) | Usable Observations | Proportion (%) |
| :--- | :--- | :--- | :---: | :---: |
| **`ZONE_1`** | Main Approach Corridor (AB Road North Inflow) | `LANE_NORTH_INFLOW` | 17,936 | 61.55% |
| **`UNKNOWN`** | Perimeter / Unzoned Roadways | `UNKNOWN` | 5,591 | 19.19% |
| **`ZONE_2`** | Eastbound Turning / Exit Lane | `LANE_EAST_TURNING` | 3,078 | 10.56% |
| **`ZONE_5`** | Westbound Inflow Approach | `LANE_WEST_INFLOW` | 1,347 | 4.62% |
| **`ZONE_3`** | Central Intersection Junction Box (Crossing) | `UNKNOWN` (Intersection Core) | 684 | 2.35% |
| **`ZONE_4`** | Southbound Queue Inflow Area | `LANE_SOUTH_QUEUE` | 342 | 1.17% |
| **`ZONE_6`** | Southeast Exit Corridor | `LANE_SE_OUTFLOW` | 161 | 0.55% |
| **Total** | | | **29,139** | **100.0%** |

* **Lane Assignment Rationality:** Zone 3 is a multi-directional intersection crossing box rather than a single dedicated lane; consequently, its `lane_id` is formally marked as `UNKNOWN` to avoid false lane attribution.

---

## 6. Engineered Features Inventory (46 Columns)

The feature engineering pipeline computes **46 feature columns** structured across 7 functional categories:

### A. Core Identifiers & Temporal Anchors
1. `frame` (`int64`): Frame index ($0 \le \text{frame} \le 1529$).
2. `time` (`float64`): Video timestamp in seconds ($0.0 \le t \le 50.967$).
3. `track_id` (`int64`): ByteTrack persistent vehicle tracking identifier.
4. `class` (`int64`): Numerical vehicle class ($2, 3, 5, 7$).
5. `vehicle_type` (`object`): String vehicle name (`"car"`, `"motorcycle"`, `"bus"`, `"truck"`).
6. `frame_delta` (`int64`): Discrete step difference from prior observation ($\Delta \text{frame} = 1$ for continuous tracks; $0$ at track start).
7. `time_delta` (`float64`): Elapsed time in seconds from prior observation ($\Delta t \approx 0.033\text{s}$ at 30 FPS).

### B. Image-Space Geometry & Bounding Box Properties
8. `x1` (`float64`): Bounding box top-left $x$ coordinate in pixels.
9. `y1` (`float64`): Bounding box top-left $y$ coordinate in pixels.
10. `x2` (`float64`): Bounding box bottom-right $x$ coordinate in pixels.
11. `y2` (`float64`): Bounding box bottom-right $y$ coordinate in pixels.
12. `center_x` (`float64`): Bounding box centroid $x$: $(x_1 + x_2) / 2.0$.
13. `center_y` (`float64`): Bounding box centroid $y$: $(y_1 + y_2) / 2.0$.
14. `box_width` (`float64`): Box width: $x_2 - x_1$.
15. `box_height` (`float64`): Box height: $y_2 - y_1$.
16. `box_area` (`float64`): Box area in square pixels: $\text{width} \times \text{height}$.
17. `aspect_ratio` (`float64`): Box elongation: $\text{width} / \text{height}$.
18. `bottom_center_x` (`float64`): Ground contact patch $x$ coordinate.
19. `bottom_center_y` (`float64`): Ground contact patch $y$ coordinate ($y_2$).

### C. Image-Space Kinematics & Angular Dynamics
20. `delta_x` (`float64`): Centroid horizontal displacement: $x_t - x_{t-1}$.
21. `delta_y` (`float64`): Centroid vertical displacement: $y_t - y_{t-1}$.
22. `displacement_image` (`float64`): Euclidean step distance: $\sqrt{\Delta x^2 + \Delta y^2}$ in pixels.
23. `speed_image_px_per_sec` (`float64`): Instantaneous image velocity: $\text{displacement} / \Delta t$ in pixels/second.
24. `accel_image_px_per_sec2` (`float64`): Instantaneous image acceleration: $\Delta \text{speed} / \Delta t$ in $\text{px}/\text{s}^2$.
25. `heading_rad` (`float64`): Motion vector direction angle in radians: $\text{atan2}(\Delta y, \Delta x) \in [-\pi, \pi]$.
26. `heading_deg` (`float64`): Motion vector direction angle in degrees: $[0, 360^\circ)$.
27. `direction` (`object`): 8-way compass category (`"EAST"`, `"SOUTHEAST"`, `"SOUTH"`, `"SOUTHWEST"`, `"WEST"`, `"NORTHWEST"`, `"NORTH"`, `"NORTHEAST"`, or `"STATIONARY"`).
28. `heading_change_rad` (`float64`): Angular deviation between consecutive steps wrapped to $[-\pi, \pi]$.
29. `heading_change_deg` (`float64`): Angular deviation in degrees $[-180^\circ, 180^\circ]$.

### D. Perspective & Ground-Plane Projection (Bird's-Eye View)
30. `ground_x` (`float64`): Projected horizontal coordinate on the $1200 \times 700$ bird's-eye canvas via $H$.
31. `ground_y` (`float64`): Projected vertical coordinate on the $1200 \times 700$ bird's-eye canvas via $H$.
32. `ground_delta_x` (`float64`): Ground-plane horizontal displacement: $\text{ground\_x}_t - \text{ground\_x}_{t-1}$.
33. `ground_delta_y` (`float64`): Ground-plane vertical displacement: $\text{ground\_y}_t - \text{ground\_y}_{t-1}$.
34. `displacement_ground` (`float64`): Ground-plane Euclidean step distance.
35. `ground_speed_norm_per_sec` (`float64`): Normalized ground velocity: $\text{displacement\_ground} / \Delta t$ (units/sec).

### E. Spatial Zoning & Infrastructure Roles
36. `zone_id` (`object`): Spatial containment polygon (`"ZONE_1"` through `"ZONE_6"` or `"UNKNOWN"`).
37. `lane_id` (`object`): Functional corridor role (`"LANE_NORTH_INFLOW"`, `"LANE_EAST_TURNING"`, etc. or `"UNKNOWN"`).

### F. Cumulative & Track-Level Historical Metrics
38. `distance_travelled` (`float64`): Cumulative image-plane distance travelled from vehicle entry up to current frame.
39. `trajectory_length` (`float64`): Total aggregate image-plane distance travelled across the vehicle's entire lifetime.
40. `track_duration` (`float64`): Total lifetime of the vehicle track in seconds ($t_{\text{last}} - t_{\text{first}}$).
41. `elapsed_track_time` (`float64`): Seconds elapsed since vehicle first entered the camera view ($t - t_{\text{first}}$).
42. `track_total_observations` (`int64`): Total observation count for this vehicle track.

### G. Stationary & Dispersion Characteristics
43. `is_stopped` (`bool`): Indicator flag (`True` if `speed_image_px_per_sec` $< 5.0\text{ px/s}$).
44. `stopped_duration` (`float64`): Continuous consecutive seconds vehicle has remained in the stopped state.
45. `speed_variance_rolling5` (`float64`): 5-step rolling sample variance of image-space speed.
46. `displacement_variance_rolling5` (`float64`): 5-step rolling sample variance of image-space displacement.

---

## 7. Data Quality & Integrity Validation

A comprehensive programmatic audit of all output partitions was performed:

| Check | Target / Rule | Result | Verification Status |
| :--- | :--- | :---: | :---: |
| **Missing Values (NaNs)** | 0 across all 46 columns in all files | **0** | **PASS** |
| **Infinite Values ($\pm \infty$)** | 0 across all numerical columns | **0** | **PASS** |
| **Raw File Preservation** | `traffic_tracks.csv` must match 34,679 rows $\times$ 10 cols | **Unchanged** | **PASS** |
| **Track-Level Partition Leakage** | $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$ | **0 overlap** | **PASS** |
| **Index Alignment** | Observations monotonic within each `track_id` | **Strictly ordered** | **PASS** |

---

## 8. Track-Aware Stratified Dataset Splitting

To ensure machine learning models evaluate generalizable behavior rather than memorizing vehicle IDs, partitioning was executed strictly at the **`track_id` level** (preventing data leakage across adjacent frames), stratified across vehicle classes:

```
Total Usable Tracks: 697 (29,139 observations)
      │
      ├──► Train Split (69.87% of tracks): 487 tracks, 19,825 observations
      │     - Cars: 8,495 | Motorcycles: 6,105 | Trucks: 3,246 | Buses: 1,979
      │
      ├──► Validation Split (15.06% of tracks): 105 tracks, 4,484 observations
      │     - Cars: 1,940 | Motorcycles: 1,440 | Trucks: 698 | Buses: 406
      │
      └──► Test Split (15.06% of tracks): 105 tracks, 4,830 observations
            - Cars: 2,070 | Motorcycles: 1,469 | Trucks: 812 | Buses: 479
```

* **Zero Leakage Confirmation:** Set intersection analysis verified zero common track IDs between Train, Validation, and Test sets.

---

## 9. Coordinate Systems & Calibration Declaration

1. **Image Coordinate System:**
   * **Dimensions:** $1920 \times 1080$ pixels.
   * **Origin:** Top-left corner $(0, 0)$; $+X$ extends rightward, $+Y$ extends downward.
   * **Sensor Context:** Elevated angled infrastructure camera observing an urban cross-corridor.
2. **Ground-Plane Projective System:**
   * **Dimensions:** $1200 \times 700$ bird's-eye canvas.
   * **Transformation:** Projective homography $H$ (`perspective_matrix.npy`) derived from 4 road boundary calibration points (`calibration_points.npy`).
3. **Real-World Metric Calibration:**
   * **Declaration:** **NOT USED / NOT AVAILABLE**.
   * **Rationale:** Physical surveying of road distances (e.g. laser distance measurement of lane widths or stop-line distances) was not performed on this camera feed.
   * **Standard Enforced:** Velocities are explicitly labeled as `speed_image_px_per_sec` (pixels/second) and `ground_speed_norm_per_sec` (normalized units/second). **Under no circumstances are these values converted to uncalibrated km/h.**

---

## 10. Temporal Sequences & Tabular Lagged Formats

The companion module [`sequence_builder.py`](file:///c:/Users/ayush/Emerge%20Root00/ml/hybrid_traffic/sequence_builder.py) was tested against `processed_features.csv` to ensure compatibility with diverse model families:

1. **3D Sequence Tensors (RNN / LSTM / Transformer):**
   * Output shape: `(19,893, 10, 17)` representing 19,893 temporal sliding windows of length $L=10$ frames ($0.33\text{s}$) across 17 kinematic features.
   * Target array: `(19,893, 2)` representing future displacement $(\Delta x_{\text{fut}}, \Delta y_{\text{fut}})$ over horizon $H=5$ frames ($0.16\text{s}$).
2. **2D Tabular Lagged Matrices (`scikit-learn` Random Forest / Gradient Boosting / SVR):**
   * Produces 176 flattened feature columns with standardized lag prefixes (`feat_lag9`, ..., `feat_lag0`) and continuous displacement/speed targets.

---

## 11. Pipeline Module Architecture

The completed Phase 2 pipeline is packaged cleanly under `ml/hybrid_traffic/`:

```
ml/
└── hybrid_traffic/
    ├── __init__.py               # Public API exports
    ├── config.py                 # Hyperparameters, paths, and thresholds
    ├── feature_engineering.py    # Spatial, kinematic, and zone calculations
    ├── sequence_builder.py       # 3D tensor and 2D tabular window generators
    ├── split_dataset.py          # Track-stratified leakage-free splitting
    └── dataset.py                # High-level pipeline orchestrator
```

Generated data artifacts are persisted under `data/hybrid_traffic/`:
* `data/hybrid_traffic/processed_features.csv` (29,139 rows $\times$ 46 columns)
* `data/hybrid_traffic/train_data.csv` (19,825 rows $\times$ 46 columns)
* `data/hybrid_traffic/val_data.csv` (4,484 rows $\times$ 46 columns)
* `data/hybrid_traffic/test_data.csv` (4,830 rows $\times$ 46 columns)
* `data/hybrid_traffic/dataset_metadata.json` (Structured operational telemetry)

---

## 12. Conclusion & Readiness

The Phase 2 training-data pipeline has been executed, mathematically verified, and fully persisted without disturbing any pre-existing code. Zero errors, zero missing values, and zero data leakage were detected.

**Phase 2 is Complete. Awaiting authorization before proceeding to Phase 3 model architecture selection and training.**
