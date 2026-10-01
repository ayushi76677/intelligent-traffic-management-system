# Phase 6: Real-Time Hybrid Model Integration — Architecture & Pipeline Audit

**Project:** Smart Traffic Management System (Emerge)  
**Phase:** Phase 6 — Real-Time Hybrid Model Integration  
**Date:** September 2026  
**Status:** Audit Completed  

---

## 1. Executive Summary

This audit assesses the entire existing Smart Traffic Management System pipeline to establish the exact integration points for the trained **TCN-Transformer Gated Hybrid Architecture**. The trained model operates over a sliding temporal window of 20 timesteps with 20 kinematic, spatial, geometric, and ground-plane features to simultaneously predict 9 multi-task traffic intelligence targets plus 1 auxiliary trajectory displacement head.

The audit verifies that:
1. No existing verified functionality (YOLO detection, ByteTrack tracking, deterministic congestion scoring, violation rule engine, SUMO simulation, Authority Mode, or Driver Mode) is duplicated or removed.
2. The hybrid model acts as an **analytical intelligence enhancement layer** alongside deterministic calculations rather than replacing verified physical measurements.
3. Live track observations can be buffered into per-track temporal sequences, scaled with the training scaler (`checkpoints/feature_scaler.joblib`), and passed through the model (`checkpoints/best_model.pth`) with zero data distribution drift.

---

## 2. Component-by-Component System Audit

### 2.1 Video Input & Sensor Parameters
- **Source Video:** `traffic2.mp4` (1920×1080 resolution, 30.0 FPS, 1,530 frames total, 51.0 seconds duration).
- **Coordinate Space:** Image pixel coordinates $(x, y) \in [0, 1920] \times [0, 1080]$, where $+X$ is eastward/rightward and $+Y$ is downward.
- **Physical Calibration:** Perspective homography matrix stored in `perspective_matrix.npy` (derived from 4 ground calibration points in `calibration_points.npy`) transforms vehicle tire-road contact points to normalized ground coordinates $(gx, gy)$.

### 2.2 YOLO Detection Implementation
- **Script:** `traffic_yolo.py`
- **Model:** Ultralytics YOLOv8m (`yolov8m.pt`, 52.1 MB).
- **Target Classes:** 4 vehicle categories from COCO:
  - Class 2: `car`
  - Class 3: `motorcycle`
  - Class 5: `bus`
  - Class 7: `truck`
- **Inference Configuration:** `conf=0.20`, `imgsz=1920`.
- **Precomputed Ground Truth:** Complete tracking records are cached in `traffic_tracks.csv` (29,139 observations across 1,530 frames).

### 2.3 Object Tracking & Track ID Generation
- **Script:** `traffic_tracking.py`
- **Tracker:** ByteTrack (`bytetrack.yaml` tracker via YOLOv8).
- **Configuration:** `conf=0.25`, `imgsz=1920`, `vid_stride=1`.
- **Track IDs:** Integer identifiers persistent across consecutive frames. Across the 1,530 frames of `traffic2.mp4`, 697 unique vehicle tracks are identified.
- **Track Records:** Columns in `traffic_tracks.csv`: `frame, time, track_id, class, x1, y1, x2, y2`.

### 2.4 Vehicle Class Extraction
- Handled uniformly across the project via class dictionary:
  ```python
  CLASS_NAMES = {
      2: "car",
      3: "motorcycle",
      5: "bus",
      7: "truck"
  }
  ```
- Vehicle contact patch is computed as bottom-center: $(x_{bottom}, y_{bottom}) = ((x_1 + x_2)/2, y_2)$.

### 2.5 Speed Estimation
- **Scripts:** `traffic_speed.py`, `clean_ground_motion.py`, `ml/hybrid_traffic/feature_engineering.py`.
- **Methodology:**
  - 10-frame rolling window smoothing on bounding box centers (`smooth_x`, `smooth_y`).
  - Spatial displacement: $\Delta x, \Delta y, d = \sqrt{\Delta x^2 + \Delta y^2}$.
  - Velocity: $v_{px} = d / \Delta t$ in pixels/second.
  - Ground-plane velocity: $v_{gnd} = d_{gnd} / \Delta t$ in normalized projection units/second.
  - Acceleration: $a = \Delta v / \Delta t$ in $\text{px}/\text{s}^2$.
  - Summary metrics stored in `traffic_vehicle_speed_summary.csv` and `traffic_smoothed_movement.csv`.
- **Safety Rule:** No uncalibrated metric km/h conversion is assumed for ML training; velocities remain in mathematically sound px/s and normalized ground units.

### 2.6 Lane and Zone Geometry
- **Zone Polygon Files (.npy):**
  - `ZONE 1` (`lane_zone_points.npy`): Main Approach Corridor (`LANE_NORTH_INFLOW`)
  - `ZONE 2` (`zone_2_points.npy`): Eastbound Exit / Turning Lane (`LANE_EAST_TURNING`)
  - `ZONE 3` (`zone_3_points.npy`): Central Intersection Junction Box (`UNKNOWN` / crossing core)
  - `ZONE 4` (`zone_4_points.npy`): Southbound Queue Area (`LANE_SOUTH_QUEUE`)
  - `ZONE 5` (`zone_5_points.npy`): Westbound Inflow Lane (`LANE_WEST_INFLOW`)
  - `ZONE 6` (`zone_6_points.npy`): Southeast Exit Lane (`LANE_SE_OUTFLOW`)
- **Containment Test:** `cv2.pointPolygonTest(polygon, (bottom_x, bottom_y), False) >= 0`.

### 2.7 Traffic Density Calculation
- **Script:** `traffic_intelligence.py`
- Continuous rolling 10-second history window ($10 \text{ s} \times 30 \text{ fps} = 300 \text{ frames}$) per zone.
- Rolling metrics computed: `current_count`, `average_count` (mean over 300 frames), `peak_count`, and `sustained_seconds`.
- Density categories:
  - $\le 2$ vehicles: `LOW`
  - $3 - 5$ vehicles: `MEDIUM`
  - $\ge 6$ vehicles: `HIGH`

### 2.8 Deterministic Congestion Engine
- **Script:** `congestion_engine.py`
- Formula:
  $$\text{Congestion Score} = 0.45 \times \text{Density Score} + 0.35 \times \text{Flow Score} + 0.20 \times \text{Variation Score}$$
  - $\text{Density Score} = \min(100, (\text{total\_vehicles} / 200) \times 100)$
  - $\text{Flow Score} = \min(100, (\max(\text{flow}) / 150) \times 100)$
  - $\text{Variation Score} = \min(100, (\text{mean}(\Delta\text{flow}) / 150) \times 100)$
- Classification:
  - $< 25$: `LOW`
  - $25 - 50$: `MODERATE`
  - $50 - 75$: `HIGH`
  - $\ge 75$: `SEVERE`
- **Real-Time Priority Election:** `traffic_intelligence.py` computes 3-second trend (90 frames) comparing recent vs previous mean. Priority zone elected by tuple: `(level_score, trend_score, avg_count, sustained_sec)`.

### 2.9 Violation Detection Engine
- **Script:** `traffic_violation_engine.py`
- Analyzes tracking observations for infractions:
  - Overspeeding (clocked speed vs speed limits from `traffic/speed_limits.json`).
  - High-speed approaching threat (from `approaching_vehicle.py`).
  - Dangerous maneuvers and illegal stopping.
- Persisted in `traffic/violations.json`.

### 2.10 SUMO / TraCI Simulation Engine
- **Script:** `sumo_simulation_engine.py`
- Manages co-simulation with Eclipse SUMO 1.27.1 / TraCI.
- Reads `sumo/simulation.sumocfg`, `sumo/intersection.net.xml`, and `sumo/routes.rou.xml`.
- Gracefully degrades with clear diagnostic status if SUMO binaries are absent on PATH.

### 2.11 Existing API / Backend
- **Script:** `app.py` (Flask on port 5000, threaded, static file serving, byte-range video streaming).
- Key endpoints:
  - `/api/status`: System status and video metrics.
  - `/api/frame/<frame_num>` & `/api/telemetry/<frame_num>`: Frame-level vehicle detections and zone telemetry.
  - `/api/telemetry_batch`: Chunked batch telemetry for frontend client-side playback.
  - `/api/zones`: Zone polygon coordinates and roles.
  - `/api/violations` & `/api/violation_stats`: Active and historical infractions.
  - `/api/alerts`: Real-time safety alerts.
  - `/api/routes/evaluate`: Route hazard and congestion evaluation.
  - `/api/driver/feed`: Driver Mode synchronized feed.
  - `/api/simulation/status` & `/api/simulation/<action>`: SUMO simulation interface.

### 2.12 Frontend & Modes
- **Authority Mode UI:** `static/index.html` and `static/js/dashboard.js` render live video overlays, bounding box highlights, zone polygon canvases, priority dispatch badges, and statistical charts.
- **Driver Mode:** Syncs via `/api/driver/feed` with congestion warnings and advisory notes.

---

## 3. Trained Hybrid Model Specifications

- **Architecture:** `TCNTransformerHybrid` (`models/tcn_transformer_hybrid.py`)
- **Trained Weights:** `checkpoints/best_model.pth` (Epoch 7, Validation Loss: 6.6147, 226,937 parameters).
- **Input Dimension:** 20 timesteps ($T=20$) $\times$ 20 features ($F=20$).
- **Fitted Scaler:** `checkpoints/feature_scaler.joblib` (StandardScaler fitted on Phase 4 training sequences).
- **Exact Feature Vector (20 Features in Order):**
  1. `center_x` — Bounding box horizontal center (px)
  2. `center_y` — Bounding box vertical center (px)
  3. `delta_x` — Frame displacement $\Delta x$ (px)
  4. `delta_y` — Frame displacement $\Delta y$ (px)
  5. `displacement_image` — Euclidean pixel displacement $\sqrt{\Delta x^2 + \Delta y^2}$ (px)
  6. `speed_image_px_per_sec` — Velocity in image plane (px/s)
  7. `accel_image_px_per_sec2` — Acceleration in image plane ($\text{px}/\text{s}^2$)
  8. `heading_rad` — Movement heading angle $[-\pi, \pi]$ (rad)
  9. `heading_change_rad` — Angular heading change $[-\pi, \pi]$ (rad)
  10. `ground_x` — Homography transformed ground $X$ coordinate
  11. `ground_y` — Homography transformed ground $Y$ coordinate
  12. `ground_delta_x` — Ground displacement $\Delta gx$
  13. `ground_delta_y` — Ground displacement $\Delta gy$
  14. `displacement_ground` — Euclidean ground displacement $\sqrt{\Delta gx^2 + \Delta gy^2}$
  15. `ground_speed_norm_per_sec` — Ground velocity (units/s)
  16. `box_width` — Bounding box width $x_2 - x_1$ (px)
  17. `box_height` — Bounding box height $y_2 - y_1$ (px)
  18. `box_area` — Bounding box area $(x_2 - x_1)(y_2 - y_1)$ ($\text{px}^2$)
  19. `stopped_duration` — Cumulative consecutive seconds stopped (s)
  20. `speed_variance_rolling5` — 5-observation rolling variance of speed ($(\text{px}/\text{s})^2$)

- **Model Target Heads (10 Heads):**
  1. `target_congestion_level`: Multiclass (3 classes: LOW, MEDIUM, HIGH)
  2. `target_congestion_score`: Regression ([10.0, 100.0])
  3. `target_risk_level`: Multiclass (3 classes: SAFE, WARNING, HIGH)
  4. `target_is_approaching`: Binary (2 classes: NON_APPROACHING, APPROACHING)
  5. `target_approach_score`: Regression ([0.03, 100.0])
  6. `target_motion_state`: Multiclass (4 classes: STOPPED, ACCELERATING, DECELERATING, CRUISING)
  7. `target_maneuver_type`: Multiclass (4 classes: STATIONARY, TURNING_LEFT, TURNING_RIGHT, STRAIGHT)
  8. `target_has_infraction`: Binary (2 classes: COMPLIANT, INFRACTION)
  9. `target_infraction_type`: Multiclass (3 classes: NONE, OVERSPEEDING, ILLEGAL_STOPPING)
  10. `aux_next_displacement`: Regression (dim 2: $dx, dy$ in px)

---

## 4. Hybrid Model Integration Plan

```
VIDEO / CAMERA (traffic2.mp4 / live CCTV)
         ↓
YOLOv8m DETECTION (bounding boxes, confidence, class_id)
         ↓
ByteTrack OBJECT TRACKING (persistent track_id)
         ↓
FEATURE ADAPTER (20 exact features computed per observation)
         ↓
TEMPORAL SEQUENCE BUFFER (per-track FIFO deque, len=20, warmup accounting)
         ↓
STANDARD SCALER (checkpoints/feature_scaler.joblib)
         ↓
TCN-TRANSFORMER GATED HYBRID (checkpoints/best_model.pth)
         ↓
MULTI-TASK DECODER (10 output heads decoded with probabilities & confidences)
         ↓
SAFETY & UNCERTAINTY GATE (marks low-confidence predictions as uncertain)
         ↓
TRAFFIC-LEVEL AGGREGATOR (Vehicle-level, Lane-level, Zone-level, System-level)
         ↓
INTEGRATED CONGESTION ENGINE (Measured values + Model predictions = Fused intelligence)
         ↓
BACKEND REST APIS (app.py: /api/frame, /api/telemetry_batch, /api/driver/feed, /api/alerts)
         ↓
AUTHORITY MODE & DRIVER MODE DASHBOARDS
```

---

## 5. Audit Conclusion & Compliance Check

| Item | Status | Notes |
| :--- | :---: | :--- |
| **No retraining** | Verified | Frozen checkpoint `best_model.pth` will be loaded in inference mode (`eval()`). |
| **No dataset modification** | Verified | All `data/hybrid_traffic/` files remain completely untouched. |
| **No scaler alteration** | Verified | Loaded directly via `joblib.load("checkpoints/feature_scaler.joblib")`. |
| **Feature order consistency** | Verified | Exact 20 features mapped identically to `sequence_builder.py`. |
| **No removal of existing logic** | Verified | Deterministic congestion engine and rule-based priority election are preserved. |
| **Architecture separation** | Verified | Clean module structure in `ml/hybrid_traffic/` with production interfaces. |
