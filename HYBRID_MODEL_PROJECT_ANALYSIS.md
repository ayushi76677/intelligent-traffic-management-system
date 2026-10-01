# HYBRID MODEL PROJECT ANALYSIS & SYSTEM AUDIT
**Project:** Intelligent Traffic Management System (Emerge)  
**Document:** `HYBRID_MODEL_PROJECT_ANALYSIS.md`  
**Date:** September 2026  
**Status:** Pre-ML Technical Audit & System Inspection (Read-Only Stage)  
**Target:** Integration of a New Machine Learning Layer on Top of Existing Tracking

---

## Executive Summary & Strict Constraints

This document performs an exhaustive, ground-truth inspection of the entire intelligent traffic management codebase prior to designing or implementing any new Machine Learning (ML) layer.

**Core Directives Observed:**
- **Zero Disruption Rule:** No existing working functionality has been modified, deleted, replaced, or broken.
- **No Early Training:** No training scripts or model training runs have been executed.
- **Empirical Grounding:** Class meanings, kinematic units, and labels are reported strictly as implemented in code and persisted in data files—no guessing, no assuming pixel movement is km/h, and no invented labels.

---

## A. Existing Architecture

The system is a dual-mode, multi-tier traffic intelligence platform combining computer vision perception, spatial geometry, heuristic traffic engineering engines, digital-twin simulation, a Flask REST/streaming server, and an interactive web command center.

### 1. Dual Operational Modes
* **Authority Mode (Macroscopic Intersection Command Center):**
  * Ingests elevated municipal CCTV video (`traffic2.mp4`) covering a major urban intersection (modeled on AB Road & Ring Road crossing, Indore).
  * Tracks multi-lane vehicle flow across 6 distinct spatial zones.
  * Continuously evaluates queue lengths, moving average counts, 3-second traffic volume trends, and priority lane elections.
  * Manages municipal camera inventory, configurable speed limits, incident reporting, and traffic violations registry.
  * Connects to a SUMO 1.27.1 digital twin simulation network.
* **Driver Mode (Microscopic Situational Awareness HUD):**
  * Displays an in-cockpit HUD (`#driverModeView`) with ego-vehicle speedometer and road speed limit sign.
  * Synchronizes with the authority backend via `/api/driver/feed` to receive active severe bottleneck alerts, nearby zone congestion scores, active road hazards, and dynamic rerouting guidance.
  * Evaluates navigation paths against congested corridors via `/api/routes/evaluate`.

### 2. Architectural Subsystem Breakdown

```
Perception Layer (YOLOv8m + ByteTrack)
      │
      ▼
Raw Trajectory Persistence (traffic_tracks.csv: 34,679 rows)
      │
      ├──────────────────────────────┬─────────────────────────────┐
      ▼                              ▼                             ▼
Spatial Geometry & Zones      Kinematics & Motion Filtering    Perspective Transformation
(6 Polygons, Point-in-Poly)   (10-frame smoothing, median)    (H homography matrix, bird's-eye)
      │                              │                             │
      ▼                              ▼                             ▼
Zone Flow & Congestion        Relative Kinetic Speed           Normalized Ground Coordinates
(density, flow, variation)    (pixels/sec, ground units/sec)   (0..100 x 0..50 ground plane)
      │                              │                             │
      └──────────────────────────────┼─────────────────────────────┘
                                     ▼
                      Analytical Engines
                      - TrafficIntelligenceEngine (rolling 10s avg, 3s trend, priority election)
                      - TrafficViolationEngine (overspeeding, red-light, wrong-way, stopped)
                      - CongestionEngine (multi-factor composite score 0-100)
                      - SumoSimulationEngine (TraCI / XML network topology digital twin)
                                     │
                                     ▼
                      Backend Server (app.py: Flask on port 5000)
                      - 24+ REST endpoints
                      - HTTP 206 Partial Content video streaming (/video/traffic2.mp4)
                                     │
                                     ▼
                      Web Interface (static/index.html, dashboard.js, dashboard.css)
                      - Video canvas overlay with bounding boxes, contact points, zone polygons
                      - Synchronized 1530-frame scrubber with zero-lag O(1) telemetry playback
                      - Chart.js telemetry charts & Google Maps / Leaflet geospatial integration
                      - Authority Mode Command Center & Driver Mode Cockpit HUD
```

---

## B. Existing Data Pipeline

The project implements a sequential, multi-stage data processing pipeline operating on recorded CCTV footage:

```
[traffic2.mp4] (1920x1080 @ 30 FPS, 1530 frames, 50.97s)
      │
      ▼ (traffic_yolo.py / traffic_tracking_data.py)
[YOLOv8m Detection + ByteTrack Tracking]
      │ - Confidence: 0.25, imgsz: 1920, classes: [2, 3, 5, 7]
      │ - Tracker: bytetrack.yaml, vid_stride: 1
      ▼
[traffic_tracks.csv] (34,679 rows, 10 columns)
      │
      ├──► [traffic_speed.py]
      │     - Filters tracks with < 10 observations (retains 29,139 rows across 697 unique vehicles)
      │     - Rolling 10-frame window position smoothing (smooth_x, smooth_y)
      │     - Calculates dx, dy, distance_pixels, relative_pixel_speed, dominant direction
      │     - Outputs: traffic_smoothed_movement.csv (29,139 rows) & traffic_smoothed_summary.csv (697 rows)
      │
      ├──► [ground_plane.py]
      │     - Computes bottom-center contact point: bottom_center_x = (x1+x2)/2, bottom_center_y = y2
      │     - Warps contact points via homography H to arbitrary normalized space [0..100, 0..50]
      │     - Outputs: vehicle_ground_plane.csv (29,139 rows)
      │
      ├──► [clean_ground_motion.py]
      │     - Applies 7-frame median filter (smooth_ground_x, smooth_ground_y)
      │     - Rejects displacement jumps > 10.0 units
      │     - Outputs: clean_ground_motion.csv (29,139 rows)
      │
      ├──► [vehicle_closing_motion.py]
      │     - Measures 10-observation displacement (delta_x, delta_y, stable_speed)
      │     - Classifies approaching (delta_y > 0.20) vs receding
      │     - Outputs: vehicle_closing_motion.csv (29,139 rows)
      │
      ├──► [zone_traffic_analysis.py]
      │     - Filters clean_ground_motion.csv against Zone 1 polygon using cv2.pointPolygonTest
      │     - Aggregates vehicle counts, modal split, average clean speed, 10s interval flow
      │     - Outputs: zone_traffic_analysis.csv (1 row) & zone_traffic_flow.csv (6 intervals)
      │
      ├──► [congestion_engine.py]
      │     - Ingests zone_traffic_analysis.csv and zone_traffic_flow.csv
      │     - Evaluates density score, flow score, variation score
      │     - Outputs: congestion_analysis.csv (1 row)
      │
      └──► [traffic_intelligence.py]
            - Ingests traffic_tracks.csv and all 6 zone .npy files
            - Assigns zone containment to all 34,679 observations across 1530 frames
            - Precomputes 10-second rolling averages, 3-second trends, recommendations, and priority zone
            - Delivers instantaneous O(1) frame telemetry to Flask API
```

---

## C. Existing Model(s)

1. **Object Detection Model:**
   * **Model Architecture:** YOLOv8 Medium (`yolov8m.pt`, 52.1 MB).
   * **Origin:** Ultralytics pre-trained checkpoint on the COCO (Common Objects in Context) dataset.
   * **Active Filtered Classes:** Class IDs `[2, 3, 5, 7]`.
   * **Confidence Threshold:** 0.20 (`traffic_yolo.py`) and 0.25 (`traffic_tracking.py`, `traffic_tracking_data.py`).
   * **Input Resolution:** 1920 pixels (`imgsz=1920`).
2. **Multi-Object Tracking Algorithm:**
   * **Algorithm:** ByteTrack (`bytetrack.yaml`).
   * **Mechanism:** Two-stage association using Kalman filter motion prediction and bounding box IoU (Intersection-over-Union) bipartite matching for both high-confidence and low-confidence detection boxes.
   * **Output:** Persistent integer `track_id` per vehicle across frames.
3. **Perspective Transformation Model:**
   * **Homography Matrix ($H$):** $3 \times 3$ projective transformation matrix (`perspective_matrix.npy`) computed via OpenCV `cv2.getPerspectiveTransform(src, dst)` using 4 calibrated road points.
4. **Deterministic Heuristic Models:**
   * **Congestion Scoring Model:** Multi-factor weighted index ($0.45 \cdot \text{density} + 0.35 \cdot \text{flow} + 0.20 \cdot \text{variation}$).
   * **Trend Dynamics Model:** Moving average difference over 3-second (90-frame) windows with $\pm 0.75$ vehicle tolerance.
   * **Priority Zone Election Model:** Deterministic multi-attribute lexicographic sorter.
   * **Approaching Vehicle Model:** Heuristic scoring based on bounding box growth rate and vertical pixel displacement.
   * **Violation Detection Rules:** Threshold-based logic checking speed limits, red-light temporal windows, and zone flow direction vectors.

---

## D. Existing Features

### 1. `traffic_tracks.csv` (Primary Raw Tracking Feature Set)
Verified dimensions: **34,679 rows $\times$ 10 columns** across 1530 frames (0 to 1529, 0.0s to 50.967s).

| Column Name | Data Type | Physical / Mathematical Definition |
| :--- | :--- | :--- |
| `frame` | `int64` | Discrete video frame index ($0 \le \text{frame} \le 1529$). |
| `time` | `float64` | Video playback timestamp in seconds ($\text{frame} / 30.0$, rounded to 3 decimals, $0.0 \le t \le 50.967$). |
| `track_id` | `int64` | ByteTrack persistent vehicle identifier ($1 \le \text{track\_id} \le 1754$). |
| `class` | `int64` | COCO class category index (`2`, `3`, `5`, or `7`). |
| `x1` | `float64` | Bounding box top-left horizontal coordinate in image pixels ($0.0 \le x_1 \le 1920.0$). |
| `y1` | `float64` | Bounding box top-left vertical coordinate in image pixels ($0.0 \le y_1 \le 1080.0$). |
| `x2` | `float64` | Bounding box bottom-right horizontal coordinate in image pixels ($0.0 \le x_2 \le 1920.0$). |
| `y2` | `float64` | Bounding box bottom-right vertical coordinate in image pixels ($0.0 \le y_2 \le 1080.0$). |
| `center_x` | `float64` | Horizontal center of bounding box: $(x_1 + x_2) / 2.0$ in pixels. |
| `center_y` | `float64` | Vertical center of bounding box: $(y_1 + y_2) / 2.0$ in pixels. |

### 2. Derived Kinematic & Spatial Features (Across Downstream CSVs)

| Feature Name | Found In | Formula / Definition | Unit / Scale |
| :--- | :--- | :--- | :--- |
| `bottom_center_x` | `ground_plane.py`, `clean_ground_motion.py` | $(x_1 + x_2) / 2$ | Image pixels |
| `bottom_center_y` | `ground_plane.py`, `clean_ground_motion.py` | $y_2$ (contact patch) | Image pixels |
| `smooth_x`, `smooth_y` | `traffic_smoothed_movement.csv` | Rolling 10-frame mean of `center_x`, `center_y` | Image pixels |
| `dx`, `dy` | `traffic_smoothed_movement.csv` | $\text{smooth\_pos}(t) - \text{smooth\_pos}(t - 10)$ | Image pixels |
| `distance_pixels` | `traffic_smoothed_movement.csv` | $\sqrt{\Delta x^2 + \Delta y^2}$ | Image pixels |
| `relative_pixel_speed` | `traffic_smoothed_movement.csv` | $\text{distance\_pixels} / \Delta t$ | Pixels / second |
| `ground_x`, `ground_y` | `vehicle_ground_plane.csv` | OpenCV homography perspective transform of $(x_c, y_2)$ | Normalized units $[0..100, 0..50]$ |
| `ground_speed` | `vehicle_ground_plane.csv` | $\sqrt{\Delta ground\_x^2 + \Delta ground\_y^2} / \Delta t$ | Ground units / second |
| `smooth_ground_x`, `_y` | `clean_ground_motion.csv` | 7-frame rolling median of `ground_x`, `ground_y` | Ground units |
| `clean_distance` | `clean_ground_motion.csv` | Filtered ground displacement ($\le 10.0$ jump limit) | Ground units |
| `clean_speed` | `clean_ground_motion.csv` | $\text{clean\_distance} / \Delta t$ | Ground units / second |
| `box_width`, `box_height` | `approaching_vehicle.py` | $x_2 - x_1$, $y_2 - y_1$ | Image pixels |
| `box_area` | `approaching_vehicle.py` | $\text{box\_width} \times \text{box\_height}$ | Pixels$^2$ |
| `size_growth` | `approaching_vehicle.py` | $(\text{area}_{\text{final}} - \text{area}_{\text{initial}}) / \text{area}_{\text{initial}}$ | Ratio |
| `density_score` | `congestion_analysis.csv` | $\min(100, (\text{total\_vehicles} / 200) \times 100)$ | Score $0 - 100$ |
| `flow_score` | `congestion_analysis.csv` | $\min(100, (\text{peak\_interval\_flow} / 150) \times 100)$ | Score $0 - 100$ |
| `variation_score` | `congestion_analysis.csv` | $\min(100, (\text{mean\_abs\_diff\_flow} / 150) \times 100)$ | Score $0 - 100$ |
| `congestion_score` | `congestion_analysis.csv` | $0.45 S_{\text{density}} + 0.35 S_{\text{flow}} + 0.20 S_{\text{variation}}$ | Score $0 - 100$ |

---

## E. Existing Labels

### 1. Supervised Labels (Ground Truth from Object Detection Model)
The **ONLY** supervised learning labels in the entire project are the 4 COCO vehicle classes:

| Class ID | Explicit Mapping in Code | Total Observations in `traffic_tracks.csv` | Share (%) |
| :---: | :---: | :---: | :---: |
| **`2`** | `"car"` | 13,596 | 39.21% |
| **`3`** | `"motorcycle"` | 12,609 | 36.36% |
| **`7`** | `"truck"` | 5,274 | 15.21% |
| **`5`** | `"bus"` | 3,200 | 9.23% |
| **Total** | | **34,679** | **100.0%** |

### 2. Rule-Based Heuristic Labels (NOT Supervised Ground Truth)
All other "labels" throughout the codebase are deterministic outputs of heuristic rules and threshold comparisons:

* **Movement Direction Labels** (from `traffic_speed.py` / `clean_ground_motion.py`):
  * `"STATIONARY"`: $|\Delta x| < 3.0$ and $|\Delta y| < 3.0$ pixels.
  * `"LEFT"`, `"RIGHT"`: $|\Delta x| > |\Delta y|$ and $\Delta x < 0$ or $> 0$.
  * `"UP"`, `"DOWN"`: $|\Delta y| \ge |\Delta x|$ and $\Delta y < 0$ or $> 0$.
* **Traffic Level Labels:**
  * In `zone_traffic_analysis.py`: `"LOW"` ($<10$), `"MEDIUM"` ($10-24$), `"HIGH"` ($\ge 25$).
  * In `traffic_intelligence.py`: `"LOW"` ($\le 2$), `"MEDIUM"` ($3-5$), `"HIGH"` ($\ge 6$).
* **Traffic Trend Labels** (from `traffic_intelligence.py`):
  * `"INCREASING"`: $\Delta_{\text{recent 90 frames}} - \Delta_{\text{prior 90 frames}} > +0.75$.
  * `"DECREASING"`: Difference $< -0.75$.
  * `"STABLE"`: Difference within $[-0.75, +0.75]$.
* **Congestion Severity Labels** (from `congestion_engine.py`):
  * `"LOW"` ($S < 25$), `"MODERATE"` ($25 \le S < 50$), `"HIGH"` ($50 \le S < 75$), `"SEVERE"` ($S \ge 75$).
* **Advisory / Action Labels:**
  * `"NORMAL FLOW"`, `"MONITOR"`, `"MONITOR - BUILDING"`, `"MONITOR - CLEARING"`, `"SUSTAINED HIGH TRAFFIC"`, `"CONGESTION BUILDING"`, `"CONGESTION CLEARING"`.
* **Motion Category Labels** (from `vehicle_closing_motion.py`):
  * `approaching = True` ($\Delta y_{\text{ground}} > 0.20$ and distance $> 0.20$).
* **Risk Level Labels** (from `approaching_vehicle.py`):
  * `"SAFE"` ($\text{score} < 60$), `"WARNING"` ($60 \le \text{score} < 80$), `"HIGH"` ($\text{score} \ge 80$).
* **Violation Type Labels** (from `traffic_violation_engine.py`):
  * `"Overspeeding"`, `"Red-light violation"`, `"Wrong-way driving"`, `"Dangerous approach"`, `"Illegal stopping"`.

> [!WARNING]
> **No Ground-Truth Congestion or Violation Labels Exist:**  
> These categorical outputs were created by heuristic business logic, not human annotators or external ground-truth sensors. Training a supervised ML model directly on these labels would constitute *pseudo-label distillation* of the heuristic rules, not learning empirical traffic phenomena.

---

## F. Existing Calibration

### 1. Calibration Points (`calibration_points.npy`)
An array of 4 vertices (shape $4 \times 2$, integer/float32) defining a trapezoidal road segment in the $1920 \times 1080$ camera view:
```
Point 1 (P1, Top-Left):     (600, 516)
Point 2 (P2, Top-Right):    (1904, 528)
Point 3 (P3, Bottom-Left):  (12, 1062)
Point 4 (P4, Bottom-Right): (1905, 1056)
```

### 2. Homography Perspective Matrix (`perspective_matrix.npy`)
Computed via `cv2.getPerspectiveTransform(src, dst)` where `src = [P1, P2, P4, P3]` and `dst` is the $1200 \times 700$ bird's-eye image frame:
$$\begin{bmatrix}
 1.45346734\times 10^{0} &  1.56527252\times 10^{0} & -1.67976102\times 10^{3} \\
-2.78616116\times 10^{-2} &  3.02762846\times 10^{0} & -1.54553932\times 10^{3} \\
-4.94424453\times 10^{-5} &  1.30787120\times 10^{-3} &  1.00000000\times 10^{0}
\end{bmatrix}$$

### 3. Scientific Reality: Speed Calculation & Calibration Limitations
* **Arbitrary Normalized Space:** In `ground_plane.py`, destination points map to an arbitrary $[0..100, 0..50]$ normalized plane:
  `DEST_POINTS = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)`
* **Absence of Metric Surveying:** No physical metric distance survey (e.g., laser distance measurement between lane lines or ground landmarks) was conducted for this video feed.
* **Pixel Speed vs. km/h:** In `traffic_speed.py`, speed is explicitly named `relative_pixel_speed` ($\text{pixels} / \text{second}$). In `clean_ground_motion.py`, speed is in normalized ground units per second.
* **Demonstration Conversion Factor:** In `traffic_violation_engine.py`, raw pixel speeds are scaled using an empirical multiplier (`speed_kmh = round(min(115.0, max(12.0, raw_max_px * 0.048)), 1)`) purely for demonstration and evidence logging.
* **Scientific Constraint:** **Pixel displacement does NOT natively equal physical km/h**. Any machine learning model estimating velocity must treat speeds as relative kinetic features unless calibrated against surveyed ground-truth distances.

---

## G. Existing Zones

The intersection is partitioned into 6 analytical zones defined by 4-vertex convex polygons in the $1920 \times 1080$ pixel coordinate space:

| Zone | File Name | Vertices $[(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)]$ | Functional Corridor Role | Speed Limit |
| :--- | :--- | :--- | :--- | :---: |
| **ZONE 1** | `lane_zone_points.npy` | `[[6, 548], [1009, 488], [1905, 518], [1268, 1070]]` | Main Approach Corridor (AB Road North Inflow) | 40 km/h |
| **ZONE 2** | `zone_2_points.npy` | `[[1358, 518], [1748, 595], [1905, 516], [1688, 435]]` | Eastbound Exit / Turning Lane (Ring Road Outflow) | 35 km/h |
| **ZONE 3** | `zone_3_points.npy` | `[[1019, 488], [732, 600], [379, 545], [700, 476]]` | Intersection Core Junction (Central Square Crossing) | 25 km/h |
| **ZONE 4** | `zone_4_points.npy` | `[[225, 766], [1422, 1027], [1792, 624], [1006, 512]]` | Southbound Inflow Queue Area (Palasia Approach) | 30 km/h |
| **ZONE 5** | `zone_5_points.npy` | `[[380, 528], [484, 616], [16, 902], [19, 556]]` | Westbound Inflow Approach (Left Side Inflow) | 30 km/h |
| **ZONE 6** | `zone_6_points.npy` | `[[1393, 532], [1769, 588], [1582, 1054], [1105, 952]]` | Southeast Exit Corridor (MR-10 Connector Outflow) | 35 km/h |

### Spatial Point-in-Polygon Methodology
To determine whether a vehicle belongs to a zone, the system evaluates the **bottom-center point of the bounding box**:
$$(x_{\text{bottom}}, y_{\text{bottom}}) = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
Using OpenCV's `cv2.pointPolygonTest(zone_polygon, (x_bottom, y_bottom), measureDist=False) >= 0`. This contact-point method prevents false positives from tall vehicles whose roofs extend into adjacent zones.

---

## H. Existing Congestion Logic

Congestion quantification operates across two distinct mathematical engines:

### 1. Macroscopic Multi-Factor Index (`congestion_engine.py`)
Computes an objective, normalized congestion score ($0 \le S \le 100$):
* **Vehicle Density ($S_{\text{density}}$):**
  $$S_{\text{density}} = \min\left(100, \frac{\text{Total Vehicles in Zone}}{200} \times 100\right)$$
* **Flow Volume ($S_{\text{flow}}$):**
  $$S_{\text{flow}} = \min\left(100, \frac{\text{Peak 10s Vehicle Flow}}{150} \times 100\right)$$
* **Flow Variation ($S_{\text{variation}}$):**
  $$S_{\text{variation}} = \min\left(100, \frac{\text{Mean Absolute Flow Change between 10s Bins}}{150} \times 100\right)$$
* **Composite Congestion Score ($S_{\text{congestion}}$):**
  $$S_{\text{congestion}} = 0.45 \cdot S_{\text{density}} + 0.35 \cdot S_{\text{flow}} + 0.20 \cdot S_{\text{variation}}$$
* **Categorical Mapping:**
  * $0 \le S < 25$: `LOW`
  * $25 \le S < 50$: `MODERATE`
  * $50 \le S < 75$: `HIGH`
  * $75 \le S \le 100$: `SEVERE`

### 2. Real-Time Moving Average & Priority Election (`traffic_intelligence.py`)
* **Moving Window Average:** 10-second history window ($W_{\text{history}} = 300\text{ frames}$ at 30 FPS).
* **Trend Analysis:** 3-second comparison window ($W_{\text{trend}} = 90\text{ frames}$):
  $$\Delta_{\text{trend}} = \overline{\text{Count}}_{\text{last 90 frames}} - \overline{\text{Count}}_{\text{prior 90 frames}}$$
  * $\Delta_{\text{trend}} > +0.75 \rightarrow$ `INCREASING`
  * $\Delta_{\text{trend}} < -0.75 \rightarrow$ `DECREASING`
  * $|\Delta_{\text{trend}}| \le 0.75 \rightarrow$ `STABLE`
* **Priority Zone Election:** Deterministic ranking of active zones using the multi-attribute tuple:
  $$\text{Priority} = \arg\max_{z \in \text{Zones}} \Big( \text{level\_score}(z), \; \text{trend\_score}(z), \; \overline{\text{count}}_{10s}(z), \; T_{\text{sustained}}(z) \Big)$$
  where $\text{level\_score} \in \{\text{LOW}: 0, \text{MEDIUM}: 1, \text{HIGH}: 2\}$ and $\text{trend\_score} \in \{\text{DECREASING}: 0, \text{STABLE}: 1, \text{INCREASING}: 2\}$.

---

## I. Existing Violation Logic

The violation engine (`traffic_violation_engine.py`, `traffic/violations.json`) evaluates 5 distinct infraction types grounded in tracking evidence and zone constraints:

1. **Overspeeding:**
   * Reads road speed limits from `traffic/speed_limits.json` (e.g. 50 km/h for AB Road).
   * Compares vehicle measured speed against allowed limit.
   * Severity: Excess $<10 \rightarrow$ `LOW`; $10-19 \rightarrow$ `MEDIUM`; $20-34 \rightarrow$ `HIGH`; $\ge 35 \rightarrow$ `CRITICAL`.
   * Penalties: INR 1000 (`LOW`/`MEDIUM`) or INR 2000 (`HIGH`/`CRITICAL`).
2. **Red-Light Violation:**
   * Identifies vehicles traversing the stop line or intersection core (Zone 3) during active red phase (frames 300 to 600 baseline window).
   * Penalty: INR 1000. Severity: `CRITICAL` or `HIGH`.
3. **Wrong-Way Driving:**
   * Tracks directional trajectory vectors in designated one-way outflow zones (e.g. westbound movement in Zone 2).
   * Penalty: INR 5000. Severity: `CRITICAL`.
4. **Dangerous High-Speed Approach:**
   * Evaluates rapid closing velocity combined with high bounding box area expansion rate ($\ge 80/100$) in approach corridors.
   * Penalty: INR 2500. Severity: `HIGH`.
5. **Illegal Stopping / Yellow Box Blocking:**
   * Detects vehicles remaining stationary (`speed == 0.0`) inside the central junction box (Zone 3) for $>180$ frames (6 seconds).
   * Penalty: INR 500. Severity: `MEDIUM`.

All violations are persisted to `traffic/violations.json` with unique IDs (`VIO-2026-0101` etc.), timestamps, evidence strings, and lifecycle statuses (`ACTIVE`, `REVIEWED`, `CITATION ISSUED`, `DISMISSED`).

---

## J. Existing API Structure

The Flask backend (`app.py`, port 5000) exposes 24+ REST and streaming endpoints:

### Core Telemetry & Streaming
* `GET /`: Serves the Authority Command Center web application (`static/index.html`).
* `GET /video/traffic2.mp4`: **HTTP 206 Partial Content (Byte-Range)** video streaming enabling instant seeking and zero-buffering hardware-accelerated playback.
* `GET /api/status`: System health, video dimensions, FPS, total frames, zones count, unique vehicle count.
* `GET /api/health`: Health probe returning operational status.
* `GET /api/frame/<int:frame_num>` / `GET /api/telemetry/<int:frame_num>`: Instant $O(1)$ telemetry for any frame (vehicle boxes, track IDs, classes, zone metrics, priority election).
* `GET /api/telemetry_batch?start=0&count=150`: Windowed telemetry batching for zero-lag local frontend caching.
* `GET /api/analytics`: Aggregated dataset metrics, fleet class distribution, average speeds, flow timeline for dashboard charts.

### Spatial Zones & Municipal Infrastructure
* `GET /api/zones`: Full metadata for all 6 monitoring zones, congestion scores, coordinates, and pixel polygons.
* `POST /api/zones`: Dynamically creates a new monitoring zone.
* `PUT /api/zone/<zone_id>`: Updates existing zone attributes.
* `DELETE /api/zone/<zone_id>`: Deletes a custom zone.
* `POST /api/save_zone_config`: Saves camera-specific monitoring zone configurations.
* `GET /api/locations`: Registry of major Indian cities, coordinates, and intersections (`locations/cities.json`).
* `GET /api/cameras`: Inventory of 13 municipal CCTV cameras across Indore with operational statuses (`cameras/cameras.json`).
* `POST /api/cameras`: Deploys a new CCTV camera sensor record.
* `GET /api/camera/<camera_id>`: Specific camera telemetry and live metrics.
* `DELETE /api/camera/<camera_id>`: Removes camera sensor.
* `GET /api/traffic_corridors`: Map arterial corridors and flow status (`traffic/traffic_data.json`).
* `GET /api/location_history`: Historical audit records of traffic intelligence by city/location.

### Violations, Speed Limits & Incidents
* `GET /api/violations`: Filterable violations list (by `camera_id`, `severity`, `status`).
* `POST /api/violations`: Registers and persists a new detected violation.
* `PUT /api/violation/<violation_id>/status`: Updates violation lifecycle status.
* `GET /api/violation_stats`: Violation aggregates grouped by severity and infraction type.
* `GET /api/speed_limits`: Configurable road-level and zone-level speed limits (`traffic/speed_limits.json`).
* `POST /api/speed_limits`: Updates speed limit for a designated corridor.
* `GET /api/incidents`: Active traffic incidents and hazards (`traffic/incidents.json`).
* `POST /api/incidents`: Reports a new road incident.
* `PUT /api/incident/<incident_id>`: Updates incident details or status.
* `GET /api/alerts`: Real-time combined critical alerts (high-speed, severe congestion, active hazards).

### Driver Mode & Route Evaluation
* `GET /api/driver/feed`: Synchronized intelligence endpoint consumed by Driver Mode HUD (current road, speed limits, nearby severe zones, active hazards, diversion advisories).
* `POST /api/routes/evaluate`: Evaluates navigation waypoints against active severe congestion polygons and road hazards; returns delay estimates and reroute recommendations.

### SUMO Simulation Controls
* `GET /api/simulation/status`: Reports digital-twin status, active vehicles, edge counts, and host SUMO/TraCI availability.
* `POST /api/simulation/<action>`: Executes simulation lifecycle actions (`start`, `pause`, `stop`, `reset`, `step`).

---

## K. Existing Frontend Integration Points

The web frontend (`static/index.html`, `static/js/dashboard.js`, `static/css/dashboard.css`) connects directly to the backend through cleanly defined DOM hooks:

1. **Synchronized Video + Canvas Overlay:**
   * HTML `<video id="trafficVideo">` streams `/video/traffic2.mp4` with hardware decoding.
   * Overlay `<canvas id="cvCanvas">` renders frame-synchronized bounding boxes, tracking IDs, vehicle class tags, contact points, and zone polygons via `requestAnimationFrame(animationLoop)`.
   * Controlled via Scrubber `<input type="range" id="videoScrubber">` (0 to 1529 frames).
   * Layer toggle checkboxes: `#toggleBBoxes`, `#toggleTrackIds`, `#toggleZones`, `#toggleContactPoints`.
2. **KPIs & Priority Decision Display:**
   * DOM elements `#kpiActiveVehicles`, `#kpiHighestZone`, `#kpiTrend`, `#kpiPriority`, `#kpiPriorityAction`.
   * Decision panel: `#decisionZoneTitle`, `#decisionLevelBadge`, `#decisionAvg`, `#decisionPeak`, `#decisionTrend`, `#decisionAction`, `#decisionReason`.
3. **Telemetry & Chart.js Visualizations:**
   * Canvas `#chartZoneVolumes`: Bar chart tracking 6 zones' average vs. peak vehicle volume.
   * Canvas `#chartModalSplit`: Doughnut chart visualizing fleet mix (Cars, Motorcycles, Buses, Trucks).
4. **Geospatial & Map Components:**
   * Google Maps / Leaflet integration rendering municipal CCTV camera locations, corridor flow heatmaps, and zone bounding polygons.
5. **Mode Switching:**
   * Header toggle buttons `#btnModeAuthority` and `#btnModeDriver`.
   * Switching to Driver Mode displays `#driverModeView` (cockpit speedometer `#dmEgoSpeed`, speed limit sign `#dmSpeedLimitSign`, bottleneck banner `#dmBottleneckText`, hazards list `#dmHazardsList`, and reroute button `#btnDmAcceptReroute`).
6. **Modals Management:**
   * `#addCctvModal`: Deploys new cameras to `/api/cameras`.
   * `#speedLimitsModal`: Updates corridor limits to `/api/speed_limits`.
   * `#violationsModal`: Interactive table of infractions with status updates to `/api/violation/<id>/status`.
   * `#simulationModal`: Interactive controls for SUMO simulation (`#btnSimStart`, `#btnSimPause`, `#btnSimStep`, `#btnSimReset`).
   * `#zoneModalBackdrop`: Interactive calibration of zone approach names, directions, and signal phase assignments.
   * `#historyModalBackdrop`: Historical traffic audit logs.

---

## L. Files That Must NOT Be Modified

To preserve 100% of working functionality, the following existing files must remain strictly untouched:

| File Name | Functional Reason for Protection |
| :--- | :--- |
| `traffic2.mp4` | Master raw video recording (1920x1080 @ 30 FPS, 1530 frames). Cannot be corrupted or overwritten. |
| `yolov8m.pt` | Master pre-trained YOLOv8 Medium checkpoint (52.1 MB). |
| `traffic_tracks.csv` | Master ground-truth tracking dataset (34,679 rows). Used by all downstream pipelines. |
| `traffic_yolo.py` | Working YOLOv8 detection verification script. |
| `traffic_tracking.py` | Working ByteTrack tracking verification script. |
| `traffic_tracking_data.py` | Generator script that produced `traffic_tracks.csv`. |
| `calibration_points.npy` | Calibrated camera road boundary coordinates. Modifying breaks homography. |
| `perspective_matrix.npy` | Calibrated $3 \times 3$ perspective transformation matrix. |
| `lane_zone_points.npy` | Polygon coordinates for Zone 1. |
| `zone_2_points.npy` to `zone_6_points.npy` | Polygon coordinates for Zones 2 through 6. |
| `traffic_speed.py` | Trajectory smoothing and relative velocity calculation script. |
| `ground_plane.py` | Perspective transformation of contact points into ground-plane coordinates. |
| `clean_ground_motion.py` | Median smoothing and jump rejection script. |
| `vehicle_closing_motion.py` | 10-observation closing motion and approaching detector. |
| `zone_traffic_analysis.py` | Zone 1 volume and flow aggregation script. |
| `congestion_engine.py` | Multi-factor congestion scoring script. |
| `traffic_intelligence.py` | Core analytical engine providing precomputed frame telemetry to `app.py`. |
| `traffic_violation_engine.py` | Infraction detection and management engine. |
| `sumo_simulation_engine.py` | SUMO/TraCI co-simulation and fallback topology parser. |
| `sumo/*` (`simulation.sumocfg`, `*.xml`) | Working SUMO digital-twin network and route files. |
| `app.py` | Working Flask server and REST/streaming API. |
| `static/index.html` | Authority Command Center and Driver HUD markup. |
| `static/js/dashboard.js` | Frontend controller managing canvas loop, charts, and API polling. |
| `static/css/dashboard.css` | Production styling and UI theme. |

---

## M. Files That Can Safely Be Extended

The following extension points can safely host the new machine-learning layer without altering existing working code:

| Extension Target | Extension Strategy & Safe Mechanism |
| :--- | :--- |
| **New Module: `ml_engine.py`** | Standalone ML inference engine encapsulating trained models, feature transformers, and prediction pipelines. Can be imported cleanly into backend or run independently. |
| **New Directory: `models/` or `ml_models/`** | Directory for storing trained model artifacts (`.joblib`, `.json`, or weights), scaler objects, and feature column schemas. |
| **New Pipeline: `traffic_ml_features.py`** | Feature engineering pipeline that reads `traffic_tracks.csv` or `clean_ground_motion.csv` and derives sliding-window, spatial, lag, and kinetic feature matrices. |
| **New Training Script: `train_hybrid_model.py`** | Dedicated, standalone training script executed strictly when authorized, with train/test splits, cross-validation, and metric reporting. |
| **Backend Integration in `app.py`** | Adding *new, additive REST endpoints* (e.g. `/api/ml/predict_congestion`, `/api/ml/forecast_flow`, `/api/ml/anomaly_score`) without touching any of the 24 existing routes. |
| **Frontend Integration in `static/js/dashboard.js`** | Calling new ML endpoints to display ML forecast overlays or confidence intervals on existing chart panels or HUD badges. |

---

## N. Missing Information Required for Training

Before training any machine-learning model on top of the tracking data, the following fundamental questions and technical requirements must be clarified:

### 1. Target Variable & Objective Function Formulation
* **What is the exact task of the ML model?**
  * *Option 1: Temporal Traffic Flow / Volume Forecasting* — Predicting total vehicle count or zone density at time $t + \Delta t$ (e.g., forecasting 5 seconds or 10 seconds ahead into the future).
  * *Option 2: Congestion State Classification* — Classifying intersection stress into `LOW`, `MODERATE`, `HIGH`, `SEVERE`. (Must decide whether to predict heuristic congestion scores or define a new metric).
  * *Option 3: Kinetic Anomaly / Incident Detection* — Unsupervised or semi-supervised detection of abnormal vehicle deceleration, sudden swerves, or stopped vehicle bottlenecks.
  * *Option 4: Multi-Zone Demand Prediction* — Predicting which zone will become the next priority bottleneck.
  * *Option 5: Trajectory Prediction* — Predicting future $(x, y)$ positions of active vehicle tracks over the next 1-3 seconds.

### 2. Nature of Training Labels
* As demonstrated in Section E, **there are no human-labeled ground-truth congestion or safety targets** in the repository.
* If a supervised model is trained to predict `congestion_level` or `traffic_level`, it will learn to approximate the hardcoded threshold formulas ($0.45\text{ density} + 0.35\text{ flow} + 0.20\text{ variation}$). Is pseudo-labeling / rule distillation acceptable, or should the ML task be framed as **self-supervised time-series forecasting** (where future vehicle counts serve as natural ground truth)?

### 3. Temporal Horizon & Sample Size
* `traffic_tracks.csv` covers **50.97 seconds** (1530 frames) of recorded footage.
* While 34,679 vehicle observations provide substantial trajectory samples for tracking kinematics, 51 seconds represents a limited temporal window for macroscopic time-series forecasting (e.g., only five 10-second intervals or fifty 1-second intervals).
* Will the ML layer:
  * Operate microscopically at the **vehicle trajectory / track level** ($N = 697$ to $1,754$ tracks, thousands of frame transitions)?
  * Operate at the **spatial zone level** across sliding 1-second time windows ($N \approx 1,500$ temporal feature vectors across 6 zones)?
  * Augment with synthetic simulation data from SUMO?

### 4. Computational Framework & Dependencies
* Current host Python environment: **Python 3.13.7**.
* Installed libraries: `scikit-learn 1.8.0`, `scipy 1.17.0`, `numpy 2.4.4`, `pandas 3.0.2`, `opencv-python 5.0.0.93`, `matplotlib 3.10.8`, `joblib 1.5.3`.
* Libraries NOT currently installed: `torch`, `torchvision`, `ultralytics`, `xgboost`, `lightgbm`, `tensorflow`.
* The ML layer can either:
  * Utilize high-performance classical ML via `scikit-learn` (Random Forest Regressor/Classifier, Gradient Boosting, Ridge/Lasso, Support Vector Machines, Multi-Layer Perceptrons, Isolation Forests) — fully supported immediately without installing new packages.
  * Or require installation of deep learning packages (`torch` / `xgboost`), subject to Python 3.13 Windows wheel compatibility.

---

## Conclusion & Readiness

The repository inspection is complete. Every pipeline, file, zone coordinate, formula, and endpoint has been verified against active code and persistent files. The existing application remains fully intact, operational, and prepared for the addition of an isolated, non-breaking hybrid ML layer once the modeling objectives are confirmed.

**DO NOT TRAIN YET — Awaiting confirmation of Section N.**
