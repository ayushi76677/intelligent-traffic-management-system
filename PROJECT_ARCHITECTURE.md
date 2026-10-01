# Smart Traffic Management System — Project Architecture Document

**Author & Role:** Lead Software Engineer  
**Project:** Smart Traffic Management System (Authority Mode & Driver Mode)  
**Status:** Pre-implementation System Audit & Technical Specification  
**Technology Stack:** Python, OpenCV, Ultralytics YOLOv8, ByteTrack, NumPy, Pandas, Eclipse SUMO 1.27.1, TraCI, HTML/CSS/JavaScript (Frontend) — *Strictly Non-LLM Architecture*

---

## Executive Summary

The **Smart Traffic Management System** is a dual-mode, computer-vision and simulation-driven traffic intelligence platform. The system operates in two distinct operational paradigms:
1. **Authority Mode:** Macroscopic intersection intelligence, multi-zone vehicle tracking, queue and flow rate quantification, trend and congestion scoring, priority lane identification, and digital-twin signal synchronization.
2. **Driver Mode:** Microscopic, driver-perspective situational awareness (distance closure and approaching vehicle warnings, developed under rigorous scientific boundaries).

This document captures the pre-existing codebase, validates working pipelines, catalogues data schemas, establishes architectural boundaries, and outlines the system roadmap.

---

## A. Existing Files & Directory Inventory

The workspace currently contains 49 top-level files, 1 top-level `runs` directory, and 1 `sumo` simulation directory.

### 1. Video and Deep Learning Assets
* **`traffic2.mp4`** (102.9 MB): Primary high-definition (1920x1080, ~30 FPS, 51.0 seconds, 1531 frames) video capturing an elevated, angled view of a busy multi-lane Indian urban intersection with mixed traffic (cars, auto-rickshaws, motorcycles, buses, trucks).
* **`yolov8m.pt`** (52.1 MB): Pre-trained Ultralytics YOLOv8 Medium object detection model checkpoint.
* **`runs/detect/predict/traffic2.avi`** (127.2 MB): Raw YOLOv8 detections rendered with bounding boxes.
* **`runs/detect/track-3/traffic2.avi`** (143.1 MB): YOLOv8 + ByteTrack output video with tracking IDs rendered.
* **`traffic_final.avi`** (126.9 MB): Annotated video featuring a black telemetry overlay box with interval traffic statistics.
* **`bird_eye_view.jpg`** (170 KB): Output image of the perspective transformation showing the bird's-eye projection of the intersection road surface.

### 2. Camera Calibration & Perspective Transformation
* **`calibration_tool.py`** (7.3 KB, 334 lines): Interactive OpenCV GUI showing a side-by-side view (original camera view on left, live warped bird's-eye perspective on right) for selecting 4 road calibration points. Saves `calibration_points.npy` and `perspective_matrix.npy`.
* **`camera_calibration.py`** (2.6 KB, 136 lines): Earlier calibration utility allowing manual selection of 4 road boundary points.
* **`perspective_transform.py`** (2.2 KB, 119 lines): Script that loads `calibration_points.npy`, generates a 1200x700 homography matrix, warps frame 0, and saves `perspective_matrix.npy` and `bird_eye_view.jpg`.
* **`calibration_points.npy`** (192 bytes): Quad coordinates representing the trapezoidal road region in image pixels:
  $$\text{Points} = [[600, 516], [1904, 528], [12, 1062], [1905, 1056]]$$
* **`perspective_matrix.npy`** (200 bytes): 3x3 OpenCV perspective homography transformation matrix.

### 3. Object Detection, Tracking & Vehicle Kinematics
* **`traffic_yolo.py`** (225 B, 13 lines): Executes YOLOv8m detection on `traffic2.mp4` with classes `[2, 3, 5, 7]` (car, motorcycle, bus, truck) at confidence 0.20 and image size 1920.
* **`traffic_tracking.py`** (278 B, 15 lines): Executes YOLOv8m with ByteTrack (`bytetrack.yaml`) tracking on `traffic2.mp4`.
* **`traffic_tracking_data.py`** (2.1 KB, 81 lines): Core data generator. Streams YOLOv8m + ByteTrack predictions across all frames and exports frame-by-frame vehicle trajectories into `traffic_tracks.csv`.
* **`traffic_tracks.csv`** (2.1 MB, 29,139 rows): Frame-by-frame tracking log with bounding boxes, vehicle class, and center coordinates.
* **`traffic_speed.py`** (5.8 KB, 294 lines): Applies rolling-average position smoothing (`WINDOW_FRAMES = 10`), derives displacement ($\Delta x, \Delta y$), calculates relative pixel velocity, determines movement direction (LEFT, RIGHT, UP, DOWN, STATIONARY), and aggregates per-vehicle summaries.
* **`traffic_smoothed_movement.csv`** (5.7 MB): Observation-level dataset enriched with smoothed coordinates and relative pixel speeds.
* **`traffic_smoothed_summary.csv`** (53.0 KB, 697 vehicles): Per-vehicle summary dataset (average and max pixel speeds, dominant direction, observation count).
* **`traffic_vehicle_movement.csv`** (6.5 MB) & **`traffic_vehicle_speed_summary.csv`** (167 KB): Earlier unsmoothed movement datasets (superseded by `traffic_smoothed_movement.csv`).

### 4. Ground Plane & Motion Analysis
* **`ground_plane.py`** (4.3 KB, 199 lines): Transforms vehicle ground-contact points (bottom-center: $x = (x_1 + x_2)/2, y = y_2$) through the homography matrix $H$ into a normalized ground-plane coordinate space $[0..100, 0..50]$ and calculates raw ground speed.
* **`vehicle_ground_plane.csv`** (8.7 MB): Tracking observations mapped to ground-plane coordinates.
* **`clean_ground_motion.py`** (4.4 KB, 203 lines): Filters noise from ground-plane motion using median smoothing (`window = 7`), rejects impossible jumps ($> 10.0$ units/frame), and computes clean ground speed.
* **`clean_ground_motion.csv`** (12.1 MB): Outlier-rejected ground-plane motion dataset.
* **`vehicle_closing_motion.py`** (4.2 KB, 186 lines): Analyzes long-term displacement over a 10-observation window, classifying motion state into `STATIONARY`, `APPROACHING`, or `RECEDING` based on ground-y delta.
* **`vehicle_closing_motion.csv`** (15.5 MB): Motion classification and closing speed dataset.
* **`approaching_vehicle.py`** (8.1 KB, 352 lines): Experimental driver-warning heuristic analyzing bounding box area expansion rate, downward movement, and pixel speed to assign an `approach_score` (0-100) and `risk_level` (`SAFE`, `WARNING`, `HIGH`).
* **`approaching_vehicle_analysis.csv`** (125.2 KB): Output of the approaching vehicle analysis.

### 5. Zone Definition & Spatial Traffic Analysis
* **`lane_zone_tool.py`** (4.5 KB, 231 lines): Interactive OpenCV tool for selecting the 4 points of Zone 1 (`lane_zone_points.npy`).
* **`multi_zone_tool.py`** (3.6 KB, 158 lines): Interactive OpenCV tool that sequentially defines and saves zones 2 through 6 (`zone_2_points.npy` through `zone_6_points.npy`).
* **`lane_zone_points.npy`** (192 bytes): Quad polygon points for Zone 1.
* **`zone_2_points.npy`** to **`zone_6_points.npy`** (160 bytes each): Quad polygon points for Zones 2, 3, 4, 5, and 6.
* **`zone_traffic_analysis.py`** (6.3 KB, 335 lines): Evaluates vehicle presence in Zone 1 using `cv2.pointPolygonTest`, computes modal class distribution, speeds, and 10-second flow rates.
* **`zone_traffic_analysis.csv`** (240 bytes): Statistical breakdown for Zone 1.
* **`zone_traffic_flow.csv`** (76 bytes): Vehicle counts across 10-second intervals for Zone 1.

### 6. Congestion & Decision Engines
* **`congestion_engine.py`** (5.5 KB, 273 lines): Multi-factor congestion scoring engine combining density, flow, and variation scores into a composite metric (0-100) with levels `LOW`, `MODERATE`, `HIGH`, `SEVERE`.
* **`congestion_analysis.csv`** (272 bytes): Output of the congestion engine.
* **`traffic_decision_engine.py`** (24.2 KB, 1094 lines): Comprehensive OpenCV dashboard rendering 6 zones, computing 10-second rolling averages, 3-second traffic trends (`INCREASING`, `STABLE`, `DECREASING`), recommendations (`CONGESTION BUILDING`, `CONGESTION CLEARING`, etc.), and dynamically electing a priority zone.
* **`traffic_analysis.py`** (2.6 KB, 132 lines): Samples 5 discrete frames across 10-second intervals from `traffic2.mp4` and exports vehicle counts to `traffic_timeline.csv`.
* **`traffic_analyzer.py`** (2.8 KB, 124 lines): Evaluates `traffic_timeline.csv` for high-traffic patterns and saves `analysis_report.json`.
* **`analysis_report.json`** (428 bytes): Summary of interval traffic and vehicle class distribution.
* **`traffic_policy.py`** (2.1 KB, 83 lines): Rule-based signal timing advisory engine evaluating `traffic_timeline.csv` to suggest green time adjustments.
* **`traffic_policy_report.csv`** (514 bytes): Actionable policy recommendations per interval.
* **`traffic_timeline.csv`** (199 bytes): 5-interval summary of vehicle counts and levels.

### 7. Dashboards & Video Utilities
* **`authority_dashboard.py`** (17.1 KB, 680 lines): Standalone OpenCV desktop GUI for Authority Mode monitoring all 6 zones with rolling averages, peak tracking, sustained high traffic timers, and congested zone identification.
* **`make_final_video.py`** (3.5 KB, 169 lines): Ingests ByteTrack video `runs/detect/track-3/traffic2.avi` and embeds interval telemetry boxes, saving `traffic_final.avi`.
* **`check_video.py`** (412 bytes, 17 lines): Diagnostic utility reporting video resolution, FPS, total frames, and duration.

### 8. SUMO Digital Twin Simulation Files
Located in `sumo/`:
* **`sumo/simulation.sumocfg`** (305 bytes): SUMO simulation run configuration file referencing `intersection.net.xml` and `routes.rou.xml` with 600s simulation horizon.
* **`sumo/routes.rou.xml`** (933 bytes): Defines vehicle type `car` and 4 directional routes and flows (`traffic_1` through `traffic_4`) across edges `A0A1`, `A1B1`, `B0B1`, `B1B0`.
* **`sumo/intersection.net.xml`** (10.7 KB): 2x2 grid junction road network generated via `netgenerate`.
* **`sumo/final_network/intersection.nod.xml`** (288 bytes): Node configuration file defining a central traffic-light-controlled intersection node (`center` at $x=0, y=0$) connected to 4 perimeter priority nodes (`west`, `east`, `north`, `south` at 300m distances).
* **`sumo/results/`**: Output directory for simulation results.

---

## B. Existing Computer Vision Pipeline Architecture

The CV pipeline processes recorded traffic footage through structured, sequential stages:

```
[traffic2.mp4]
      │
      ▼
[Ultralytics YOLOv8m] (Classes: Car [2], Motorcycle [3], Bus [5], Truck [7])
      │
      ▼
[ByteTrack Multi-Object Tracking] (Association across frames, track_id persistence)
      │
      ▼
[traffic_tracks.csv] (frame, time, track_id, class, x1, y1, x2, y2, center_x, center_y)
      │
      ├──────────────────────────────┬─────────────────────────────┐
      ▼                              ▼                             ▼
[Spatial Zone Allocation]   [Trajectory Smoothing]     [Ground-Plane Projection]
6 Polygon Zones             Window = 10 frames         Bottom-center (cx, y2) -> H
(cv2.pointPolygonTest)      Relative pixel velocity    vehicle_ground_plane.csv
      │                              │                             │
      ▼                              ▼                             ▼
[Zone Traffic Metrics]      [Dominant Direction]       [Median Noise Filter]
Counts, Modal Mix, Flow     LEFT/RIGHT/UP/DOWN         Window = 7, jump <= 10.0
zone_traffic_analysis.csv   traffic_smoothed_*.csv     clean_ground_motion.csv
      │                                                            │
      ▼                                                            ▼
[Decision Engine]                                      [Closing Motion Engine]
Rolling Averages, Trends,                              10-frame delta_y analysis
Priority Zone Election                                 Approaching/Receding/Stationary
```

### Scientific Limitations & Guardrails
1. **Pixel & Ground-Plane Speeds Are Arbitrary Units:**  
   The perspective transformation projects pixel coordinates into an arbitrary normalized space ($100 \times 50$). Because there is no metric surveying of physical road markers, **speeds cannot and must not be reported as true km/h**. They represent relative kinetic velocity.
2. **Camera Perspective Prohibits Driver Collision Risk / TTC Estimation:**  
   The intersection camera is an elevated, oblique infrastructure sensor. It has no ego-vehicle perspective. Calculating Time-to-Collision (TTC) or driver-centric safety margins from this viewpoint is scientifically invalid.
3. **Bounding Box Growth Does Not Prove Rash Driving:**  
   Bounding box size variations occur naturally as vehicles approach an elevated camera or change orientation while turning. Area expansion alone cannot be used as an indicator of aggressive driving.
4. **Domain Separation:**  
   * The elevated intersection camera is strictly reserved for **Authority Mode** (queue lengths, zone densities, signal demand, macro-flow).
   * **Driver Mode** must operate on a driver-perspective (dashcam/rear-view) feed where an ego-vehicle is explicitly defined.

---

## C. Existing Zone System

The intersection is partitioned into 6 distinct analytical zones defined by 4-point convex polygons:

| Zone Identifier | Coordinate File | Polygon Vertices $(x, y)$ | Functional Traffic Role |
| :--- | :--- | :--- | :--- |
| **ZONE 1** | `lane_zone_points.npy` | $(6, 548), (1009, 488), (1905, 518), (1268, 1070)$ | Main intersection approach & entry lane |
| **ZONE 2** | `zone_2_points.npy` | $(1358, 518), (1748, 595), (1905, 516), (1688, 435)$ | Right-side outgoing / turning lane |
| **ZONE 3** | `zone_3_points.npy` | $(1019, 488), (732, 600), (379, 545), (700, 476)$ | Central junction crossway |
| **ZONE 4** | `zone_4_points.npy` | $(225, 766), (1422, 1027), (1792, 624), (1006, 512)$ | Lower foreground queue area |
| **ZONE 5** | `zone_5_points.npy` | $(380, 528), (484, 616), (16, 902), (19, 556)$ | Far-left approach / side lane |
| **ZONE 6** | `zone_6_points.npy` | $(1393, 532), (1769, 588), (1582, 1054), (1105, 952)$ | Lower-right exit corridor |

### Spatial Containment Principle
Vehicles are assigned to zones using `cv2.pointPolygonTest`:
* **Evaluation Point:** Bottom-center of the bounding box $(x_c, y_2) = \left(\frac{x_1 + x_2}{2}, y_2\right)$.
* **Physical Basis:** The bottom-center closely approximates the vehicle's tire-road contact patch, preventing false assignments caused by tall vehicle roofs (e.g. double-decker buses or trucks).

---

## D. Existing Congestion Engine

The congestion engine (`congestion_engine.py`) derives an objective traffic stress index using a weighted multi-factor model:

### 1. Component Formulations
* **Density Score ($S_{\text{density}}$):**
  $$S_{\text{density}} = \min\left(100, \frac{\text{Total Vehicles}}{M_{\text{zone}}}\times 100\right), \quad M_{\text{zone}} = 200$$
* **Flow Score ($S_{\text{flow}}$):**
  $$S_{\text{flow}} = \min\left(100, \frac{\text{Peak 10s Vehicle Flow}}{M_{\text{interval}}}\times 100\right), \quad M_{\text{interval}} = 150$$
* **Variation Score ($S_{\text{variation}}$):**
  $$S_{\text{variation}} = \min\left(100, \frac{\text{Mean } |\Delta \text{Flow}|}{M_{\text{interval}}}\times 100\right)$$

### 2. Composite Score & Classification
$$S_{\text{congestion}} = 0.45 \cdot S_{\text{density}} + 0.35 \cdot S_{\text{flow}} + 0.20 \cdot S_{\text{variation}}$$

| Score Range | Congestion Level | Traffic System Meaning |
| :--- | :--- | :--- |
| $0 \le S < 25$ | **LOW** | Free-flow traffic; minimum green duration sufficient |
| $25 \le S < 50$ | **MODERATE** | Stable flow; balanced signal splits |
| $50 \le S < 75$ | **HIGH** | Approaching capacity; signal extension recommended |
| $75 \le S \le 100$ | **SEVERE** | Queue spillover; aggressive phase priority required |

---

## E. Existing Decision Engine & Signal Policy

The decision logic implemented across `traffic_decision_engine.py`, `authority_dashboard.py`, and `traffic_policy.py` provides real-time situational awareness.

### 1. Zone Classification & Trend Dynamics
* **Vehicle Count Levels:**
  * $\le 2$ vehicles: `LOW` (Green)
  * $3 - 5$ vehicles: `MEDIUM` (Yellow/Orange)
  * $\ge 6$ vehicles: `HIGH` (Red)
* **Trend Estimation ($W_{\text{trend}} = 3\text{ seconds}$):**
  $$\Delta_{\text{trend}} = \overline{\text{Count}}_{\text{recent}} - \overline{\text{Count}}_{\text{previous}}$$
  * $\Delta_{\text{trend}} > +0.75$: **`INCREASING`**
  * $\Delta_{\text{trend}} < -0.75$: **`DECREASING`**
  * $|\Delta_{\text{trend}}| \le 0.75$: **`STABLE`**

### 2. Status Recommendation Matrix
```
Level: HIGH   + Trend: INCREASING  -->  "CONGESTION BUILDING"
Level: HIGH   + Trend: DECREASING  -->  "CONGESTION CLEARING"
Level: HIGH   + Trend: STABLE      -->  "SUSTAINED HIGH TRAFFIC"
Level: MEDIUM + Trend: INCREASING  -->  "MONITOR - BUILDING"
Level: MEDIUM + Trend: DECREASING  -->  "MONITOR - CLEARING"
Level: MEDIUM + Trend: STABLE      -->  "MONITOR"
Level: LOW    + Any Trend          -->  "NORMAL FLOW"
```

### 3. Priority Zone Selection Algorithm
When multiple zones experience congestion, the priority zone is elected deterministically using a multi-attribute sorting key:
$$\text{Priority} = \arg\max_{z \in \text{Zones}} \Big( \text{Score}_{\text{level}}(z), \; \text{Score}_{\text{trend}}(z), \; \overline{\text{Count}}_{10s}(z), \; T_{\text{sustained}}(z) \Big)$$
Where:
* $\text{Score}_{\text{level}} \in \{ \text{LOW}: 0, \text{MEDIUM}: 1, \text{HIGH}: 2 \}$
* $\text{Score}_{\text{trend}} \in \{ \text{DECREASING}: 0, \text{STABLE}: 1, \text{INCREASING}: 2 \}$
* $\overline{\text{Count}}_{10s}$ is the 10-second moving average vehicle volume.
* $T_{\text{sustained}}$ is the continuous duration in seconds the zone has remained at `HIGH`.

---

## F. Existing SUMO / TraCI Simulation Setup

### Current SUMO Infrastructure
* **SUMO Version:** 1.27.1
* **Simulation Configuration:** `sumo/simulation.sumocfg` executes a 600-second simulation linking `intersection.net.xml` and `routes.rou.xml`.
* **Current Network (`sumo/intersection.net.xml`):** A 2x2 grid junction network with 4 nodes ($A0, A1, B0, B1$) generated via `netgenerate`.
* **Target Network (`sumo/final_network/intersection.nod.xml`):** A 4-way intersection specification with a central traffic light node (`center` at $0, 0$, type `traffic_light`) and 4 arterial arms (`west`, `east`, `north`, `south` at $300\text{m}$ offset, type `priority`).

```
                   North (0, 300)
                         │
                         ▼
West (-300, 0) ──► Center [TLS] (0, 0) ◄── East (300, 0)
                         ▲
                         │
                   South (0, -300)
```

* **Demand Flows (`sumo/routes.rou.xml`):**
  * `traffic_1`: 300 veh/hr
  * `traffic_2`: 250 veh/hr
  * `traffic_3`: 300 veh/hr
  * `traffic_4`: 250 veh/hr
* **TraCI Interface:** Python can communicate with SUMO using the TraCI protocol to dynamically inspect edge queues and adjust traffic light phase durations.

---

## G. Existing Frontend Status & Architecture

### Findings
1. **No Web Frontend Currently Exists:**  
   There are no HTML, CSS, JavaScript, React, Vue, or frontend server files in the repository. The only `.json` file is `analysis_report.json`.
2. **Current Interface Is Purely Desktop OpenCV:**  
   Both `authority_dashboard.py` and `traffic_decision_engine.py` render directly to an X11/Win32 desktop window using `cv2.imshow()`.
3. **Reusable Desktop Logic:**  
   `authority_dashboard.py` and `traffic_decision_engine.py` contain well-structured rendering parameters (colors, layouts, bounding boxes, text placement, rolling statistics) that can immediately be repurposed to build:
   * A high-performance Python backend (FastAPI/Flask) serving real-time telemetry and annotated video streams.
   * A state-of-the-art Web Dashboard featuring modern glassmorphism, responsive zone telemetry cards, live signal status, and interactive controls.

---

## H. What Is Working

The following components are verified and operational:
1. **YOLOv8 + ByteTrack Detection and Tracking:** Successfully executed on `traffic2.mp4`; full detection logs stored in `traffic_tracks.csv`.
2. **Trajectory Smoothing & Pixel Speed Analytics:** `traffic_speed.py` correctly generates `traffic_smoothed_movement.csv` and `traffic_smoothed_summary.csv`.
3. **Ground-Plane Calibration:** Perspective homography correctly maps road bounds to a planar bird's-eye view (`perspective_matrix.npy`, `bird_eye_view.jpg`).
4. **Clean Ground Motion Filtering:** `clean_ground_motion.py` cleans coordinate outliers and produces `clean_ground_motion.csv`.
5. **Vehicle Approach/Closing Motion Classification:** `vehicle_closing_motion.py` processes ground trajectories into stationary/approaching/receding states.
6. **Multi-Zone Spatial System:** All 6 zones are cleanly defined in numpy arrays and verified against the 1080p frame space.
7. **Zone Statistics & Flow Calculation:** `zone_traffic_analysis.py` computes vehicle class distribution and 10s flow rates.
8. **Congestion Scoring Engine:** `congestion_engine.py` computes composite scores and exports to `congestion_analysis.csv`.
9. **Pattern Detection & Policy Generation:** `traffic_analyzer.py` and `traffic_policy.py` run cleanly without errors.
10. **OpenCV Authority Dashboards:** `authority_dashboard.py` and `traffic_decision_engine.py` contain functional real-time rendering logic.

---

## I. What Should Be Preserved

To maintain stability and avoid redundant compute:
1. **Preserve All Pre-computed CSV Datasets:** `traffic_tracks.csv` contains 29,139 verified detections. Re-running YOLOv8m on 1530 frames of 1080p video takes several minutes; keeping this dataset enables instant playback and fast iteration.
2. **Preserve Zone Definition Arrays:** `lane_zone_points.npy` and `zone_2_points.npy` through `zone_6_points.npy` represent precise manual calibrations matching `traffic2.mp4`.
3. **Preserve Calibration Arrays:** `calibration_points.npy` and `perspective_matrix.npy` are verified.
4. **Preserve Decision & Congestion Heuristics:** The mathematical formulas, moving average windows, threshold levels (LOW/MEDIUM/HIGH), and trend criteria must remain intact.
5. **Preserve Working Scripts:** No scripts will be deleted or rewritten with dummy code.

---

## J. What Needs to Be Built Next

1. **Modern Web-Based Authority Dashboard:**  
   Replace reliance on local `cv2.imshow` windows with a browser-based dashboard powered by a lightweight Python backend (FastAPI), providing real-time telemetry, zone cards, trend charts, and MJPEG/Canvas video streaming.
2. **SUMO 4-Way Intersection Digital Twin:**  
   Compile `sumo/final_network/intersection.nod.xml` into a complete single 4-way intersection (`intersection.net.xml`) with edges and connections.
3. **TraCI Real-Time Bridge:**  
   Build a bridge script that translates live CV-detected zone counts directly into SUMO traffic injection rates and executes adaptive signal timing.
4. **Adaptive Signal Control Experiment:**  
   Implement a benchmark comparing fixed-time signal cycles against CV-actuated dynamic green splits, recording queue lengths, delays, and vehicle throughput.
5. **Dedicated Driver Mode Module:**  
   Create a driver-perspective interface with clear scientific boundaries (using ego-vehicle reference frames for distance closure and tailgating warnings).
6. **Unified Management Hub:**  
   A centralized launcher allowing seamless toggling between Authority Mode and Driver Mode with presentation-ready metrics.

---

## K. Consolidated End-to-End System Architecture (Phase 10 Final)

```
                 OPTICAL CCTV VIDEO / REAL-WORLD SENSORS
                               │
                               ▼
                   YOLOv8m VEHICLE DETECTION
                               │
                               ▼
                     BYTETRACK OBJECT TRACKING
                               │
                               ▼
                  20-FEATURE KINEMATIC EXTRACTION
                               │
                               ▼
                    20-STEP ROLLING FIFO BUFFER
                               │
                               ▼
                  FROZEN FEATURE SCALER (JOBLIB)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │ PROPOSED TCN-TRANSFORMER GATED HYBRID MODEL  │
        │             (226,937 Parameters)             │
        ├──────────────────────┬───────────────────────┤
        │  Dilated Causal TCN  │ Multi-Head Attention  │
        │  (Residual Blocks)   │  (2-Layer Transformer)│
        └──────────────┬───────┴───────────────┬───────┘
                       │                       │
                       └───────────┬───────────┘
                                   ▼
                           GATED FUSION UNIT
                     (Learned Sigmoid Combination)
                                   │
                                   ▼
                         10 MULTI-TASK HEADS
        (Congestion, Risk, Approaching, Motion, Infractions)
                                   │
                                   ▼
                      TRAFFIC INTELLIGENCE ENGINE
                (Strict Measured / Predicted / Derived)
                     /             │            \
                    ▼              ▼             ▼
               CONGESTION     VIOLATIONS       ALERTS
                    \              │            /
                     └─────────────┼───────────┘
                                   │
                        UNIFIED FLASK REST API
                         /                 \
                        ▼                   ▼
                 AUTHORITY MODE        DRIVER MODE
                 (OPERATOR HUD)        (COMMUTER HUD)
                        \                   /
                         \                 /
                          ▼               ▼
                 ECLIPSE SUMO 1.27.1 / TRACI CO-SIMULATION
                                  │
                                  ▼
                     CLOSED-LOOP CONTROL POLICY
                   (VSL & Dynamic Rerouting Engine)
                                  │
                                  ▼
                            ECLIPSE SUMO
                    (New Physical State Iteration)
```

### Data Provenance Isolation Rules
* **MEASURED:** Ground-truth sensors (speeds, tracking bounding boxes, physical lane counts).
* **PREDICTED:** Inferences from the Proposed TCN-Transformer Gated Hybrid model.
* **DERIVED:** Analytical formulas combining multi-signal inputs.
* **SIMULATED:** Telemetry generated strictly by the Eclipse SUMO simulation environment.
