# INTELLIGENT TRAFFIC MANAGEMENT SYSTEM — DEMONSTRATION GUIDE
**Project:** Intelligent Traffic Management System (Emerge)  
**Architecture:** Proposed TCN-Transformer Gated Hybrid Architecture  
**Verification Status:** Verified (95/95 Tests Passing)  
**Submission Date:** October 2026  

---

## 1. Prerequisites & Environment Check

Before initiating the final demonstration, verify the environment and verified model checkpoints:

```bash
# 1. Verify Python & PyTorch environment
python --version
python -c "import torch; print('PyTorch Version:', torch.__version__)"

# 2. Verify frozen model artifacts (MUST REMAIN UNMODIFIED)
python -c "import os; print('Checkpoint Exists:', os.path.exists('checkpoints/best_model.pth'))"
python -c "import os; print('Scaler Exists:', os.path.exists('checkpoints/feature_scaler.joblib'))"

# 3. Verify SUMO microscopic simulator
python -c "import traci, sumolib; print('SUMO Binary:', sumolib.checkBinary('sumo'))"

# 4. Verify complete automated test suite (95 tests)
python -m unittest discover tests -p "test_*.py"
```

---

## 2. Launching the Traffic Management Center

Launch the unified Flask service hosting the REST API, Authority Command Center, Driver Mode HUD, and SUMO co-simulation engine:

```bash
python app.py
```
* **Web Application:** `http://127.0.0.1:5000`
* Open `http://127.0.0.1:5000` in Google Chrome, Microsoft Edge, or Mozilla Firefox.

---

## 3. Four-Part Final Submission Demonstration

### DEMO 1 — AUTHORITY COMMAND CENTER & TIME-VARYING REPLAY

1. **Open Authority Mode:** Navigate to `http://127.0.0.1:5000`. Authority Mode is active by default.
2. **Interactive Map & Roads:** Observe the interactive map displaying monitored junctions, road corridors, traffic zones, and deployed municipal CCTV markers.
3. **Select Monitored CCTV Location:**
   - Click on **Indore - Vijay Nagar Crossing** from the camera list (or search `Vijay Nagar Intersection, Indore`).
   - Observe **Selected Location Card**:
     - `CCTV STATUS: AVAILABLE (ONLINE)`
     - `CURRENT TRAFFIC: AVAILABLE`
     - `PROJECT COVERAGE: ACTIVE (CAM-IND-001)`
     - `DATA PROVENANCE: MEASURED / PREDICTED / DERIVED`
4. **Initiate Time-Varying Video Replay:**
   - In the Computer Vision & Video panel, locate the replay toolbar.
   - Click **[ ▶ PLAY ]** (or press Space / click `#btnExplicitPlay`).
   - Observe the **Video / Replay Time** readout progressing dynamically: `00:00 / 00:51`, `00:15 / 00:51`, etc.
   - Adjust playback rate: **[ 1x ]**, **[ 2x ]**, **[ 4x ]**.
5. **Observe Dynamic Telemetry Updates:**
   - As the video advances across its 1530 frames, watch the dashboard update in real time:
     - **Vehicle Count & Modal Fleet:** Cars, motorcycles, buses, and trucks updated dynamically from ground-truth detections.
     - **Traffic Flow & Density:** Dynamically computed from rolling 10-second approach windows.
     - **Congestion Level:** Congestion scores and levels (`LOW`, `MODERATE`, `HIGH`, `SEVERE`) update per zone.
     - **Active Violations & Alerts:** Real-time overspeeding, wrong-way, and high-risk alerts trigger dynamically.
6. **Inspect Authority Policy Recommendations:**
   - In the **Traffic Policy Recommendations** card (`#policyRecommendationsCard`), observe the real-time condition-action output:
     - **Condition:** e.g. `SEVERE CONGESTION (Score: 85.8) ON ZONE 1 APPROACH`
     - **Recommendation:** `Reroute northbound traffic via Eastern Bypass / MR-10; activate priority green phase extension (+15s)`
     - **Reason:** `Vehicle density on ZONE 1 exceeds corridor exit capacity; TCN-Transformer predicts sustained queue.`
     - **Data Source:** `MEASURED / PREDICTED / DERIVED`
     - **Status Badge:** Explicitly displayed as `RECOMMENDATION ONLY — NOT REAL-ROAD ACTUATION`.
     - **Simulated Control Action:** `Adaptive signal split extension validated in SUMO Digital Twin (+15s green phase)`.
7. **Pause & Restart:**
   - Click **[ ❚❚ PAUSE ]** (`#btnExplicitPause`) to freeze analysis at any moment.
   - Click **[ ↺ RESTART ]** (`#btnExplicitRestart`) to reset replay to frame 0 (`00:00 / 00:51`).

---

### DEMO 2 — LOCATION SEARCH WITHOUT CCTV (ZERO FAKE DATA)

1. **Search Unmonitored Indian Location:**
   - In the global search bar, type:
     ```text
     MG Road, Katihar, Bihar
     ```
     *(Or test `MG Road, Indore`, `Bhopal Junction`, `Delhi ITO`, `Bengaluru Silk Board`, `Mumbai Andheri`).*
   - Press Enter or click Search.
2. **Verify Map & Geocoding:**
   - The map smoothly re-centers on the searched location in Katihar, Bihar (`25.5398° N, 87.5721° E`).
   - A location pin is placed on the road network.
3. **Verify Location Intelligence Card:**
   - The system inspects municipal camera sensor databases and determines that no project CCTV is deployed here.
   - **CCTV STATUS:** `NOT AVAILABLE`
   - **CURRENT TRAFFIC:** `UNAVAILABLE`
   - **PROJECT TRAFFIC COVERAGE:** `NOT AVAILABLE`
   - **LOCATION STATUS:** `ROAD LOCATION IDENTIFIED`
   - **TRAFFIC INTELLIGENCE:** `LIMITED`
   - **Notice Displayed:**
     > *"No connected CCTV or project traffic sensor is currently available for this location. Road geometry and historical network data displayed without fabricated telemetry."*
   - **Typical Traffic Character:**
     > `BUSY / HIGH ACTIVITY (Source: HISTORICAL / AVAILABLE DATA)`
4. **Verify Strict Provenance Matrix:**
   - Map: `AVAILABLE`
   - Road Network: `AVAILABLE`
   - Project CCTV: `NOT AVAILABLE`
   - Live Traffic: `UNAVAILABLE`
   - Historical Data: `AVAILABLE`
   - Model Prediction: `UNAVAILABLE`
   - Simulation (SUMO): `UNAVAILABLE`
5. **Verify Zero Fabrication:**
   - The system displays **NO fake vehicle counts**, **NO fake congestion scores**, **NO fake speeds**, and **NO fake violations**.
   - Monitoring buttons are disabled (`NO CCTV STREAM AVAILABLE`).

---

### DEMO 3 — DRIVER MODE DUAL ROUTING (SHORTEST vs. TRAFFIC-AWARE)

1. **Switch to Driver Mode:**
   - Click the **"DRIVER MODE"** toggle button in the top navigation bar (`#btnModeDriver`).
   - The UI transitions into the in-vehicle commuter Cockpit HUD.
2. **Set Origin & Destination:**
   - Origin: `Vijay Nagar Intersection (Current)`
   - Destination: Select preset chip `Palasia` (or enter `Palasia Square, Indore`).
3. **Calculate Routes:**
   - Click **"Calculate Route"** (`#btnDmCalculateRoute`).
4. **Inspect Route Options & Comparison:**
   - The **Route Choices & Comparison** panel (`#dmRouteOptionsContainer`) appears:
   - **Option 1: SHORTEST ROUTE**
     - Minimizes total physical road distance (`4.1 km`).
     - Estimated Travel Time: `18 mins`.
     - Congestion Exposure: `SEVERE` (direct arterial corridor bottleneck).
     - Traffic Optimization: `NO (Minimizes road distance only)`.
     - Policy Impact: Direct corridor subject to high vehicle queue and signal delay.
   - **Option 2: TRAFFIC-AWARE BEST ROUTE**
     - Minimizes traffic-aware cost: $\text{Cost} = \text{Travel Time} + \text{Congestion Penalty} + \text{Incident Penalty} + \text{Policy Penalty}$.
     - Distance: `4.8 km` (+0.7 km lateral bypass).
     - Estimated Travel Time: `8 mins` (**saves ~10 minutes**).
     - Congestion Exposure: `LOW`.
     - Traffic Optimization: `YES (Traffic-Aware)`.
     - Reason: Lower congestion exposure despite longer physical distance.
     - Policy Impact: Corridor aligned with Authority traffic diversion recommendation.
5. **Inspect Map Visualizations:**
   - Both routes appear rendered on the map:
     - **Shortest Route:** Rendered as an amber/orange dashed line (`#f59e0b`, `8, 8` dash).
     - **Traffic-Aware Best Route:** Rendered as a cyan/blue solid line (`#0284c7`).
6. **Interactive Route Selection:**
   - Click **[ Traffic-Aware ]** toggle (`#btnShowTrafficAwareRoute`): highlights the traffic-aware route and sets active navigation.
   - Click **[ Shortest ]** toggle (`#btnShowShortestRoute`): highlights the shortest route and displays nominal travel times.
   - Click **[ Compare Both ]** toggle (`#btnCompareBothRoutes`): displays both routes side-by-side with full visibility.
   - Click **[ Use Shortest ]** / **[ Selected ]** buttons to activate either preference.
7. **Test Route in Unmonitored Area:**
   - Enter destination: `MG Road, Katihar, Bihar`.
   - Click **"Calculate Route"**.
   - Notice the system displays nominal road distances without fabricated traffic and displays:
     > *"Traffic data unavailable for portions of this route. No connected CCTV or project traffic sensor."*

---

### DEMO 4 — SUMO / TRACI DIGITAL TWIN CO-SIMULATION

1. **Open SUMO Simulation Modal:**
   - In Authority Mode, click the **"SUMO SIMULATION"** button (`#btnOpenSimModal`).
   - The modal displays the Eclipse SUMO 1.27.1 co-simulation console.
2. **Run Baseline Scenario:**
   - Observe baseline intersection flow without active control.
   - Baseline average delay: ~45.2 seconds/vehicle.
3. **Run Controlled Scenario (Proposed Hybrid AI):**
   - Click **"▶ Start Simulation"** (`#btnSimStart`).
   - The closed-loop controller continuously feeds TraCI kinematic vectors into the frozen hybrid model.
   - The safety whitelist validates speed harmonization and green phase extensions.
   - Average controlled delay drops to ~31.8 seconds/vehicle (**~29.6% reduction**).
4. **Verify Provenance Label:**
   - All simulation telemetry is strictly labeled:
     > `SIMULATED CONTROL ACTION (SUMO Digital Twin) — NOT REAL-ROAD ACTUATION`.

---

## 4. Troubleshooting & Verification Reference

| Scenario | System Behavior | Visual Verification |
| :--- | :--- | :--- |
| **No CCTV at Searched Location** | Transparent fallback; historical data cited; no fake numbers | Gray status tags; provenance matrix shows `UNAVAILABLE` |
| **Video Replay Navigation** | Replay time updates from 00:00 to 00:51 | Dynamic vehicle breakdown, KPIs, and alerts change per frame |
| **Dual Route Calculation** | Compares physical shortest vs traffic-aware bypass | Amber dashed line vs Cyan solid line on map |
| **Policy Recommendation** | Actionable advice with non-actuation disclaimer | Badge: `RECOMMENDATION ONLY — NOT REAL-ROAD ACTUATION` |
| **SUMO Simulation** | Validated closed-loop control | Labeled strictly as `SIMULATED` |
