# PHASE 7 — AUTHORITY MODE ARCHITECTURAL AUDIT
**Date:** 2026-09-30  
**Status:** COMPLETE & VERIFIED  
**System:** Smart Traffic Management System — Authority Command & Control  

---

## 1. Executive Summary

This audit assesses the existing Authority Mode frontend and backend infrastructure in preparation for integrating the Phase 6 verified real-time TCN-Transformer Gated Hybrid traffic intelligence system. In accordance with strict guidelines, no duplicate dashboards will be created; instead, the existing Authority Mode web architecture is analyzed, preserved, and targeted for surgical intelligence enhancement.

---

## 2. Authority Mode Entry Point & Application Flow

### 2.1 Backend Entry Point
- **Server:** [`app.py`](file:///c:/Users/ayush/Emerge%20Root00/app.py) running Flask.
- **Route `/`**: Serves [`static/index.html`](file:///c:/Users/ayush/Emerge%20Root00/static/index.html).
- **Core Controller:** [`static/js/dashboard.js`](file:///c:/Users/ayush/Emerge%20Root00/static/js/dashboard.js) loaded as an ES module via `@googlemaps/js-api-loader`.
- **Styling:** [`static/css/dashboard.css`](file:///c:/Users/ayush/Emerge%20Root00/static/css/dashboard.css) and [`static/css/leaflet.css`](file:///c:/Users/ayush/Emerge%20Root00/static/css/leaflet.css).

### 2.2 Mode Switching
- Header toggle group (`#btnModeAuthority` and `#btnModeDriver`) allows seamless switching between:
  1. **Authority Mode (Default):** Complete command center with GIS map, CCTV sensor inspection, zone calibrations, real-time CV video canvas with bounding boxes, congestion matrix, charts, and alert drawer.
  2. **Driver Mode:** Cockpit HUD synchronized to the authority backend telemetry via `/api/driver/feed`.

---

## 3. Map Implementation & Geographic Road Data

### 3.1 Map Provider & Architecture
- **Primary Provider:** Google Maps JavaScript API via `@googlemaps/js-api-loader` (`weekly` channel, vector/raster base map support).
- **Fallback Capability:** Local Leaflet.js runtime (`static/js/leaflet.js` & `static/css/leaflet.css`) present in workspace for network-resilient offline operation.
- **Map Container:** `<div id="googleMap" class="real-google-map"></div>` in [`static/index.html`](file:///c:/Users/ayush/Emerge%20Root00/static/index.html#L319).
- **Map Layers:**
  - Base layers: Roadmap (`btnMapTypeRoadmap`), Satellite (`btnMapTypeSatellite`), Terrain (`btnMapTypeTerrain`).
  - Native Google Maps TrafficLayer (`gTrafficLayer = new TrafficLayer()`) with live congestion overlays.
  - Interactive AdvancedMarkerElement markers for municipal CCTV cameras and traffic incidents.
  - Interactive Google Maps Polygons for monitored traffic zones (`ZONE 1` to `ZONE 6`).
  - DirectionsService / DirectionsRenderer for route calculations and congestion evaluation.

### 3.2 Location Selection
- Search bar `#gmpSearchInput` with Google Places Autocomplete (`AutocompleteSuggestion` API, country restricted to `['in']`).
- Quick-selection pills for Indian junctions:
  - Indore: Vijay Nagar (Primary Camera / Ground Truth Benchmark)
  - Bhopal & Bhopal Junction
  - Indore: MG Road
  - Delhi: ITO
  - Bengaluru: Central Silk Board
  - Mumbai: Western Express Highway (Andheri)
- Interactive Map Clicks (`handleMapLocationClick(lat, lng)`):
  - Drops a location marker (`🎯`).
  - Queries server-side geocoding proxy (`/api/geocode?lat=...&lng=...`).
  - Updates location card (`#selLocationName`, `#selFormattedAddress`, `#selLat`, `#selLng`, `#roadGeocodeText`).
  - Automatically identifies and activates the nearest CCTV camera node.

### 3.3 Data Source Classification
To prevent fabricating traffic data where no live source exists, cameras and locations are categorized into:
- **`LIVE DATA`**: `CAM-IND-001` (Vijay Nagar, Indore) connected to verified `traffic2.mp4` ground-truth feed with 1530 precomputed frames and Phase 6 hybrid inferences.
- **`LOCATION ONLY`**: Municipal ITMS sensor coordinates deployed at junction without active video stream.
- **`SIMULATION DATA`**: Eclipse SUMO 1.27.1 / TraCI digital twin simulated corridor.
- **`DEMO DATA` / `UNAVAILABLE DATA`**: Network-offline or uncalibrated junction points.

---

## 4. Live Vehicle & Video Canvas Tracking

### 4.1 Video & Canvas Elements
- **Video:** `<video id="trafficVideo" src="/video/traffic2.mp4">` with byte-range HTTP 206 streaming (`app.py:stream_video`).
- **Canvas:** `<canvas id="cvCanvas">` positioned directly over the video frame with automatic scale synchronization (`scaleX`, `scaleY`).
- **Playback Controls:** Play/Pause, Frame Scrubber (0–1529 frames), Step Back/Forward (+/- 30 frames), Playback Speed (0.5x, 1x, 2x).
- **Video HUD:** Live elapsed time, current frame counter (`Frame X / 1530`), and current priority zone tag.

### 4.2 Existing Vehicle Visualization
- In `drawCanvasOverlay(data)`:
  - Bounding boxes: `ctx.strokeRect(x1, y1, w, h)`.
  - Track ID & class tags: `#<track_id> <CLASS>` (e.g. `#7 CAR`, `#12 MOTORCYCLE`).
  - Road contact points: `ctx.arc(bottom_x, bottom_y, 3, 0, 2*PI)` in rose red.
  - Zone polygons: Rendered on canvas in green/yellow/red with centroid labels and current vehicle counts.

### 4.3 Phase 7 Enhancement Target
- Color-code vehicle bounding boxes by **Hybrid Risk State** (Green = Safe/Compliant, Yellow = Warning/Approaching, Red = High Risk / Active Infraction).
- Display hybrid status indicators: Model Confidence, Prediction Readiness (`READY` vs `WARMING_UP`), and Infraction Badges.

---

## 5. Congestion Visualization Architecture

### 5.1 Existing Congestion Engine
- Evaluates 6 physical intersection zones defined by `.npy` polygon files (`lane_zone_points.npy`, `zone_2_points.npy` through `zone_6_points.npy`).
- Displays ground-truth congestion metrics in the hero card (`.congestion-hero-card`):
  - Congestion Score ($0-100$)
  - Level (`NORMAL`, `BUSY`, `CONGESTED`, `SEVERE`)
  - Total Vehicles, Peak Interval, Average Interval, Density Score, Flow Score, Variation.

### 5.2 Phase 7 Enhancement Target: Three-Way Separation
To strictly distinguish **MEASURED**, **MODEL PREDICTED**, and **DERIVED** data:
1. **MEASURED (Empirical):** Exact vehicle count, density score, average pixel velocity, vehicle class breakdown.
2. **MODEL PREDICTED (TCN-Transformer Gated Hybrid):** Mean predicted congestion score, risk level distribution (`SAFE`, `WARNING`, `HIGH`), approaching vehicle threat score, predicted maneuver/motion.
3. **DERIVED (Fused Intelligence):** Fused congestion score ($S_{fused} = 0.5 \cdot S_{base} + 0.5 \cdot S_{pred}$), adaptive actuation recommendations, priority zone election.

---

## 6. Traffic Intelligence & Zone Analysis Panel

### 6.1 Monitored Intersection Zones
- 6 calibrated zones rendered as cards in `#zoneCardsGrid`:
  - `ZONE 1`: Main Approach Corridor (North Inflow)
  - `ZONE 2`: Eastbound Exit / Turning Lane
  - `ZONE 3`: Northbound Core Crossing
  - `ZONE 4`: Southbound Inflow Queue
  - `ZONE 5`: Westbound Left Approach
  - `ZONE 6`: Southeast Exit Link
- Each card shows: Current Count, Rolling 10s Average, Peak, Level Pill, Progress Bar, Trend, and Flow Recommendation.

### 6.2 Decision Engine & Actuation
- `#decisionLevelBadge`: Dynamic traffic level (`HIGH TRAFFIC`, `BALANCED`).
- Priority Zone Election: Focuses on highest congestion/risk approach.
- Signal Actuation Advisory: Emits adaptive split recommendations (e.g. `Extend Green +15s`, `Hold Red Demand Priority`).

---

## 7. Alerts & Violations Engine

### 7.1 Backend Alerts API (`/api/alerts`)
Merges:
1. **Overspeeding Violations:** From `TrafficViolationEngine` (e.g. `VEH_2 clocked at 74.8 km/h`).
2. **Severe Congestion Alerts:** Zones exceeding score 80 or tagged `SEVERE`.
3. **Active Road Incidents:** Reported road blocks, accidents, and hazards.
4. **Hybrid Model Trajectory & Infraction Alerts:** Dynamic alerts emitted when vehicles exhibit high risk or predicted infractions.

### 7.2 Frontend Alert Components
- Floating drawer `#alertsDrawer` with expandable alert cards.
- Violations modal `#violationsModal` with full tabular registry of infractions.

---

## 8. Charts & Telemetry Buffering

### 8.1 Chart.js Visualizations
- `chartZoneVolumes`: Bar chart comparing current vehicle counts vs. 10s rolling averages across Z1–Z6.
- `chartModalSplit`: Doughnut chart illustrating fleet composition (Motorcycles, Cars, Trucks, Buses).
- `chartFlowTimeline`: Historical timeline of vehicle flow intervals.
- `chartCongestionRadar`: Radar plot of density, flow, variation, and congestion indices.

### 8.2 Telemetry Buffering
- `fetchTelemetryBatch(start, count)`: Buffers frames ahead in `telemetryCache` (Map) for instantaneous frame seeking and rendering without network lag.

---

## 9. API Schema Summary & Verification

| Endpoint | Method | Response Schema Summary | Status |
| :--- | :--- | :--- | :--- |
| `/api/status` | GET | `status`, `video`, `zones_count`, `active_models`, `hybrid_model` | Verified |
| `/api/hybrid/status` | GET | `status`, `checkpoint`, `feature_count`, `scaler`, `telemetry` | Verified |
| `/api/hybrid/frame/<num>` | GET | `frame`, `vehicle_level`, `lane_level`, `zone_level`, `system_level` | Verified |
| `/api/alerts` | GET | `total`, `alerts: [{alert_id, type, severity, speed_kmh, road, ...}]` | Verified |
| `/api/driver/feed` | GET | `service`, `nearby_severe_zones`, `active_road_hazards`, `hybrid_intelligence` | Verified |
| `/api/zones` | GET/POST | `total`, `zones: [...]`, `pixel_polygons: {...}` | Verified |
| `/api/cameras` | GET/POST | `total`, `cameras: [{camera_id, feed_status, road, ...}]` | Verified |
| `/api/violations` | GET/POST | `total`, `violations: [{violation_id, type, speed_kmh, ...}]` | Verified |

---

## 10. Audit Conclusion & Next Steps

The existing Authority Mode frontend is robust, well-structured, and completely free of artificial placeholders. The map, canvas overlay, telemetry buffering, and modals are fully operational.
In Steps 2–13, we will integrate Phase 6 hybrid intelligence into:
1. **Vehicle Markers & Canvas Overlay:** Risk-colored bounding boxes and prediction tags.
2. **Congestion Engine:** Explicit 3-way separation of MEASURED, MODEL PREDICTED, and DERIVED metrics.
3. **Hybrid Intelligence Panel:** Real-time model prediction status, confidence scores, and threat indicators.
4. **Zone Analysis:** Interactive zone inspector displaying lane-level and model predictions for any selected zone.
5. **Violation / High-Speed Highlighting:** Automatic visual emphasis of violating vehicles on the canvas and map.
6. **Data Mode Clarity:** Badging locations as LIVE DATA, SIMULATION DATA, DEMO DATA, or UNAVAILABLE DATA.
