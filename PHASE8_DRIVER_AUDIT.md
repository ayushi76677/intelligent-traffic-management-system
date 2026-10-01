# PHASE 8 AUDIT — DRIVER MODE INTELLIGENT TRAFFIC ASSISTANCE

**Date:** 2026-09-30  
**Status:** AUDIT COMPLETED  
**Objective:** Assess current Driver Mode implementation, frontend state, backend APIs, mapping infrastructure, and identify gaps prior to implementing the Phase 8 real-time driver traffic intelligence interface.

---

## 1. Driver Mode Page & Component Inventory

| Component / File | Current State | Findings / Architecture |
| :--- | :--- | :--- |
| `static/index.html` (`#driverModeView`) | Partially Implemented Overlay | An overlay positioned fixed (`inset: 65px 0 0 0`) containing top bar, speedometer (`#dmEgoSpeed`), speed limit sign (`#dmSpeedLimitSign`), bottleneck warning banner (`#dmSevereBanner`), live hazards list (`#dmHazardsList`), and reroute advisory (`#btnDmAcceptReroute`). **Critical Gap:** No map container exists inside `#driverModeView`. The driver cannot see their vehicle position, road geometry, or route visually. |
| `static/css/dashboard.css` | Basic Cockpit Layout | Styles exist for `.driver-mode-hud-view`, `.dm-top-bar`, `.dm-cockpit-grid`, `.dm-gauge-card`, `.dm-speed-number`, `.dm-limit-sign`, and `.dm-warning-banner`. Needs layout adjustments for an integrated driver map, destination HUD, and responsive mobile layout. |
| `static/js/dashboard.js` | Bare-bones Sync Loop | Function `setupModeSwitching()` toggles `#driverModeView` display and runs `syncDriverModeFeed()` on a 2500ms interval. Reroute button only fires a browser `alert()`. |
| Header Navigation | Present & Functional | Mode switch buttons `#btnModeAuthority` and `#btnModeDriver` toggle between views cleanly. |

---

## 2. Map Implementation Audit

- **Primary Map Engine (Google Maps Platform):**
  - Integrated via `@googlemaps/js-api-loader` on `#googleMap` element in Authority Mode.
  - Initializes `google.maps.Map`, `google.maps.TrafficLayer`, `DirectionsService`, and `DirectionsRenderer`.
  - Attributed with `gmp_git_agentskills_v1`.
- **Secondary / Fallback Engine (Leaflet):**
  - Automatically loads if `GOOGLE_MAPS_API_KEY` is absent or network fails (`initLeafletFallbackMap()`).
  - Renders OpenStreetMap raster tiles, CCTV markers, and zone polygons.
- **Driver Mode Map State:**
  - Currently, when `#driverModeView` is shown, the map in `#mapWorkspaceGrid` is hidden beneath the overlay.
  - Driver Mode must either integrate a dedicated driver map container (`#driverMap`) or share/re-dock the map canvas with driver-centric perspective (driver location marker, heading arrow, route polyline with traffic-tier coloration, destination pin).

---

## 3. Browser Geolocation Audit

- **Current Implementation:** Completely absent (`navigator.geolocation` is not referenced anywhere in `dashboard.js`).
- **Required Implementation:**
  - Must request permission only when entering Driver Mode or upon explicit user action ("Enable Live GPS" / "Track My Location").
  - Do NOT repeatedly prompt or spam permission requests.
  - Handle all 5 standard W3C states:
    1. Permission Granted (`navigator.geolocation.watchPosition` / `getCurrentPosition`).
    2. Permission Denied (`error.PERMISSION_DENIED` -> Fallback to Vijay Nagar corridor center with clear indicator).
    3. Position Unavailable (`error.POSITION_UNAVAILABLE`).
    4. Location Timeout (`error.TIMEOUT`).
    5. Geolocation Unsupported in browser.
  - Maintain driver position with latitude, longitude, heading (if provided by sensor/computed from movement delta), speed, and accuracy radius.
  - Never log or leak raw location data to unauthorized external endpoints.

---

## 4. Route Calculation & Evaluation Audit

- **Existing Modal (`#findRouteModal`):**
  - Provides inputs for `#routeStartInput` and `#routeDestInput` and a button `#btnCalculateBestRoute`.
  - Uses `directionsService.route()` with `google.maps.TravelMode.DRIVING`.
  - Extracts path coordinates and posts to `/api/routes/evaluate`.
- **Backend Route Evaluator (`/api/routes/evaluate` in `app.py`):**
  - Accepts `start`, `destination`, and `points` (or path coordinates).
  - Evaluates points against severe congestion zones (`congestion_level in ['SEVERE', 'CONGESTED']`) and active incidents (`status == 'ACTIVE'`).
  - Emits:
    - `severe_congestion_detected`: boolean.
    - `warning_headline`: string.
    - `warnings`: list of warning strings.
    - `impacted_zones`: list of zone objects.
    - `impacted_incidents`: list of incident objects.
    - `recommended_action`: diversion recommendation (e.g. "Reroute via Eastern Bypass / Ring Road").
    - `estimated_delay_minutes`: estimated delay.
- **Routing Gaps in Driver Mode:**
  - Route calculation is currently trapped in an administrative modal dialog; not built into the driver cockpit workflow.
  - When Leaflet is active, `directionsService` fails because Google Maps is not loaded; there is no offline route calculation or local corridor waypoint interpolation.
  - No colored traffic-tier route segment display (Normal / Congested / Severe / Incident).

---

## 5. Traffic, Congestion, Hybrid Model, and Alert APIs Audit

| Endpoint | Method | Response Fields | Driver Mode Relevance |
| :--- | :--- | :--- | :--- |
| `GET /api/driver/feed` | GET | `service`, `current_road`, `current_speed_limit_kmh`, `intersection_speed_limit_kmh`, `nearby_severe_zones`, `active_road_hazards`, `safety_advisory`, `recommended_divert`, `hybrid_intelligence` | Primary Driver Mode feed. Needs query parameter support for driver `lat`, `lng`, and `dest` to calculate actual proximity distances. |
| `POST /api/routes/evaluate` | POST | `severe_congestion_detected`, `warnings`, `impacted_zones`, `impacted_incidents`, `recommended_action`, `estimated_delay_minutes` | Used to evaluate driver routes. |
| `GET /api/alerts` | GET | `total`, `alerts: [{alert_id, type, title, severity, road, timestamp, message}]` | Combines High Speed (violations), Severe Congestion, Road Hazards, and Hybrid Model Risk Alerts. Needs proximity filtering for driver relevance. |
| `GET /api/hybrid/live` & `GET /api/hybrid/status` | GET | Sequence buffer warmup, system fused scores, priority zone recommendations. | Source of truth for model-derived vs empirical separation. |
| `GET /api/zones` | GET | 6 calibrated zones (`Z1`–`Z6`) with polygons and congestion scores. | Used for zone proximity and corridor corridor traffic state. |
| `GET /api/cameras` | GET | 13 CCTV nodes with coordinates and operational status. | Spatial anchor points across Indore corridors. |

---

## 6. Frontend State Management Audit

- **Global Variables in `static/js/dashboard.js`:**
  - `gMap`, `leafletMap`, `isLeafletActive`, `directionsService`, `directionsRenderer`.
  - `allCameras`, `allZones`, `allIncidents`, `allViolations`, `allAlerts`.
  - `driverModeInterval`: Set to 2500ms when Driver Mode opens.
- **Driver State Gaps:**
  - Missing driver location state: `driverLocation = { lat, lng, heading, speed, accuracy, source: 'GPS' | 'FALLBACK' }`.
  - Missing driver destination state: `driverDestination = { name, lat, lng }`.
  - Missing active route state: `driverActiveRoute = { geometry, distance, duration, congestion_level, segments, alternative }`.
  - Missing real-time change detector / movement threshold (`lastEvaluatedLocation`, movement threshold > 100m).
  - Missing layer controls for Driver Map (driver marker, destination marker, route path, nearby alerts).

---

## 7. Plan of Action for Phase 8

1. **Step 2 (Location Permission):** Implement non-intrusive Geolocation manager supporting Granted, Denied, Unavailable, and Timeout with a prominent GPS Status Pill.
2. **Step 3 & 4 (Current Location & Map):** Implement `#driverMapContainer` with dual engine support (Google Maps + Leaflet fallback), featuring a high-contrast vehicle marker with heading orientation and road geometry.
3. **Step 5 (Nearby Traffic):** Query `/api/driver/feed` with driver coordinates; filter zones, incidents, and congestion within proximity radius (< 2.5 km).
4. **Step 6 & 7 (Routing & Traffic-Aware Evaluation):** Implement clean routing engine supporting both Google Directions API and an autonomous local Indore road network graph for Leaflet fallback, evaluated against `/api/routes/evaluate`.
5. **Step 8 (Route Display):** Render `CURRENT LOCATION -> ROUTE -> DESTINATION` with color-coded traffic condition segments (Green, Amber, Red).
6. **Step 9 (Real-Time Route Monitoring):** Implement threshold-based route re-evaluation (only on > 100m movement or significant corridor status change).
7. **Step 10 & 11 (Driver Alerts & Violations):** Filter alerts for driver route relevance, categorizing as MEASURED (empirical radar/sensor) vs MODEL-DERIVED (TCN-Transformer trajectory risk).
8. **Step 12 & 13 (Driver UI & Destination Input):** Refactor `#driverModeView` with clean driver HUD, search input, quick presets (Radisson, Palasia, Airport, Bhanwarkua), ETA, distance, and responsive CSS.
9. **Step 14 & 15 (Backend & Performance):** Optimize `/api/driver/feed` to accept driver position; preserve 24.7 FPS baseline.
10. **Step 16 (Safety/Uncertainty):** Provide transparent tags (`LIVE DATA`, `ESTIMATED`, `MODEL-DERIVED`, `CORRIDOR BASELINE`).
11. **Step 17 & 18 (Testing & Report):** Automated test suite `tests/test_phase8_driver_mode.py` and `PHASE8_DRIVER_MODE_REPORT.md`.
