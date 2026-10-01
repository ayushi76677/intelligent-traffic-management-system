# SYSTEM LIMITATIONS & DEPLOYMENT BOUNDARIES
**Project:** Intelligent Traffic Management System (Emerge)  
**Document:** `SYSTEM_LIMITATIONS.md`  
**Phase:** Phase 10 — Final System Integration & Project Demonstration  
**Status:** Formal Engineering Specification  

---

## 1. Overview & Purpose

This document provides a rigorous, transparent account of the engineering boundaries, operational assumptions, and technical limitations of the Intelligent Traffic Management System. These constraints delineate the prototype and simulation validation from real-world road deployment.

---

## 2. Microscopic Simulation vs. Real-World Physical Actuation

1. **Simulation Abstraction:**
   * Eclipse SUMO (v1.27.1) provides a deterministic/stochastic car-following environment governed by the Krauss model. While SUMO captures microscopic lane-changing and acceleration dynamics, it does not model real-world vehicle breakdowns, pedestrian jaywalking, road debris, dynamic weather friction, or unexpected human non-compliance.
2. **Simulation Telemetry Distinction:**
   * Telemetry extracted via TraCI (`set_edge_max_speed`, `rerouteTraveltime`) represents synthetic simulation states. Measured metrics from SUMO (e.g. 237 speed harmonization actions) demonstrate closed-loop control mechanics under simulation conditions and **are not equivalent to real-world road validation**.
3. **Safety Authorization:**
   * Any actuation on physical municipal roadways requires regulatory approval from transport authorities, hardware-in-the-loop (HIL) safety interlocks, manual override failsafes, and adherence to regional traffic control standards (e.g., NEMA, 170/2070, or IRC standards).

---

## 3. Machine Learning Model & Distributional Assumptions

1. **Training Distribution Scope:**
   * The Proposed TCN-Transformer Gated Hybrid model (`checkpoints/best_model.pth`, 226,937 parameters) was trained on 1,530 frames and 34,679 trajectory points collected from the Vijay Nagar Indore arterial corridor.
   * Model inference reliability is bounded by this distribution. Extreme weather (dense monsoon fog, heavy rain, water logging), night-time glare, high-speed expressways (>100 km/h), or multi-lane roundabouts represent out-of-distribution (OOD) scenarios.
2. **Temporal Warm-up Horizon:**
   * The model requires an unpadded rolling history of $T=20$ timesteps (~0.67s at 30 FPS, or 20s at 1 Hz). For newly detected vehicles ($t < 20$), predictions cannot be issued and the system reports `WARMING_UP`.
3. **Multi-Task Class Imbalance:**
   * Extreme traffic infractions (e.g. wrong-way driving) are inherently rare events in observational datasets. While class-weighted loss functions were applied, rare-class predictions carry higher epistemic uncertainty and must be corroborated by physical tracking before enforcement actions are taken.

---

## 4. Computer Vision, Tracking & Calibration Dependencies

1. **Detection Quality:**
   * The upstream object detector (YOLOv8m) is subject to optical challenges: partial vehicle occlusions behind heavy trucks, lens smudges, and shadows. Detector false negatives propagate downstream into fragmented track IDs.
2. **Homography & Perspective Calibration:**
   * Metric ground-plane coordinates $(X, Y)$ and velocities depend on the perspective transformation matrix (`perspective_matrix.npy`). If the camera is moved, bumped, or vibrates due to high wind, the homography matrix must be recalibrated; otherwise, metric speed calculations will exhibit proportional scale errors.
3. **Track Association & Re-Identification:**
   * Fast-moving vehicles crossing occluding objects may trigger track ID switches, causing a reset of the rolling sequence buffer back to `WARMING_UP`.

---

## 5. Mapping, Geocoding & Routing Externalities

1. **External API Dependencies:**
   * Driver Mode routing utilizes external mapping services (Google Maps JavaScript API, Directions API, and Geocoding API) with Leaflet and OpenStreetMap as autonomous fallbacks.
   * Routing times and geocoding accuracy depend on external network connectivity and valid API credentials. In offline fallback mode, routing utilizes local road geometry heuristics.
2. **Browser Geolocation Permissions:**
   * Driver Mode location accuracy depends on client hardware (GPS/Wi-Fi positioning) and user-granted browser permissions. When permissions are denied or unavailable, the system safely falls back to default corridor coordinates.

---

## 6. Real-World Deployment Requirements

To transition this system from a research and co-simulation prototype to physical deployment, the following steps are mandatory:
1. **Multi-Camera Edge Deployment:** Transition from single-node edge processing to distributed camera sensor fusion across intersecting corridors.
2. **Hardware-in-the-Loop (HIL) Actuation:** Connect TraCI control policies to physical signal controllers (e.g., TS2 or 2070 cabinets) with hardware conflict monitors.
3. **Extensive Field Testing:** Shadow-mode deployment across seasonal weather variations and lighting conditions before enabling automated advisory systems.
