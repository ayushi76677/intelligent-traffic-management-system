# Intelligent Traffic Management System (Emerge)
**Multi-Modal Traffic Intelligence with Proposed TCN-Transformer Gated Hybrid Architecture & SUMO Closed-Loop Co-Simulation**

[![System Status](https://img.shields.io/badge/System%20Status-FINAL%20INTEGRATION%20VERIFIED-brightgreen.svg)]()
[![Regression Tests](https://img.shields.io/badge/Tests-95%2F95%20PASS%20(100%25)-success.svg)]()
[![Model Parameters](https://img.shields.io/badge/Model%20Params-226%2C937-blue.svg)]()
[![Inference Speed](https://img.shields.io/badge/Inference%20Throughput-24.7%20FPS-purple.svg)]()
[![Simulation Engine](https://img.shields.io/badge/Co--Simulation-Eclipse%20SUMO%201.27.1-orange.svg)]()

---

## 1. Project Purpose

The **Intelligent Traffic Management System (Emerge)** is a multi-modal urban traffic monitoring, predictive intelligence, and autonomous control platform. Designed for complex arterial intersections, the platform fuses optical camera tracking, deep spatial-temporal neural networks, dual-mode operational dashboards (Authority & Driver), and microscopic simulation to observe, predict, and proactively mitigate traffic congestion and road hazards.

---

## 2. Core Architectural Pillars

```
                 OPTICAL CCTV VIDEO / LIVE FEED
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
        └──────────────┬───────┴───────────────┬───────┘
                       │                       │
                       └───────────┬───────────┘
                                   ▼
                           GATED FUSION UNIT
                                   │
                                   ▼
                         10 MULTI-TASK HEADS
                                   │
                                   ▼
                      TRAFFIC INTELLIGENCE ENGINE
                      (Measured / Predicted / Derived)
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
                  (OPERATOR HUD)       (COMMUTER HUD)
                        \                   /
                         \                 /
                          ▼               ▼
                 ECLIPSE SUMO 1.27.1 / TRACI CO-SIMULATION
                                  │
                                  ▼
                     CLOSED-LOOP CONTROL POLICY
                   (VSL & Dynamic Rerouting Engine)
```

---

## 3. Strict Data Mode Demarcation

To prevent artificial data contamination, the system strictly separates operational modalities:

* **TRAINING:** Offline model parameter optimization on 1,530 annotated frames and 34,679 trajectory points using PyTorch. Checkpoint frozen at `checkpoints/best_model.pth`.
* **INFERENCE:** Real-time forward pass execution (24.7 FPS, 16.1 - 17.8 ms model latency) over 20-step rolling trajectory sequences.
* **LIVE TRAFFIC:** Ground-truth optical observations from municipal cameras (CAM-IND-001, Vijay Nagar Crossing, Indore), labeled as `MEASURED`.
* **SIMULATION:** Microscopic synthetic vehicles simulated within Eclipse SUMO 1.27.1 via TraCI, labeled strictly as `SIMULATION DATA`.

---

## 4. Key Components

### 4.1 Proposed TCN-Transformer Gated Hybrid Model
* **Architecture:** Dilated Causal TCN (3 residual blocks) + 2-layer Multi-Head Self-Attention Transformer + Sigmoid Gated Fusion Unit.
* **Parameters:** 226,937 parameters (0.87 MB checkpoint).
* **Multi-Task Outputs:** 10 decoders predicting congestion levels, continuous congestion scores ($10-100$), risk levels, approaching threats, motion states, maneuvers, and infractions.

### 4.2 Traffic Intelligence Engine
* Segregates all telemetry into **MEASURED** (sensor ground truth), **PREDICTED** (ML forward outputs), and **DERIVED** (fused indices and signal recommendations).
* Analyzes 6 spatial polygonal zones covering intersection approaches and inner conflict areas.

### 4.3 Authority Mode Dashboard
* Municipal operator console hosted at `http://127.0.0.1:5000`.
* Dual-engine map (Google Maps JS API + Leaflet fallback).
* HTML5 video playback synchronized with Canvas bounding box rendering.
* Three-Way Congestion Matrix hero card.

### 4.4 Driver Mode HUD
* In-vehicle commuter assistance cockpit.
* Location permission handling with non-intrusive fallbacks.
* Congestion-colored route calculation.
* Proximity hazard alerts and one-click bypass diversion via Eastern Bypass.

### 4.5 SUMO / TraCI Closed-Loop Co-Simulation
* Co-simulation coupling SUMO 1.27.1 to the trained hybrid model.
* TraCI socket bridge extracting per-step vehicle states and dispatching validated Variable Speed Limit (VSL) speed-harmonization and rerouting commands.

---

## 5. Technology Stack

* **Programming Language:** Python 3.13 (AMD64)
* **Deep Learning Framework:** PyTorch 2.6.0+cpu
* **Computer Vision:** Ultralytics YOLOv8m, OpenCV, NumPy 2.2.3, SciPy
* **Web Backend:** Flask 3.1.2, Flask-CORS, Werkzeug
* **Web Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6+), Leaflet 1.9.4, Chart.js
* **Traffic Co-Simulation:** Eclipse SUMO 1.27.1, TraCI 1.27.1, SumoLib

---

## 6. Installation & Configuration

### Prerequisites
* Windows 10/11 or Linux
* Python 3.11+ (Python 3.13 recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/emerge/intelligent-traffic-system.git
cd "Emerge Root00"

# Install Python dependencies
pip install torch torchvision numpy scipy scikit-learn joblib flask flask-cors ultralytics opencv-python eclipse-sumo traci sumolib
```

### Configuration
Optionally configure Google Maps API keys by creating a `.env` file:
```env
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
GOOGLE_MAPS_MAP_ID=DEMO_MAP_ID
```
*(If no API key is provided, the system automatically activates the autonomous Leaflet / OpenStreetMap fallback map engine).*

---

## 7. Running the System

### 7.1 Start Web Server (Authority & Driver Cockpit)
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your web browser.

### 7.2 Run Standalone Closed-Loop Demonstration
```bash
python demo_phase9_closed_loop.py --steps 25
```

### 7.3 Run 600-Second Controlled Simulation Experiments
```bash
python run_phase9_experiments.py
```

---

## 8. Verification & Automated Testing

Execute the complete 95-test automated regression battery:
```bash
python -m unittest discover tests -p "test_*.py"
```

| Test Suite | Focus Area | Tests | Result |
| :--- | :--- | :---: | :---: |
| [`test_tcn_transformer_hybrid.py`](tests/test_tcn_transformer_hybrid.py) | Model Architecture & Fusion Unit | 8 | **PASS** |
| [`test_phase6_realtime_integration.py`](tests/test_phase6_realtime_integration.py) | Real-Time Inference & Buffering | 16 | **PASS** |
| [`test_phase7_authority_mode.py`](tests/test_phase7_authority_mode.py) | Authority Mode Operational Console | 11 | **PASS** |
| [`test_phase8_driver_mode.py`](tests/test_phase8_driver_mode.py) | Driver Mode HUD & Route Routing | 15 | **PASS** |
| [`test_phase9_sumo_traci.py`](tests/test_phase9_sumo_traci.py) | SUMO / TraCI Closed-Loop Control | 17 | **PASS** |
| [`test_phase10_system_integration.py`](tests/test_phase10_system_integration.py) | End-to-End System Integration | 15 | **PASS** |
| [`test_location_search.py`](tests/test_location_search.py) | Arbitrary Indian Location Search | 8 | **PASS** |
| [`test_final_submission_dual_routes.py`](tests/test_final_submission_dual_routes.py) | Dual Routing & Provenance Matrix | 5 | **PASS** |
| **Total Regression Battery** | **Complete System Regression** | **95** | **100% PASS** |

---

## 9. Key Documentation Artifacts

* [`DEMO_GUIDE.md`](DEMO_GUIDE.md): Step-by-step commands to demonstrate every system capability.
* [`MODEL_CARD.md`](MODEL_CARD.md): Formal model card documenting parameters, inputs, outputs, and evaluation metrics.
* [`SYSTEM_LIMITATIONS.md`](SYSTEM_LIMITATIONS.md): Operational boundaries and deployment caveats.
* [`FINAL_PROJECT_REPORT.md`](FINAL_PROJECT_REPORT.md): Comprehensive 24-section technical report.
* [`PHASE10_FINAL_SYSTEM_AUDIT.md`](PHASE10_FINAL_SYSTEM_AUDIT.md): Final integration entry-point audit.

---

## 10. System Limitations & Deployment Boundaries

* **Simulation vs. Real Road:** Microscopic co-simulation in SUMO validates algorithmic control logic; it does not replace municipal hardware-in-the-loop (HIL) safety certification.
* **Camera Calibration:** Speed and metric displacement calculations require accurate camera perspective matrices.
* **Model Generalization:** The model is trained on urban intersection kinematics; OOD conditions (severe fog, uncalibrated expressway cameras) require retraining.
* **Policy Recommendation Disclaimer:** Authority Mode policies are decision support recommendations (`RECOMMENDATION ONLY — NOT REAL-ROAD ACTUATION`) and do not physically modify public road infrastructure.

---

## 11. Final Status

**FINAL STATUS: PASS — FINAL SUBMISSION VERIFIED (95/95 TESTS PASSING)**
